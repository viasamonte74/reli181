"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

Protocol v2: free prompt selection (uniform random with cooldown skip),
M rollouts per prompt at fixed temperature T_PROTO, local reward computation,
Merkle root commitment, HTTP batch submission to validator.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import random as _random

from reliquary.constants import (
    FORCED_SEED_PROTOCOL_VERSION,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    PROMPT_RANGE_SIZE,
)
from reliquary.miner.checkpoint_identity import (
    ActivatedCheckpoint,
    CheckpointIdentityError,
    MinerCheckpointIdentityStore,
    checkpoint_identity_from_state,
    default_checkpoint_identity_path,
)
from reliquary.protocol.profiles import (
    ACTIVE_PROTOCOL_PROFILE,
    to_generation_contract,
    toploc_proof,
)
from reliquary.protocol.toploc_proof import completion_proofs_b64
from reliquary.environment.registry import get_environment_spec
from reliquary.shared.prompt_range import window_prompt_range
from reliquary.infrastructure import chain
from reliquary.protocol.submission import RolloutSubmission

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

logger = logging.getLogger(__name__)

# Between windows the validator answers 503. Windows close on fill and select
# FIFO, so re-poll at the same cadence as a non-OPEN state rather than the
# error backoff; an explicit Retry-After is honoured up to that backoff.
NO_ACTIVE_WINDOW_POLL_SECONDS = 1.0

# First forced rollouts scored before the rest of a Math group is generated.
# One clean correct and one clean wrong answer already clears σ; a long
# unanimous prefix does not, and finishing it would lose the FIFO race.
ZONE_SCREEN_ROLLOUTS = 4
ZONE_SCREEN_LONG_FRACTION = 0.5

# Bounds the fp32 [rows, vocab] block used for proof log-probs so a near-cap
# rollout does not allocate a full-sequence fp32 copy of the logits.
_LOGPROB_WORKSPACE_BYTES = 256 * 1024 * 1024


class CheckpointActivationRestartRequired(RuntimeError):
    """Checkpoint activation cannot safely continue in the current process."""


def _no_active_window_delay(exc: BaseException) -> float:
    from reliquary.constants import POLL_INTERVAL_SECONDS

    retry_after = getattr(exc, "retry_after", None)
    try:
        retry_after = float(retry_after)
    except (TypeError, ValueError):
        return NO_ACTIVE_WINDOW_POLL_SECONDS
    if not math.isfinite(retry_after):
        return NO_ACTIVE_WINDOW_POLL_SECONDS
    return min(
        max(retry_after, NO_ACTIVE_WINDOW_POLL_SECONDS),
        float(POLL_INTERVAL_SECONDS),
    )


def _policy_token_logprobs(
    logits,
    tokens: list[int],
    positions: list[int],
    *,
    workspace_bytes: int = _LOGPROB_WORKSPACE_BYTES,
) -> list[float]:
    """fp32 log-prob of ``tokens[i]`` under ``logits[i - 1]`` for each position.

    ``log_softmax`` is row-independent, so selecting the needed rows before
    the fp32 cast gives the same values as a full-sequence ``log_softmax``
    (the validator also selects rows first). One host transfer per chunk
    replaces a device sync per token.
    """
    import torch

    if not positions:
        return []
    device = logits.device
    rows = torch.tensor([p - 1 for p in positions], device=device, dtype=torch.long)
    targets = torch.tensor(
        [tokens[p] for p in positions], device=device, dtype=torch.long,
    )
    vocab = max(1, int(logits.shape[-1]))
    chunk = max(1, int(workspace_bytes) // (vocab * 4))
    out: list[float] = []
    for start in range(0, len(positions), chunk):
        selected = logits.index_select(0, rows[start:start + chunk]).float()
        chosen = torch.log_softmax(selected, dim=-1).gather(
            1, targets[start:start + chunk].unsqueeze(1),
        )
        out.extend(chosen.squeeze(1).tolist())
    return out


def _population_sigma(rewards: list[float]) -> float:
    mean = sum(rewards) / len(rewards)
    return (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5


def _local_zone_skip_reason(
    *,
    env_name: str,
    rewards: list[float],
    completions: list[list[int]],
    texts: list[str],
    eos_ids,
) -> str | None:
    """Why admission would refuse this locally-scored group before proof.

    Mirrors the validator's pre-proof gates for single-turn environments whose
    reward the miner computes: the truncation allowance, a malformed final
    answer box, then the sigma gate with truncated and unboxed rollouts valued
    at every attainable reward. Returns ``None`` when the group is worth
    proving.
    """
    from reliquary.constants import (
        MATH_ANSWER_FORMAT,
        MAX_TRUNCATED_PER_SUBMISSION,
        MAX_TRUNCATED_PER_SUBMISSION_BY_ENV,
        ROBUST_TRUNCATION_UTILITY_ENABLED,
        SIGMA_MIN,
    )
    from reliquary.validator.boxed_integrity import (
        has_malformed_final_answer,
        is_missing_final_answer_box,
    )
    from reliquary.validator.difficulty_auction import (
        robust_uncertain_reward_utility,
    )

    if len(rewards) < 2:
        return "too_few_rollouts"
    spec = get_environment_spec(env_name)
    eos = {int(t) for t in eos_ids}
    truncated = [
        index for index, completion in enumerate(completions)
        if not any(int(t) in eos for t in completion)
    ]
    allowance = MAX_TRUNCATED_PER_SUBMISSION_BY_ENV.get(
        env_name, MAX_TRUNCATED_PER_SUBMISSION,
    )
    if len(truncated) > allowance:
        return "too_many_truncated"
    boxed_policy = spec.final_answer_policy == "boxed"
    if boxed_policy and any(
        has_malformed_final_answer(reward, text)[0]
        for reward, text in zip(rewards, texts)
    ):
        return "malformed_final_answer"
    unboxed = (
        [index for index, text in enumerate(texts) if is_missing_final_answer_box(text)]
        if boxed_policy and MATH_ANSWER_FORMAT == "boxed"
        else []
    )
    uncertain = list(dict.fromkeys([*truncated, *unboxed]))
    if ROBUST_TRUNCATION_UTILITY_ENABLED and uncertain:
        in_zone = robust_uncertain_reward_utility(
            rewards,
            sigma_min=SIGMA_MIN,
            uncertain_indices=uncertain,
            attainable_rewards=spec.attainable_rewards,
        ) > 0.0
    else:
        sigma = _population_sigma(rewards)
        in_zone = sigma >= 1e-8 and sigma >= SIGMA_MIN
    return None if in_zone else "out_of_zone"


def _prefix_screen_reason(
    *,
    env_name: str,
    rewards: list[float],
    completions: list[list[int]],
    texts: list[str],
    eos_ids,
    max_new_tokens: int,
) -> str | None:
    """Whether the first forced rollouts are worth finishing.

    ``None`` means generate the tail. A hard admission failure is already
    decisive. A long prefix with no clean correct/wrong pair is not: the
    tokens still to generate are unlikely to clear the zone cheaply.
    """
    hard = _local_zone_skip_reason(
        env_name=env_name,
        rewards=rewards,
        completions=completions,
        texts=texts,
        eos_ids=eos_ids,
    )
    if hard in {"too_many_truncated", "malformed_final_answer"}:
        return hard
    eos = {int(token) for token in eos_ids}
    truncated = {
        index for index, completion in enumerate(completions)
        if not any(int(token) in eos for token in completion)
    }
    uncertain = _uncertain_indices(env_name, completions, texts, eos_ids)
    correct = any(
        index not in uncertain and rewards[index] >= 0.5
        for index in range(len(rewards))
    )
    wrong = any(
        index not in uncertain and rewards[index] < 0.5
        for index in range(len(rewards))
    )
    if correct and wrong:
        return None
    mean_length = sum(len(completion) for completion in completions) / len(completions)
    long = bool(truncated) or (
        max_new_tokens > 0
        and mean_length >= ZONE_SCREEN_LONG_FRACTION * max_new_tokens
    )
    if long:
        return "expensive_extreme"
    return None


def _uncertain_indices(env_name: str, completions, texts, eos_ids) -> set[int]:
    """Rollouts whose local reward the validator might score differently:
    no stop token, or no final answer box where the profile requires one."""
    from reliquary.constants import MATH_ANSWER_FORMAT
    from reliquary.validator.boxed_integrity import is_missing_final_answer_box

    eos = {int(token) for token in eos_ids}
    uncertain = {
        index for index, completion in enumerate(completions)
        if not any(int(token) in eos for token in completion)
    }
    spec = get_environment_spec(env_name)
    if spec.final_answer_policy == "boxed" and MATH_ANSWER_FORMAT == "boxed":
        uncertain |= {
            index for index, text in enumerate(texts)
            if is_missing_final_answer_box(text)
        }
    return uncertain


def _unanimous_screen_stops(total: int) -> list[int]:
    """Rollout counts at which a staged group is scored: the zone-screen
    prefix, the unanimous-drop point, then the full group."""
    from reliquary.constants import MINER_UNANIMOUS_DROP_ROLLOUTS

    stops = {min(ZONE_SCREEN_ROLLOUTS, total), total}
    if MINER_UNANIMOUS_DROP_ROLLOUTS > 0:
        stops.add(min(MINER_UNANIMOUS_DROP_ROLLOUTS, total))
    return sorted(stop for stop in stops if stop > 0)


def _structural_termination_kind(
    tokens: list[int],
    prompt_length: int,
    eos_ids,
    cap: int,
) -> str:
    """Admission-shaped termination, before any proof forward.

    ``eos`` is a single trailing stop and still has to match the forced
    pick. ``truncated`` reached the protocol cap with no stop. Anything
    else is ``bad_termination``.
    """
    completion = list(tokens)[int(prompt_length):]
    if not completion:
        return "bad_termination"
    eos = {int(token) for token in eos_ids}
    positions = [
        index for index, token in enumerate(completion) if int(token) in eos
    ]
    if positions:
        if len(positions) != 1 or positions[0] != len(completion) - 1:
            return "bad_termination"
        return "eos"
    total = int(prompt_length) + len(completion)
    if total >= int(cap):
        return "truncated"
    return "bad_termination"


def _termination_upload_skip_reason(
    env_name: str, kinds: list[str | None],
) -> str | None:
    """Why this group must not be uploaded.

    ``None`` means submit. An unclassified rollout fails open: a screen
    bug must not drop a group the validator might have paid.
    """
    from reliquary.constants import max_truncated_for_environment

    if not kinds:
        return None
    for kind in kinds:
        if kind in {"bad_termination", "token_tampered"}:
            return kind
    if any(kind is None for kind in kinds):
        return None
    truncated = sum(kind == "truncated" for kind in kinds)
    if truncated > max_truncated_for_environment(env_name):
        return "too_many_truncated"
    return None


def _proof_termination_kind(
    *,
    tokens: list[int],
    prompt_length: int,
    eos_ids,
    cap: int,
    logits,
    randomness: str,
    prompt_idx: int,
    checkpoint_hash: str,
    rollout_index: int,
    token_logprobs: list[float],
) -> str:
    """Termination the validator's proof will assign to one rollout.

    Uses the proof model's logits, which is the forward the validator
    repeats. A vLLM stop that is not that forward's forced pick is
    ``bad_termination`` and must not be uploaded.
    """
    import math

    from reliquary.constants import (
        FORCED_SEED_ENFORCE,
        MIN_EOS_PROBABILITY,
        PROTOCOL_VERSION,
        TOKEN_AUTH_THRESHOLD,
    )
    from reliquary.environment.forced_sampling import u_at
    from reliquary.validator.verifier import (
        _forced_pick_diagnostics,
        _gpu_p_stop,
    )

    kind = _structural_termination_kind(tokens, prompt_length, eos_ids, cap)
    if kind == "bad_termination":
        return kind
    if token_logprobs and any(
        lp < math.log(TOKEN_AUTH_THRESHOLD) for lp in token_logprobs
    ):
        return "token_tampered"
    if kind != "eos":
        return kind
    if len(tokens) < 2:
        return "bad_termination"
    eos = {int(token) for token in eos_ids}
    completion_length = len(tokens) - int(prompt_length)
    forced, _miss = _forced_pick_diagnostics(
        logits[len(tokens) - 2],
        int(tokens[-1]),
        u_at(
            randomness, prompt_idx, checkpoint_hash, rollout_index,
            completion_length - 1,
        ),
    )
    require_forced = PROTOCOL_VERSION == 6 and FORCED_SEED_ENFORCE
    if require_forced:
        eos_ok = bool(forced)
    else:
        p_stop = _gpu_p_stop(logits, len(tokens), eos, logits.device)
        eos_ok = bool(forced) or (
            p_stop is not None and p_stop >= MIN_EOS_PROBABILITY
        )
    if eos_ok:
        return "ok"
    if int(prompt_length) + completion_length >= int(cap):
        return "truncated"
    return "bad_termination"


class AttemptedPromptJournal:
    """Prompts already drawn for one window, so a restart cannot redraw them.

    The forced stream is fixed by randomness, prompt and checkpoint, and the
    validator rejects that token content for the dedup horizon. The record is
    replaced when any of those three change.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(
        self, *, window_n: int, randomness: str, checkpoint_hash: str,
    ) -> dict[str, set[int]] | None:
        if not self.path.exists():
            return None
        record = json.loads(self.path.read_text())
        if (
            record.get("window_n") != window_n
            or record.get("randomness") != randomness
            or record.get("checkpoint_hash") != checkpoint_hash
        ):
            return None
        return {
            str(name): {int(prompt) for prompt in prompts}
            for name, prompts in record.get("prompts", {}).items()
        }

    def save(
        self,
        *,
        window_n: int,
        randomness: str,
        checkpoint_hash: str,
        attempted: dict[str, set[int]],
    ) -> None:
        payload = {
            "window_n": window_n,
            "randomness": randomness,
            "checkpoint_hash": checkpoint_hash,
            "prompts": {
                name: sorted(prompts) for name, prompts in attempted.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True))
        os.replace(temporary, self.path)


def _attempt_journal_path(wallet_address: str) -> Path:
    identity = default_checkpoint_identity_path(wallet_address)
    name = identity.name
    if name.startswith("checkpoint-"):
        name = "attempted-" + name[len("checkpoint-"):]
    else:
        name = "attempted-" + name
    return identity.with_name(name)


def _eligible_generation_mix(
    mix: list[tuple[str, int]],
    miner_state,
) -> list[tuple[str, int]]:
    """Remove lanes that already advertise closed admission.

    Open lanes are then weighted by how many admission slots they still
    have. A lane with a handful of seats left is the one that closed under
    this miner mid-generation; spending the GPU on the lane with room left
    is what gets the next valid group through the door.
    """
    if miner_state is None:
        return list(mix)
    open_lanes = []
    for environment, weight in mix:
        lane = miner_state.environments.get(environment)
        if lane is None or not lane.accepting_submissions:
            continue
        remaining = int(getattr(lane, "admission_remaining", 0) or 0)
        open_lanes.append((environment, remaining if remaining > 0 else weight))
    return open_lanes


def _state_matches_active_protocol(state) -> bool:
    """Fail closed on v3; retain compatibility with strict legacy v2 state."""

    if ACTIVE_PROTOCOL_PROFILE.protocol_version < 3:
        return state.protocol_version in (
            None,
            ACTIVE_PROTOCOL_PROFILE.protocol_version,
        )
    return (
        state.protocol_version == ACTIVE_PROTOCOL_PROFILE.protocol_version
        and state.generation_profile_id == ACTIVE_PROTOCOL_PROFILE.profile_id
        and state.generation_contract
        == to_generation_contract(ACTIVE_PROTOCOL_PROFILE)
    )


def _release_state_mismatch_reason(
    *,
    initial_state,
    release_state,
    request,
    environment_name: str,
    environment_size: int,
    cooldown_prompts: set[int],
    prompt_range: tuple[int, int] | None,
    accepting_submissions: bool | None = None,
    now: float | None = None,
) -> str | None:
    """Return why locally prepared work must not cross into live ingress."""
    from reliquary.protocol.submission import WindowState

    if release_state.state != WindowState.OPEN:
        return "window_not_open"
    if release_state.window_n != initial_state.window_n:
        return "window_changed"
    if release_state.randomness != initial_state.randomness:
        return "randomness_changed"
    if any(
        getattr(release_state, field, None)
        != getattr(initial_state, field, None)
        for field in (
            "checkpoint_n",
            "checkpoint_repo_id",
            "checkpoint_revision",
            "protocol_version",
            "generation_profile_id",
            "generation_contract",
        )
    ):
        return "contract_changed"
    if not _state_matches_active_protocol(release_state):
        return "protocol_mismatch"
    if accepting_submissions is False:
        return "admission_closed"
    if request.window_start != release_state.window_n:
        return "request_window_mismatch"
    if request.checkpoint_hash != (release_state.checkpoint_revision or ""):
        return "request_checkpoint_mismatch"
    if request.protocol_version != FORCED_SEED_PROTOCOL_VERSION:
        return "request_protocol_mismatch"
    if (
        ACTIVE_PROTOCOL_PROFILE.protocol_version >= 3
        and request.generation_profile_id != release_state.generation_profile_id
    ):
        return "request_profile_mismatch"
    deadline = getattr(release_state, "submission_deadline_at", None)
    if deadline is not None and (time.time() if now is None else now) >= deadline:
        return "submission_deadline_passed"
    if prompt_range is None:
        prompt_range = window_prompt_range(
            release_state.randomness,
            environment_name,
            environment_size,
            PROMPT_RANGE_SIZE,
        )
    lower, upper = prompt_range
    if not lower <= request.prompt_idx < upper:
        return "prompt_out_of_range"
    if request.prompt_idx in cooldown_prompts:
        return "prompt_in_cooldown"
    return None


def _warn_transformers_mismatch(local_runtime, validator_runtime) -> None:
    local = local_runtime.transformers_version
    remote = validator_runtime.transformers_version
    if local and remote and local != remote:
        logger.warning(
            "Transformers runtime differs: miner=%s validator_proof_worker=%s. "
            "Follow the operator's current runtime release; a version difference "
            "may affect numerical agreement but does not establish a proof failure.",
            local, remote,
        )


def _initial_runtime_bound_nonce(runtime_fingerprint) -> str:
    """Build a schema-valid placeholder before the submitter signs its attempt.

    ``submit_batch_v2`` replaces this with a fresh signed nonce immediately
    before each precommit. The request model still enforces the runtime binding
    at construction time, so the in-memory placeholder must obey that contract.
    """
    if runtime_fingerprint is None:
        return ""
    from reliquary.shared.runtime_fingerprint import bind_runtime_profile_nonce

    return bind_runtime_profile_nonce(
        os.urandom(16).hex(), runtime_fingerprint.profile_hash,
    )


def _release_cuda_memory() -> None:
    """Free a dropped model's blocks now rather than at the next GC pass."""
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _log_proof_memory(proof_gpu: int) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        logger.info(
            "proof copy on cuda:%d after swap: %.1f GiB allocated, "
            "%.1f GiB reserved",
            proof_gpu,
            torch.cuda.memory_allocated(proof_gpu) / 2**30,
            torch.cuda.memory_reserved(proof_gpu) / 2**30,
        )
    except Exception:
        pass


async def maybe_pull_checkpoint(
    state,
    local_n: int,
    local_repo_id: str,
    local_hash: str,
    local_model,
    *,
    download_fn,
    load_fn,
):
    """Activate a newer or same-number/different-revision checkpoint.

    state.checkpoint_repo_id + state.checkpoint_revision identify the
    HF snapshot. download_fn/load_fn still injected for testability.

    Returns ``(new_local_n, new_repo_id, new_local_hash, new_model)``. If no
    update is needed (remote is older, identities match, or the remote snapshot
    is not published yet), returns inputs unchanged.
    """
    remote_identity = checkpoint_identity_from_state(state)
    if remote_identity is None:
        return local_n, local_repo_id, local_hash, local_model
    if remote_identity.checkpoint_n < local_n:
        return local_n, local_repo_id, local_hash, local_model
    if remote_identity.checkpoint_n == local_n:
        if (
            remote_identity.repo_id == local_repo_id
            and remote_identity.oid == local_hash
        ):
            return local_n, local_repo_id, local_hash, local_model
        if local_repo_id or local_hash:
            raise CheckpointIdentityError(
                "checkpoint number was rebound to a different repository or revision"
            )
    local_path = await download_fn(remote_identity.repo_id, remote_identity.oid)
    new_model = await asyncio.to_thread(load_fn, local_path)
    if new_model is None:
        raise RuntimeError("checkpoint loader returned no activated model")
    return (
        remote_identity.checkpoint_n,
        remote_identity.repo_id,
        remote_identity.oid,
        new_model,
    )


async def _hf_download(repo_id: str, revision: str) -> str:
    """Download a snapshot into the local HF cache and return the model folder path."""
    import asyncio
    from huggingface_hub import snapshot_download
    from reliquary.shared.modeling import MODEL_SNAPSHOT_ALLOW_PATTERNS

    return await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        revision=revision,
        allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
    )


def _hf_repo_head(repo_id: str) -> str | None:
    from huggingface_hub import HfApi

    return HfApi().model_info(
        repo_id, revision="main", timeout=10, expand=["sha"],
    ).sha


class _CheckpointPrefetcher:
    """Share checkpoint downloads and start each new repo head early.

    Validators advertise a checkpoint about a minute after it lands on the Hub,
    and the download takes most of that minute. Fetching the head as soon as it
    moves leaves only the load at the window boundary; a pull for a revision
    already in flight waits on that download instead of starting another.
    """

    KEEP = 4

    def __init__(self, *, interval: float, download_fn=None, head_fn=None):
        self.interval = interval
        self._download_fn = download_fn or _hf_download
        self._head_fn = head_fn or _hf_repo_head
        self._downloads: dict[tuple[str, str], asyncio.Future] = {}
        self._repo_id = ""
        self._active = ""
        self._head = ""

    def watch(self, repo_id: str, active_revision: str) -> None:
        self._repo_id = repo_id or ""
        self._active = active_revision or ""

    def pinned(self, repo_id: str) -> set[str]:
        """Revisions of ``repo_id`` still downloading, plus the newest head seen."""
        pinned = {
            revision
            for (repo, revision), task in self._downloads.items()
            if repo == repo_id and not task.done()
        }
        if self._head and self._repo_id == repo_id:
            pinned.add(self._head)
        return pinned

    async def download(self, repo_id: str, revision: str) -> str:
        return await asyncio.shield(self._start(repo_id, revision))

    def _live(self, key: tuple[str, str]) -> asyncio.Future | None:
        task = self._downloads.get(key)
        if task is None or (
            task.done() and (task.cancelled() or task.exception() is not None)
        ):
            return None
        return task

    def _start(self, repo_id: str, revision: str, *, early: bool = False):
        key = (repo_id, revision)
        task = self._live(key)
        if task is not None:
            return task
        started = time.monotonic()
        task = asyncio.ensure_future(self._download_fn(repo_id, revision))
        self._downloads.pop(key, None)
        self._downloads[key] = task

        def finished(done: asyncio.Future) -> None:
            if done.cancelled():
                return
            exc = done.exception()
            if not early:
                return
            if exc is not None:
                logger.warning(
                    "checkpoint prefetch failed %s@%s: %s",
                    repo_id, revision[:12], exc,
                )
            else:
                logger.info(
                    "checkpoint prefetched %s@%s in %.1fs",
                    repo_id, revision[:12], time.monotonic() - started,
                )

        task.add_done_callback(finished)
        while len(self._downloads) > self.KEEP:
            oldest = next(iter(self._downloads))
            if not self._downloads[oldest].done():
                break
            del self._downloads[oldest]
        return task

    async def poll(self) -> None:
        repo_id = self._repo_id
        if not repo_id:
            return
        try:
            head = await asyncio.to_thread(self._head_fn, repo_id)
        except Exception as exc:
            logger.debug("checkpoint head lookup failed: %s", exc)
            return
        if head:
            self._head = head
        if not head or head == self._active or self._live((repo_id, head)):
            return
        logger.info(
            "prefetching checkpoint %s@%s before it is advertised",
            repo_id, head[:12],
        )
        self._start(repo_id, head, early=True)

    async def run(self) -> None:
        while True:
            await self.poll()
            await asyncio.sleep(self.interval)


def _prune_checkpoint_cache(
    repo_id: str, active: str, keep: set[str], *, cache_dir=None,
) -> tuple[int, int]:
    """Delete cached revisions of ``repo_id`` older than ``active``.

    Only revisions last written before the active one go, so a download that
    started after it (and is not yet known to ``keep``) survives. Returns the
    number of revisions deleted and the bytes freed.
    """
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir(cache_dir)
    repo = next(
        (
            entry for entry in cache.repos
            if entry.repo_type == "model" and entry.repo_id == repo_id
        ),
        None,
    )
    if repo is None:
        return 0, 0
    revisions = {revision.commit_hash: revision for revision in repo.revisions}
    if active not in revisions:
        return 0, 0
    cutoff = revisions[active].last_modified
    stale = [
        commit for commit, revision in revisions.items()
        if commit != active
        and commit not in keep
        and revision.last_modified < cutoff
    ]
    if not stale:
        return 0, 0
    # delete_revisions matches a commit hash in every cached repo.
    strategy = dataclasses.replace(
        cache, repos=frozenset({repo}),
    ).delete_revisions(*stale)
    strategy.execute()
    return len(stale), strategy.expected_freed_size


async def _prune_checkpoints(repo_id: str, active: str, keep: set[str]) -> None:
    try:
        deleted, freed = await asyncio.to_thread(
            _prune_checkpoint_cache, repo_id, active, keep,
        )
    except Exception as exc:
        logger.warning("checkpoint cache prune failed: %s", exc)
        return
    if deleted:
        logger.info(
            "pruned %d cached checkpoint revision(s) of %s, freed %.1f GB",
            deleted, repo_id, freed / 1e9,
        )


@contextlib.asynccontextmanager
async def _checkpoint_prefetch(prefetcher: _CheckpointPrefetcher):
    """Poll for new checkpoint heads for the life of the block."""
    if prefetcher.interval <= 0:
        yield prefetcher
        return
    task = asyncio.create_task(prefetcher.run())
    try:
        yield prefetcher
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
    prompt_range: tuple[int, int] | None = None,
) -> int:
    """Pick a random prompt index that isn't currently in cooldown.

    When ``prompt_range`` is given, sampling is confined to ``[lo, hi)`` —
    the per-window slice the validator enforces. The reference miner uses
    uniform-random selection with rejection sampling against the cooldown
    set; more sophisticated strategies are left to miner operators.

    Raises ``RuntimeError`` if no eligible prompt can be found.
    """
    rng = rng or _random
    n = len(env)
    lo, hi = (0, n) if prompt_range is None else prompt_range
    lo = max(0, lo)
    hi = min(n, hi)
    span = hi - lo
    if span <= 0:
        raise RuntimeError("no eligible prompt — empty range")
    cd_in_span = sum(1 for c in cooldown_prompts if lo <= c < hi)
    if cd_in_span < span / 2:
        for _ in range(max_attempts):
            idx = lo + rng.randrange(span)
            if idx not in cooldown_prompts:
                return idx
        raise RuntimeError("no eligible prompt found after max attempts")
    eligible = [i for i in range(lo, hi) if i not in cooldown_prompts]
    if not eligible:
        raise RuntimeError("no eligible prompt — range fully in cooldown")
    return rng.choice(eligible)


def pick_env_and_prompt(
    envs: dict,
    mix: list[tuple[str, int]],
    cooldown_per_env: dict[str, set[int]],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
    randomness: str | None = None,
    prompt_ranges: dict[str, tuple[int, int]] | None = None,
) -> tuple[str, int]:
    """Sample env per `mix` weights, then a prompt within that env.

    When ``randomness`` is given, each env's prompt is drawn only from that
    window's slice (``window_prompt_range``), matching the validator. Falls
    through to the next env (re-sampling with the chosen env masked) if the
    chosen env's slice is fully in cooldown.
    """
    rng = rng or _random
    names = [n for n, _ in mix]
    weights = [w for _, w in mix]
    if not names:
        raise RuntimeError("pick_env_and_prompt: empty mix")

    available = list(names)
    while available:
        avail_weights = [weights[names.index(n)] for n in available]
        env_name = rng.choices(available, weights=avail_weights)[0]
        env = envs[env_name]
        prompt_range = (
            prompt_ranges.get(env_name)
            if prompt_ranges is not None
            else None
        )
        if prompt_range is None and randomness:
            env_label = getattr(env, "name", env_name)
            prompt_range = window_prompt_range(
                randomness, env_label, len(env), PROMPT_RANGE_SIZE,
            )
        try:
            idx = pick_prompt_idx(
                env, cooldown_per_env.get(env_name, set()),
                rng=rng, max_attempts=max_attempts, prompt_range=prompt_range,
            )
            return env_name, idx
        except RuntimeError:
            available.remove(env_name)

    raise RuntimeError("pick_env_and_prompt: all envs fully in cooldown")


def _compute_merkle_root(rollouts) -> str:
    """Compute Merkle root over rollout leaves — returns 64-char hex.

    Uses canonical JSON (sort_keys=True, compact separators) for dict/list
    serialisation so the root is deterministic across Python
    implementations and refactor-stable against dict-construction-order
    changes.
    """
    import hashlib
    import json

    leaves = []
    for i, r in enumerate(rollouts):
        h = hashlib.sha256()
        h.update(i.to_bytes(8, "big"))
        h.update(json.dumps(r.tokens, separators=(",", ":")).encode())
        h.update(json.dumps(r.reward).encode())
        h.update(json.dumps(r.commit, sort_keys=True, separators=(",", ":")).encode())
        leaves.append(h.digest())

    while len(leaves) > 1:
        new = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left
            new.append(hashlib.sha256(left + right).digest())
        leaves = new
    return leaves[0].hex()


def _current_drand_round_at_send() -> int:
    """Drand quicknet round currently in progress at wall-clock now.

    Called just before POSTing /submit so the attached round matches what
    the validator sees at precommit receipt. Production is zero-tolerance;
    ``submit_batch_v2`` avoids signing inside the final second before a round
    boundary so normal scheduling latency cannot stale an honest precommit.
    """
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.infrastructure.drand import get_current_chain

    ci = get_current_chain()
    return compute_current_drand_round(time.time(), ci["genesis_time"], ci["period"])


def _bft_assemble_rollouts(
    *, model, phase1_tensor, prompt_tokens, think_close_ids, force_ids,
    eos_ids, answer_budget, randomness, hotkey, prompt_idx, checkpoint_hash,
    gen_kwargs=None,
):
    """Budget-Forced Termination assembly.

    Rows that hit EOS are kept as-is (truncated at first EOS). Rows that emitted
    ``</think>`` but did not hit EOS are naturally closed and continue sampling
    the answer for ``answer_budget`` tokens. Rows that did not close thinking are
    *forced*: ``force_ids`` are appended and the same phase-2 generation samples
    the boxed answer. Returns one rollout dict per row with ``forced`` and, for
    forced rows, ``force_span`` = (start, end) of the injected ids within
    ``tokens`` (so the validator carve-out and trainer mask can locate them).

    Phase-2 answer tokens are drawn from the same protocol forced-seed stream as
    phase-1, resuming at each row's own completion offset (its primed length past
    the prompt). The injected ``force_ids`` are not sampled and the validator
    excludes that span from the seed-consistency check.
    """
    import torch

    from reliquary.miner.forced_seed_sampler import (
        ForcedSeedLogitsProcessor, forced_seed_generate_kwargs, phase2_base_offsets,
    )
    from reliquary.shared.modeling import first_eos_index, has_think_close

    plen = len(prompt_tokens)
    n = int(phase1_tensor.shape[0])
    close_set = {int(t) for t in think_close_ids}
    force_ids = [int(t) for t in force_ids]

    out: list = [None] * n
    unfinished_idx: list[int] = []
    unfinished_primed: list[list[int]] = []
    unfinished_force_spans: list[tuple[int, int] | None] = []
    for i in range(n):
        seq = phase1_tensor[i].tolist()
        gen = seq[plen:]
        fe = first_eos_index(gen, eos_ids)
        if fe is not None:
            # Finished on EOS: trim padding/trailing garbage and keep as-is.
            gen = gen[: fe + 1]
            out[i] = {"tokens": prompt_tokens + gen,
                      "prompt_length": plen, "forced": False}
        elif has_think_close(gen, close_set):
            # Naturally closed thinking but did not EOS within phase-1. Continue
            # into the answer phase without injecting FORCE and without a carve.
            unfinished_idx.append(i)
            unfinished_primed.append(seq)
            unfinished_force_spans.append(None)
        else:
            force_start = len(seq)
            primed = seq + force_ids
            unfinished_idx.append(i)
            unfinished_primed.append(primed)
            unfinished_force_spans.append((force_start, force_start + len(force_ids)))

    if unfinished_primed:
        width = max(len(p) for p in unfinished_primed)
        pad = min(eos_ids) if eos_ids else 0
        rows = [[pad] * (width - len(p)) + p for p in unfinished_primed]
        mask = [[0] * (width - len(p)) + [1] * len(p) for p in unfinished_primed]
        device = getattr(model, "device", "cpu")
        proc = ForcedSeedLogitsProcessor(
            randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_indices=list(unfinished_idx),
            base_offsets=phase2_base_offsets(
                [len(p) for p in unfinished_primed], plen,
            ),
            start_len=width,
        )
        ans = model.generate(
            torch.tensor(rows, device=device),
            attention_mask=torch.tensor(mask, device=device),
            max_new_tokens=answer_budget,
            **forced_seed_generate_kwargs(gen_kwargs or {}, proc),
        )
        for k, i in enumerate(unfinished_idx):
            primed = unfinished_primed[k]
            tail = ans[k].tolist()[width:]
            fe = first_eos_index(tail, eos_ids)
            tail = tail[: fe + 1] if fe is not None else tail
            forced_span = unfinished_force_spans[k]
            rollout = {"tokens": primed + tail, "prompt_length": plen,
                       "forced": forced_span is not None}
            if forced_span is not None:
                rollout["force_span"] = forced_span
            out[i] = rollout
    return out


@dataclass
class _PreparedGroup:
    """A screened group waiting for its proof, with the state it was drawn in."""

    state: Any
    env_name: str
    env: Any
    prompt_idx: int
    problem: dict
    generations: list[dict]
    rewards: list[float] | None
    checkpoint_hash: str


@dataclass(frozen=True)
class _GenerationTag:
    """What a group in flight on the generator was drawn for."""

    window_n: int
    randomness: str
    env_name: str
    checkpoint_hash: str


class _HostTurn:
    """One generation thread at a time on the host (tokenizer, scoring).

    The tokenizer is not safe to share across threads. A thread holds the turn
    for its host work and hands it over while its group is on the GPU, which is
    where concurrent groups actually overlap.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()

    @contextlib.contextmanager
    def hold(self):
        with self._lock:
            self._local.held = True
            try:
                yield
            finally:
                self._local.held = False

    @contextlib.contextmanager
    def released(self):
        """Hand the turn over for work that touches neither tokenizer nor model."""
        if not getattr(self._local, "held", False):
            yield
            return
        self._local.held = False
        self._lock.release()
        try:
            yield
        finally:
            self._lock.acquire()
            self._local.held = True

    def wait(self, future):
        with self.released():
            return future.result()


# Final verdicts for a group that was admitted or proved but will not be paid:
# refused on reward spread, or arrived after its lane already had the groups.
_UNREWARDED_VERDICTS = frozenset({
    "out_of_zone",
    "proof_not_needed_target_reached",
    "batch_filled",
    "same_prompt_or_content_already_proven",
})


class _EnvironmentYield:
    """Keepable groups per generation-second, per environment.

    Each new window decays the history, so a checkpoint that makes one lane
    easier or harder shows within a few windows. The prior keeps a lane with
    no data in play until it has measurements of its own.
    """

    PRIOR_KEPT = 1.0
    PRIOR_SECONDS = 60.0
    DECAY = 0.8

    def __init__(self) -> None:
        self._kept: dict[str, float] = {}
        self._seconds: dict[str, float] = {}
        self._window: int | None = None

    def start_window(self, window_n: int) -> bool:
        if self._window == window_n:
            return False
        if self._window is not None:
            for totals in (self._kept, self._seconds):
                for name in totals:
                    totals[name] *= self.DECAY
        self._window = window_n
        return True

    def observe(self, env_name: str, seconds: float, *, kept: bool) -> None:
        self._seconds[env_name] = self._seconds.get(env_name, 0.0) + max(0.0, float(seconds))
        if kept:
            self._kept[env_name] = self._kept.get(env_name, 0.0) + 1.0

    def retract(self, env_name: str) -> None:
        """A group counted as kept earned nothing: refused, or too late."""
        self._kept[env_name] = max(0.0, self._kept.get(env_name, 0.0) - 1.0)

    def rate(self, env_name: str) -> float:
        return (self._kept.get(env_name, 0.0) + self.PRIOR_KEPT) / (
            self._seconds.get(env_name, 0.0) + self.PRIOR_SECONDS
        )

    def weigh(
        self,
        mix: list[tuple[str, float]],
        values: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """Scale each lane's weight by its rate (times its value) relative to the open lanes."""
        return _weigh_lanes(
            mix, rates={name: self.rate(name) for name, _ in mix}, values=values,
        )

    def summary(self, names) -> str:
        return " ".join(
            f"{name}={self._kept.get(name, 0.0):.1f}kept/"
            f"{self._seconds.get(name, 0.0):.0f}s"
            for name in names
        )


def _weigh_lanes(
    mix: list[tuple[str, float]],
    *,
    rates: dict[str, float] | None = None,
    values: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Scale each lane's weight by rate x value, normalised over the open lanes."""
    if len(mix) < 2 or (rates is None and not values):
        return list(mix)
    scores = {
        name: (rates[name] if rates is not None else 1.0)
        * (values.get(name, 1.0) if values else 1.0)
        for name, _ in mix
    }
    mean = sum(scores.values()) / len(scores)
    if mean <= 0:
        return list(mix)
    return [(name, weight * scores[name] / mean) for name, weight in mix]


def _parse_lane_values(raw: str) -> dict[str, float]:
    """``"env=value,env=value"`` into positive per-lane values; bad entries are skipped."""
    values: dict[str, float] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, number = item.partition("=")
        try:
            value = float(number) if sep else float("nan")
        except ValueError:
            value = float("nan")
        if not name.strip() or not math.isfinite(value) or value <= 0:
            logger.warning("ignoring lane value %r: expected env=positive-number", item)
            continue
        values[name.strip()] = value
    return values


def _live_lane_values(payload: Any, task_id: str | None = None) -> dict[str, float] | None:
    """Per-environment prices from a validator's ``GET /tasks`` body.

    ``{}`` means the price is published but every lane pays alike (not applied,
    or one price for all); ``None`` means the body carries no usable price.
    """
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list) or not tasks:
        return None
    entries = [task for task in tasks if isinstance(task, dict)]
    entry = next(
        (task for task in entries if task_id and task.get("task_id") == task_id),
        entries[0] if entries else None,
    )
    price = entry.get("price") if entry is not None else None
    if not isinstance(price, dict):
        return None
    if not price.get("applied"):
        return {}
    by_environment = price.get("by_environment")
    if not isinstance(by_environment, dict):
        return {}
    values: dict[str, float] = {}
    for name, decision in by_environment.items():
        value = decision.get("value") if isinstance(decision, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            values[str(name)] = float(value)
    return values


class _LaneValues:
    """What one selected group pays in each lane, relative to the others.

    The validator prices each environment on its own and pays a fixed share per
    selected group, so a lane's value is its price. A live ``/tasks`` price wins;
    otherwise the operator's configured values (read off the dashboard) apply.
    Declared per-environment caps are not published, so lanes are assumed to
    share one cap.
    """

    RETRY_SECONDS = 60.0
    UNPUBLISHED_RETRY_SECONDS = 1800.0

    def __init__(self, configured: dict[str, float], *, live: bool = True) -> None:
        self._configured = dict(configured)
        self._live_enabled = live
        self._live: dict[str, float] | None = None
        self._retry_at = 0.0
        self._task: asyncio.Task | None = None
        self._reported_unpublished = False

    @property
    def source(self) -> str:
        if self._live is not None:
            return "live" if self._live else "live-uniform"
        return "configured" if self._configured else "uniform"

    def values(self, names) -> dict[str, float] | None:
        source = self._live if self._live is not None else self._configured
        if not source:
            return None
        present = [source[name] for name in names if name in source]
        fallback = sum(present) / len(present) if present else 1.0
        return {name: source.get(name, fallback) for name in names}

    def describe(self, names) -> str:
        values = self.values(list(names))
        if values is None:
            return f"lane values ({self.source})"
        return f"lane values ({self.source}): " + " ".join(
            f"{name}={value:.3g}" for name, value in values.items()
        )

    def maybe_refresh(self, client, url: str) -> None:
        if not self._live_enabled or time.monotonic() < self._retry_at:
            return
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._refresh(client, url))

    async def _refresh(self, client, url: str) -> None:
        from reliquary.constants import TASK_ID

        try:
            response = await client.get(f"{url.rstrip('/')}/tasks", timeout=5)
        except Exception as exc:
            logger.debug("lane price fetch failed: %s", exc)
            self._retry_at = time.monotonic() + self.RETRY_SECONDS
            return
        if response.status_code == 404:
            if not self._reported_unpublished:
                self._reported_unpublished = True
                logger.info(
                    "validator does not publish lane prices at /tasks; using %s",
                    self.describe(self._configured),
                )
            self._retry_at = time.monotonic() + self.UNPUBLISHED_RETRY_SECONDS
            return
        live = None
        if response.status_code == 200:
            try:
                live = _live_lane_values(response.json(), TASK_ID)
            except ValueError:
                live = None
        if live is None:
            self._retry_at = time.monotonic() + self.RETRY_SECONDS
            return
        if live != self._live:
            self._live = live
            logger.info("live %s", self.describe(sorted(live) or self._configured))


class _EnvironmentWarmup:
    """Builds each environment's prompt index off the event loop.

    A remote parquet dataset lists and reads every shard footer on first use --
    one to two minutes for OpenMathInstruct -- and doing that inside the mining
    loop stalls polling and submission. A lane joins the mix once it is ready.
    """

    RETRY_SECONDS = 30.0

    def __init__(self, envs: dict) -> None:
        self._envs = dict(envs)
        self._futures: dict[str, Any] = {}
        self._retry_at: dict[str, float] = {}
        for name in self._envs:
            self._start(name)

    def _start(self, name: str) -> None:
        import concurrent.futures

        env = self._envs[name]
        future: concurrent.futures.Future = concurrent.futures.Future()

        def run() -> None:
            started = time.monotonic()
            try:
                size = len(env)
            except BaseException as exc:
                future.set_exception(exc)
                return
            elapsed = time.monotonic() - started
            if elapsed >= 1.0:
                logger.info(
                    "prompt source ready: env=%s prompts=%d in %.1fs",
                    name, size, elapsed,
                )
            future.set_result(size)

        self._futures[name] = future
        threading.Thread(target=run, name=f"env-warmup-{name}", daemon=True).start()

    def ready(self, name: str) -> bool:
        future = self._futures.get(name)
        if future is None:
            return True
        if not future.done():
            return False
        exc = future.exception()
        if exc is None:
            return True
        now = time.monotonic()
        retry_at = self._retry_at.get(name)
        if retry_at is None:
            logger.warning(
                "prompt source for %s failed to load (%s); retrying in %.0fs",
                name, exc, self.RETRY_SECONDS,
            )
            self._retry_at[name] = now + self.RETRY_SECONDS
        elif now >= retry_at:
            del self._retry_at[name]
            self._start(name)
        return False

    def pending(self, names) -> list[str]:
        return [name for name in names if not self.ready(name)]


def _prefetch_prompt_range(env_name: str, env, prompt_range: tuple[int, int]) -> bool:
    """Warm the row-groups behind a window's prompt range in the background."""
    prefetch = getattr(getattr(env, "_dataset", None), "prefetch", None)
    if not callable(prefetch):
        return False
    lo, hi = prompt_range

    def run() -> None:
        started = time.monotonic()
        try:
            touched = prefetch(lo, hi)
        except Exception as exc:
            logger.warning(
                "prompt range prefetch failed: env=%s [%d, %d): %s",
                env_name, lo, hi, exc,
            )
            return
        logger.info(
            "prompt range prefetched: env=%s [%d, %d) %d row-group(s) in %.1fs",
            env_name, lo, hi, touched, time.monotonic() - started,
        )

    threading.Thread(
        target=run, name=f"prompt-prefetch-{env_name}", daemon=True,
    ).start()
    return True


def _resolve_pipeline_depth(
    configured: int | None, *, generation_gpu: int, proof_gpu: int,
) -> int:
    if configured is not None:
        return max(0, int(configured))
    from reliquary.constants import MINER_PIPELINE_DEPTH

    if MINER_PIPELINE_DEPTH is not None:
        return MINER_PIPELINE_DEPTH
    return 1 if generation_gpu != proof_gpu else 0


def _cuda_device_scope(index: int):
    """Pin a worker thread's current CUDA device; kernels that read the
    current device instead of the tensor's must not land on the other GPU."""
    try:
        import torch
    except ImportError:
        return contextlib.nullcontext()
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    return torch.cuda.device(index)


def _stale_group_reason(
    group: _PreparedGroup, *, latest_state, checkpoint_hash: str,
) -> str | None:
    """Why a queued group is already unsubmittable, before paying for its proof."""
    if group.checkpoint_hash != checkpoint_hash:
        return "checkpoint_changed"
    if latest_state is None:
        return None
    if latest_state.window_n != group.state.window_n:
        return "window_changed"
    if latest_state.randomness != group.state.randomness:
        return "randomness_changed"
    return None


@contextlib.asynccontextmanager
async def _proof_stage(drain, enabled: bool):
    """Run ``drain`` as the proof/submit task for the life of the block."""
    if not enabled:
        yield None
        return
    task = asyncio.create_task(drain())
    try:
        yield task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@contextlib.asynccontextmanager
async def _generation_tasks(tasks: set, generator):
    """Stop every generation task (and its engine requests) when mining stops."""
    try:
        yield tasks
    finally:
        if generator is not None and hasattr(generator, "abort"):
            generator.abort(reason="miner_stopped")
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _rollout_metadata(generation: dict, token_logprobs: list) -> dict:
    """Per-rollout metadata embedded in the GRAIL commit. Carries the BFT
    ``forced`` flag and ``force_span`` so the validator carve-out and trainer
    mask can locate the injected span."""
    prompt_length = int(generation["prompt_length"])
    all_tokens = generation["tokens"]
    force_span = generation.get("force_span")
    return {
        "prompt_length": prompt_length,
        "completion_length": len(all_tokens) - prompt_length,
        "success": True,
        "total_reward": 0.0,
        "advantage": 0.0,
        "token_logprobs": token_logprobs,
        "forced": bool(generation.get("forced", False)),
        "force_span": list(force_span) if force_span else None,
    }


class MiningEngine:
    """Two-GPU mining: vLLM (GPU 0) for generation, HF (GPU 1) for proofs."""

    def __init__(
        self,
        vllm_model,
        hf_model,
        tokenizer,
        wallet,
        env: "Environment | None" = None,
        *,
        envs: "dict[str, Environment] | None" = None,
        mix: "list[tuple[str, int]] | None" = None,
        vllm_gpu: int = 0,
        proof_gpu: int = 1,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        validator_url_override: str | None = None,
        checkpoint_identity_store: MinerCheckpointIdentityStore | None = None,
        initial_checkpoint_identity: ActivatedCheckpoint | None = None,
        generator: Any | None = None,
        generation_device: str | None = None,
        pipeline_depth: int | None = None,
        generation_concurrency: int = 1,
        env_warmup: "_EnvironmentWarmup | None" = None,
    ) -> None:
        self._env_warmup = env_warmup
        # When set, single-turn rollouts come from this engine instead of
        # ``vllm_model.generate``; the proof still runs on ``hf_model``.
        self.generator = generator
        # Prompt groups generating at once. Only a generator that batches
        # across threads (vLLM) can use more than one.
        self.generation_concurrency = max(1, int(generation_concurrency))
        self.vllm_model = vllm_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.vllm_gpu = vllm_gpu
        self.proof_gpu = proof_gpu
        # Where the transformers generation copy lives. ``None`` means
        # ``cuda:{vllm_gpu}``; the vLLM backend keeps it on the host.
        self.generation_device = generation_device
        self.pipeline_depth = pipeline_depth
        self.max_new_tokens = max_new_tokens
        self.validator_url_override = validator_url_override
        self._checkpoint_identity_store = (
            checkpoint_identity_store
            or MinerCheckpointIdentityStore(
                default_checkpoint_identity_path(
                    wallet.hotkey.ss58_address,
                )
            )
        )
        self._initial_checkpoint_identity = initial_checkpoint_identity

        if envs is not None and mix is not None:
            self.envs = envs
            self.mix = mix
            self.env = next(iter(envs.values()))  # legacy fallback
        else:
            assert env is not None, "must pass either env or envs+mix"
            self.envs = {env.name: env}
            self.mix = [(env.name, 1)]
            self.env = env
        self._cooldown_per_env: dict[str, set[int]] = {n: set() for n in self.envs}

        # Lazy imports for heavy deps — keep module import cheap.
        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_batch_request_from_generations(
        self,
        *,
        generations: list[dict],
        problem: dict,
        environment: "Environment",
        randomness: str,
        prompt_idx: int,
        window_number: int,
        checkpoint_revision: str,
        runtime_fingerprint=None,
        rewards: list[float] | None = None,
    ):
        """Turn backend-produced token sequences into a protocol request.

        Generation backends only need to return the same small generation
        dictionaries as the built-in path. Proof construction, token
        log-probabilities, rewards, signatures, and request formatting remain
        on the existing canonical implementation. ``rewards``, when given, are
        the values ``_score_generations`` already computed for this group.
        """
        from reliquary.protocol.submission import BatchSubmissionRequest

        if len(generations) != M_ROLLOUTS:
            raise ValueError(
                f"expected {M_ROLLOUTS} generations, got {len(generations)}"
            )
        if rewards is not None and len(rewards) != len(generations):
            raise ValueError("rewards must align with generations")
        rollouts = [
            self._build_rollout_submission(
                generation,
                problem,
                randomness,
                env=environment,
                reward=None if rewards is None else rewards[index],
            )
            for index, generation in enumerate(generations)
        ]
        return BatchSubmissionRequest(
            miner_hotkey=self.wallet.hotkey.ss58_address,
            prompt_idx=prompt_idx,
            window_start=window_number,
            merkle_root=_compute_merkle_root(rollouts),
            rollouts=rollouts,
            checkpoint_hash=checkpoint_revision,
            runtime_fingerprint=runtime_fingerprint,
            nonce=_initial_runtime_bound_nonce(runtime_fingerprint),
            protocol_version=FORCED_SEED_PROTOCOL_VERSION,
            generation_profile_id=(
                ACTIVE_PROTOCOL_PROFILE.profile_id
                if ACTIVE_PROTOCOL_PROFILE.protocol_version >= 3
                else ""
            ),
        )

    async def mine_window(
        self,
        subtensor,
        window_start: int = 0,  # v2.0 param kept for CLI compat; ignored
        use_drand: bool = True,
    ) -> list:
        """v2.1: poll state, pull checkpoint on n-change, submit when OPEN.

        Returns the list of BatchSubmissionResponse objects collected
        across the loop. The loop exits only on external cancellation
        (asyncio.CancelledError) or if env becomes fully cooldown'd.
        """
        import httpx
        import random

        from reliquary.constants import (
            MINER_CHECKPOINT_PREFETCH_SECONDS,
            MINER_CONCURRENT_UPLOADS,
            MINER_LANE_VALUES,
            MINER_LIVE_LANE_PRICES,
            MINER_PRUNE_CHECKPOINTS,
            MINER_YIELD_WEIGHTING,
            POLL_INTERVAL_SECONDS,
        )
        from reliquary.miner.submitter import (
            EndpointNotFoundError,
            NoActiveWindowError,
            SubmissionError,
            discover_validator_url,
            get_miner_state_v1,
            log_submission_verdict,
            monitor_submission_verdicts,
            get_runtime_contract_v1,
            get_window_state_v2,
            submit_batch_v2,
        )
        from reliquary.protocol.submission import (
            RuntimeFingerprint, WindowState,
        )
        from reliquary.shared.runtime_fingerprint import (
            collect_runtime_fingerprint,
        )

        # Resolve validator URL (once).
        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        # v2.3: randomness is fetched per-window from /state instead of
        # recomputed locally. The validator aligns window OPEN to a drand
        # boundary and binds randomness to the round publishing at that
        # boundary — a value that didn't exist a few seconds earlier, so
        # nothing to pre-fetch. The miner just reads what /state reports.
        rng = random.Random()
        results = []
        persisted_identity = self._checkpoint_identity_store.load()
        initial_identity = self._initial_checkpoint_identity
        if initial_identity is None and persisted_identity is not None:
            raise CheckpointActivationRestartRequired(
                "durable checkpoint identity exists but the active model "
                "identity was not supplied"
            )
        if initial_identity is not None:
            try:
                self._checkpoint_identity_store.commit(initial_identity)
            except CheckpointIdentityError:
                raise
            except Exception as exc:
                raise CheckpointActivationRestartRequired(
                    "checkpoint identity could not be persisted"
                ) from exc
            local_n = initial_identity.checkpoint_n
            local_repo_id = initial_identity.repo_id
            local_hash = initial_identity.oid
        else:
            local_n = 0
            local_repo_id = ""
            local_hash = ""
        miner_state_supported: bool | None = None
        cached_miner_state = None
        miner_state_etag: str | None = None
        legacy_cooldown_window: int | None = None
        prompt_ranges: dict[str, tuple[int, int]] = {}
        # The forced draw is fixed per (window randomness, prompt, checkpoint),
        # so a prompt already generated this window can only reproduce the
        # same tokens: never pick it again, whatever happened to it.
        attempted_key: tuple[int, str, str] | None = None
        attempted: dict[str, set[int]] = {name: set() for name in self.envs}
        attempt_journal = AttemptedPromptJournal(
            _attempt_journal_path(self.wallet.hotkey.ss58_address)
        )

        submitted = asyncio.Event()
        # Generation (vllm_gpu) and proof (proof_gpu) overlap: while one
        # group is proved and submitted, the next is already generating.
        # Checkpoint activation swaps both models, so it waits for any proof
        # in flight; queued groups from the old checkpoint are then dropped.
        pipeline_depth = _resolve_pipeline_depth(
            getattr(self, "pipeline_depth", None),
            generation_gpu=self.vllm_gpu,
            proof_gpu=self.proof_gpu,
        )
        proof_queue: asyncio.Queue[_PreparedGroup] = asyncio.Queue(
            maxsize=max(1, pipeline_depth),
        )
        activation_lock = asyncio.Lock()
        latest_state = None
        runtime_fingerprint = None
        # Several prompts generate at once when the generator batches across
        # threads; each holds a slot until its group is queued for proof, so a
        # slow proof stage throttles generation instead of piling up groups.
        concurrency = self._resolved_generation_concurrency()
        generation_slots = asyncio.Semaphore(concurrency)
        generating: set[asyncio.Task] = set()
        generation_failures: list[BaseException] = []
        uploading: set[asyncio.Task] = set()
        upload_slots = asyncio.Semaphore(max(1, MINER_CONCURRENT_UPLOADS))
        submit_resume_at = 0.0
        generator = getattr(self, "generator", None)
        environment_yield = self._environment_yield()
        warmup = getattr(self, "_env_warmup", None) or _EnvironmentWarmup(self.envs)
        lane_values = _LaneValues(
            _parse_lane_values(MINER_LANE_VALUES), live=MINER_LIVE_LANE_PRICES,
        )
        prefetched_ranges: dict[str, tuple[int, int]] = {}
        warmup_reported_at = 0.0
        submitted_envs: dict[tuple[int, int], list[str]] = {}
        prefetcher = _CheckpointPrefetcher(
            interval=MINER_CHECKPOINT_PREFETCH_SECONDS,
        )
        prefetcher.watch(local_repo_id, local_hash)
        pruned_for = ""
        prune_task: asyncio.Task | None = None
        logger.info(
            "miner pipeline: generation cuda:%d, proof cuda:%d, depth=%d, "
            "concurrent_groups=%d",
            self.vllm_gpu, self.proof_gpu, pipeline_depth, concurrency,
        )

        retracted: set[tuple[int, int]] = set()

        def on_verdict(verdict) -> None:
            log_submission_verdict(verdict)
            reason = verdict.outcome_code or verdict.selection_reason or verdict.reason
            reason = getattr(reason, "value", reason)
            key = (int(verdict.window_n), int(verdict.prompt_idx))
            envs = submitted_envs.get(key)
            if (
                reason in _UNREWARDED_VERDICTS
                and envs
                and len(set(envs)) == 1
                and key not in retracted
            ):
                retracted.add(key)
                environment_yield.retract(envs[0])

        def abort_generating(predicate, reason: str) -> None:
            if generator is None or not hasattr(generator, "abort"):
                return
            count = generator.abort(predicate, reason=reason)
            if count:
                logger.info(
                    "aborted %d generating group(s): reason=%s", count, reason,
                )

        async def prove(group: _PreparedGroup):
            """Proof-GPU half of a release: the request to upload, or ``None``."""
            state = group.state
            env_name, prompt_idx = group.env_name, group.prompt_idx
            stale = _stale_group_reason(
                group, latest_state=latest_state, checkpoint_hash=local_hash,
            )
            if stale is not None:
                logger.info(
                    "discarding queued group before proof: reason=%s window=%d "
                    "env=%s prompt=%d",
                    stale, state.window_n, env_name, prompt_idx,
                )
                return None
            async with activation_lock:
                proof_started = time.monotonic()
                request, upload_skip = await asyncio.to_thread(
                    self._prove_group, group, runtime_fingerprint,
                )
            logger.info(
                "stage proof: window=%d env=%s prompt=%d cuda:%d %.1fs",
                state.window_n, env_name, prompt_idx, self.proof_gpu,
                time.monotonic() - proof_started,
            )
            if upload_skip is not None:
                logger.info(
                    "skipping upload: reason=%s window=%d env=%s prompt=%d",
                    upload_skip, state.window_n, env_name, prompt_idx,
                )
                return None
            return request

        async def upload(group: _PreparedGroup, request) -> None:
            nonlocal miner_state_etag, cached_miner_state, submit_resume_at
            state = group.state
            env_name, env, prompt_idx = group.env_name, group.env, group.prompt_idx
            backoff = submit_resume_at - time.monotonic()
            if backoff > 0:
                await asyncio.sleep(backoff)

            # Generation and proof construction can span a state
            # transition. Re-read the exact live lane immediately before
            # precommit and discard stale work locally.
            try:
                if miner_state_supported is not False:
                    refreshed, miner_state_etag = await get_miner_state_v1(
                        url,
                        client=client,
                        etag=miner_state_etag,
                    )
                    if refreshed is not None:
                        cached_miner_state = refreshed
                    release_state = cached_miner_state
                    if release_state is None:
                        raise SubmissionError(
                            "missing cached miner-state at release"
                        )
                    env_release = release_state.environments.get(env_name)
                    if env_release is None:
                        raise SubmissionError(
                            "release state omitted the selected environment"
                        )
                    release_cooldown = env_release.cooldown_prompts()
                    release_range = env_release.prompt_range
                    release_accepting = env_release.accepting_submissions
                else:
                    release_state = await get_window_state_v2(
                        url,
                        env=env_name,
                        client=client,
                    )
                    release_cooldown = set(release_state.cooldown_prompts)
                    release_range = window_prompt_range(
                        release_state.randomness,
                        getattr(env, "name", env_name),
                        len(env),
                        PROMPT_RANGE_SIZE,
                    )
                    release_accepting = None
            except Exception as exc:
                logger.info(
                    "state recheck failed; discarding prepared work: %s",
                    exc,
                )
                return

            mismatch = _release_state_mismatch_reason(
                initial_state=state,
                release_state=release_state,
                request=request,
                environment_name=getattr(env, "name", env_name),
                environment_size=len(env),
                cooldown_prompts=release_cooldown,
                prompt_range=release_range,
                accepting_submissions=release_accepting,
            )
            if mismatch is not None:
                logger.info(
                    "discarding prepared work before ingress: reason=%s "
                    "window=%d prompt=%d",
                    mismatch,
                    request.window_start,
                    request.prompt_idx,
                )
                return
            try:
                resp = await submit_batch_v2(
                    url,
                    request,
                    client=client,
                    wallet=self.wallet,
                    randomness=state.randomness or "",
                    drand_round_fn=_current_drand_round_at_send,
                )
                logger.info(
                    "submitted window=%d prompt=%d accepted=%s reason=%s",
                    state.window_n, prompt_idx, resp.accepted,
                    resp.reason.value if hasattr(resp.reason, "value") else resp.reason,
                )
                results.append(resp)
                if resp.accepted:
                    submitted_envs.setdefault(
                        (int(state.window_n), int(prompt_idx)), [],
                    ).append(env_name)
                    submitted.set()
                elif resp._retry_after_seconds is not None:
                    submit_resume_at = max(
                        submit_resume_at,
                        time.monotonic() + resp._retry_after_seconds,
                    )
            except SubmissionError as exc:
                logger.error("submit failed: %s", exc)

        async def release(group: _PreparedGroup) -> None:
            request = await prove(group)
            if request is not None:
                await upload(group, request)

        async def upload_in_background(group: _PreparedGroup, request) -> None:
            try:
                await upload(group, request)
            except Exception:
                logger.exception(
                    "upload failed: window=%d env=%s prompt=%d",
                    group.state.window_n, group.env_name, group.prompt_idx,
                )
            finally:
                upload_slots.release()

        async def drain() -> None:
            # The proof GPU moves on to the next group while this one's state
            # recheck and upload (seconds of network) run beside it.
            while True:
                group = await proof_queue.get()
                try:
                    request = await prove(group)
                finally:
                    proof_queue.task_done()
                if request is None:
                    continue
                await upload_slots.acquire()
                task = asyncio.create_task(upload_in_background(group, request))
                uploading.add(task)
                task.add_done_callback(uploading.discard)

        async def hand_off(group: _PreparedGroup, consumer) -> None:
            if consumer is None:
                await release(group)
                return
            put = asyncio.ensure_future(proof_queue.put(group))
            done, _ = await asyncio.wait(
                {put, consumer}, return_when=asyncio.FIRST_COMPLETED,
            )
            if put not in done:
                put.cancel()
                consumer.result()
                raise RuntimeError("proof stage stopped")

        def generation_finished(task: asyncio.Task) -> None:
            generating.discard(task)
            if not task.cancelled() and task.exception() is not None:
                generation_failures.append(task.exception())

        async def generate_and_hand_off(
            *, state, env_name: str, env, prompt_idx: int,
            randomness: str, checkpoint_hash: str,
        ) -> None:
            from reliquary.miner.vllm_generation import GenerationAborted

            try:
                # A remote prompt source fetches a whole row-group on a miss.
                try:
                    problem = await asyncio.to_thread(env.get_problem, prompt_idx)
                except Exception as exc:
                    logger.warning(
                        "prompt fetch failed: window=%d env=%s prompt=%d: %s",
                        state.window_n, env_name, prompt_idx, exc,
                    )
                    return
                generation_started = time.monotonic()
                try:
                    screened = await asyncio.to_thread(
                        self._generate_tagged_group,
                        _GenerationTag(
                            window_n=state.window_n,
                            randomness=randomness,
                            env_name=env_name,
                            checkpoint_hash=checkpoint_hash,
                        ),
                        env_name=env_name,
                        env=env,
                        problem=problem,
                        prompt_idx=prompt_idx,
                        randomness=randomness,
                        checkpoint_hash=checkpoint_hash,
                        window_n=state.window_n,
                    )
                except GenerationAborted as exc:
                    generation_seconds = time.monotonic() - generation_started
                    logger.info(
                        "stage generation: window=%d env=%s prompt=%d cuda:%d "
                        "%.1fs abandoned reason=%s",
                        state.window_n, env_name, prompt_idx, self.vllm_gpu,
                        generation_seconds, exc.reason,
                    )
                    # Lane, window and checkpoint aborts say nothing about
                    # the environment; too many truncations does.
                    if exc.reason == "too_many_truncated":
                        environment_yield.observe(
                            env_name, generation_seconds, kept=False,
                        )
                    return
                generation_seconds = time.monotonic() - generation_started
                environment_yield.observe(
                    env_name, generation_seconds, kept=screened is not None,
                )
                if screened is None:
                    logger.info(
                        "stage generation: window=%d env=%s prompt=%d cuda:%d "
                        "%.1fs screened_out",
                        state.window_n, env_name, prompt_idx, self.vllm_gpu,
                        generation_seconds,
                    )
                    return
                generations, local_rewards = screened
                completion_tokens = sum(
                    len(g["tokens"]) - int(g["prompt_length"]) for g in generations
                )
                logger.info(
                    "stage generation: window=%d env=%s prompt=%d cuda:%d %.1fs "
                    "rollouts=%d completion_tokens=%d (%.0f tok/s)",
                    state.window_n, env_name, prompt_idx, self.vllm_gpu,
                    generation_seconds, len(generations), completion_tokens,
                    completion_tokens / max(generation_seconds, 1e-6),
                )
                await hand_off(
                    _PreparedGroup(
                        state=state,
                        env_name=env_name,
                        env=env,
                        prompt_idx=prompt_idx,
                        problem=problem,
                        generations=generations,
                        rewards=local_rewards,
                        checkpoint_hash=checkpoint_hash,
                    ),
                    consumer,
                )
            finally:
                generation_slots.release()

        async with (
            httpx.AsyncClient(timeout=30, limits=httpx.Limits(keepalive_expiry=30)) as client,
            monitor_submission_verdicts(
                url, self.wallet.hotkey.ss58_address, client, submitted,
                on_verdict=on_verdict,
            ),
            # Entered before the proof stage so it exits after it: no upload
            # can be started once these are cancelled.
            _generation_tasks(uploading, None),
            _proof_stage(drain, pipeline_depth > 0) as consumer,
            _generation_tasks(generating, generator),
            _checkpoint_prefetch(prefetcher),
        ):
            try:
                contract = await get_runtime_contract_v1(url, client=client)
                runtime_fingerprint = RuntimeFingerprint.model_validate(
                    collect_runtime_fingerprint(
                        generation_model=self.vllm_model,
                        proof_model=self.hf_model,
                    )
                )
                _warn_transformers_mismatch(runtime_fingerprint, contract.validator_profile)
                logger.info(
                    "validator runtime telemetry enabled version=%d "
                    "validator_profile=%s",
                    contract.telemetry_version,
                    contract.validator_profile.profile_hash,
                )
            except (SubmissionError, ValueError):
                # Older validators may omit the capability or expose an older
                # telemetry schema. Omitting the optional request field keeps
                # mining wire-compatible in both cases.
                runtime_fingerprint = None
                logger.info("validator runtime telemetry unavailable")
            while True:
                if consumer is not None and consumer.done():
                    consumer.result()
                    raise RuntimeError("proof stage stopped")
                if generation_failures:
                    raise generation_failures[0]
                try:
                    if miner_state_supported is not False:
                        try:
                            fetched, new_etag = await get_miner_state_v1(
                                url,
                                client=client,
                                etag=miner_state_etag,
                            )
                            miner_state_supported = True
                            if fetched is not None:
                                missing = set(self.envs) - set(
                                    fetched.environments
                                )
                                if missing:
                                    raise SubmissionError(
                                        "miner-state missing environments: "
                                        f"{sorted(missing)}"
                                    )
                                cached_miner_state = fetched
                                self._cooldown_per_env = {
                                    environment: fetched.environments[
                                        environment
                                    ].cooldown_prompts()
                                    for environment in self.envs
                                }
                                prompt_ranges = {
                                    environment: fetched.environments[
                                        environment
                                    ].prompt_range
                                    for environment in self.envs
                                }
                            miner_state_etag = new_etag
                            if cached_miner_state is None:
                                raise SubmissionError(
                                    "304 before miner-state cache fill"
                                )
                            state = cached_miner_state
                        except EndpointNotFoundError:
                            miner_state_supported = False
                            miner_state_etag = None
                            cached_miner_state = None
                            state = await get_window_state_v2(
                                url,
                                client=client,
                            )
                    else:
                        state = await get_window_state_v2(url, client=client)
                except NoActiveWindowError as exc:
                    await asyncio.sleep(_no_active_window_delay(exc))
                    continue
                except SubmissionError:
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue
                except Exception as e:
                    logger.debug("state fetch failed: %s", e)
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue
                latest_state = state
                abort_generating(
                    lambda tag: tag.window_n != state.window_n
                    or bool(state.randomness and tag.randomness != state.randomness),
                    "window_changed",
                )

                # Legacy compatibility stays exact: gather every environment
                # for one immutable window or wait. Never substitute the first
                # environment's cooldown for a failed per-env request.
                if (
                    miner_state_supported is False
                    and legacy_cooldown_window != state.window_n
                ):
                    try:
                        cooldowns: dict[str, set[int]] = {}
                        derived_ranges: dict[str, tuple[int, int]] = {}
                        for environment, env in self.envs.items():
                            env_state = await get_window_state_v2(
                                url,
                                env=environment,
                                client=client,
                            )
                            if any(
                                getattr(env_state, field)
                                != getattr(state, field)
                                for field in (
                                    "state",
                                    "window_n",
                                    "randomness",
                                    "checkpoint_n",
                                    "checkpoint_repo_id",
                                    "checkpoint_revision",
                                    "protocol_version",
                                    "generation_profile_id",
                                    "generation_contract",
                                )
                            ):
                                raise SubmissionError(
                                    "legacy state crossed a window boundary"
                                )
                            cooldowns[environment] = set(
                                env_state.cooldown_prompts
                            )
                            derived_ranges[environment] = window_prompt_range(
                                env_state.randomness,
                                getattr(env, "name", environment),
                                len(env),
                                PROMPT_RANGE_SIZE,
                            )
                        self._cooldown_per_env = cooldowns
                        prompt_ranges = derived_ranges
                        legacy_cooldown_window = state.window_n
                    except Exception as exc:
                        logger.warning(
                            "incomplete environment state; waiting: %s",
                            exc,
                        )
                        await asyncio.sleep(1)
                        continue

                # Pull new checkpoint if needed (works at any state).
                advertised = checkpoint_identity_from_state(state)
                prefetcher.watch(
                    advertised.repo_id if advertised is not None else local_repo_id,
                    local_hash,
                )
                activation_scope = (
                    activation_lock
                    if advertised is not None and (
                        advertised.repo_id != local_repo_id
                        or advertised.oid != local_hash
                    )
                    else contextlib.nullcontext()
                )
                try:
                    async with activation_scope:
                        pulled = await maybe_pull_checkpoint(
                            state=state,
                            local_n=local_n,
                            local_hash=local_hash,
                            local_repo_id=local_repo_id,
                            local_model=self.hf_model,
                            download_fn=prefetcher.download,
                            load_fn=self._load_checkpoint,
                        )
                    pulled_n, pulled_repo, pulled_hash, pulled_model = pulled
                    if pulled_repo and pulled_hash:
                        try:
                            self._checkpoint_identity_store.commit(
                                ActivatedCheckpoint(
                                    checkpoint_n=pulled_n,
                                    repo_id=pulled_repo,
                                    oid=pulled_hash,
                                )
                            )
                        except CheckpointIdentityError:
                            raise
                        except Exception as exc:
                            raise CheckpointActivationRestartRequired(
                                "checkpoint identity could not be persisted"
                            ) from exc
                    swapped = (pulled_repo, pulled_hash) != (local_repo_id, local_hash)
                    local_n = pulled_n
                    local_repo_id = pulled_repo
                    local_hash = pulled_hash
                    self.hf_model = pulled_model
                    # The replaced proof copy stays referenced from here and
                    # from the pull's frame until the load returns; drop it
                    # now so the next reload can stage beside the current one.
                    pulled = pulled_model = None
                    if swapped:
                        _release_cuda_memory()
                        _log_proof_memory(self.proof_gpu)
                except (CheckpointIdentityError, CheckpointActivationRestartRequired):
                    logger.exception(
                        "checkpoint identity or activation is unsafe; "
                        "terminating for a clean restart"
                    )
                    raise
                except Exception:
                    logger.exception("checkpoint pull failed; keeping local")
                    pull_failed = True
                else:
                    pull_failed = False
                if pull_failed:
                    # A partly staged copy is only unreachable once the
                    # traceback is gone.
                    _release_cuda_memory()
                # set_weights already aborted what was in flight; this catches
                # a group that submitted its tail after the swap.
                abort_generating(
                    lambda tag: tag.checkpoint_hash != local_hash, "checkpoint_changed",
                )
                if (
                    MINER_PRUNE_CHECKPOINTS
                    and local_repo_id
                    and local_hash
                    and pruned_for != local_hash
                    and (prune_task is None or prune_task.done())
                ):
                    pruned_for = local_hash
                    prune_task = asyncio.create_task(_prune_checkpoints(
                        local_repo_id,
                        local_hash,
                        prefetcher.pinned(local_repo_id),
                    ))

                if (
                    (
                        state.checkpoint_repo_id
                        and local_repo_id != state.checkpoint_repo_id
                    )
                    or (
                        state.checkpoint_revision
                        and local_hash != state.checkpoint_revision
                    )
                ):
                    logger.warning(
                        "checkpoint activation incomplete; local=%s@%s "
                        "remote=%s@%s",
                        local_repo_id or "none",
                        local_hash[:12] or "none",
                        state.checkpoint_repo_id or "none",
                        (state.checkpoint_revision or "")[:12] or "none",
                    )
                    await asyncio.sleep(1)
                    continue

                if state.state != WindowState.OPEN:
                    await asyncio.sleep(1)
                    continue
                if not _state_matches_active_protocol(state):
                    logger.error(
                        "validator generation contract mismatch: local=%s/v%d "
                        "remote=%s/v%s",
                        ACTIVE_PROTOCOL_PROFILE.profile_id,
                        ACTIVE_PROTOCOL_PROFILE.protocol_version,
                        state.generation_profile_id,
                        state.protocol_version,
                    )
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # v2.3: trust the validator's per-window randomness rather
                # than recomputing locally. Empty string means the validator
                # hasn't yet finished _set_window_randomness — wait briefly.
                randomness = state.randomness
                if not randomness:
                    await asyncio.sleep(0.1)
                    continue

                if environment_yield.start_window(state.window_n):
                    logger.info(
                        "environment yield window=%d: %s; %s",
                        state.window_n,
                        environment_yield.summary(self.envs),
                        lane_values.describe(self.envs),
                    )
                    lane_values.maybe_refresh(client, url)
                for name, prompt_range in prompt_ranges.items():
                    if (
                        prefetched_ranges.get(name) != prompt_range
                        and name in self.envs
                        and warmup.ready(name)
                    ):
                        prefetched_ranges[name] = prompt_range
                        _prefetch_prompt_range(name, self.envs[name], prompt_range)
                generation_mix = _eligible_generation_mix(
                    self.mix,
                    cached_miner_state if miner_state_supported is True else None,
                )
                open_envs = {name for name, _ in generation_mix}
                abort_generating(
                    lambda tag: tag.env_name not in open_envs, "admission_closed",
                )
                if not generation_mix:
                    logger.info(
                        "all environment admission lanes are closed; waiting"
                    )
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue
                warming = warmup.pending(open_envs)
                if warming:
                    generation_mix = [
                        (name, weight) for name, weight in generation_mix
                        if name not in warming
                    ]
                    if not generation_mix:
                        if time.monotonic() - warmup_reported_at >= 15:
                            warmup_reported_at = time.monotonic()
                            logger.info(
                                "waiting for prompt sources: %s",
                                ", ".join(sorted(warming)),
                            )
                        await asyncio.sleep(1)
                        continue
                values = lane_values.values([name for name, _ in generation_mix])
                if MINER_YIELD_WEIGHTING:
                    generation_mix = environment_yield.weigh(
                        generation_mix, values=values,
                    )
                elif values:
                    generation_mix = _weigh_lanes(generation_mix, values=values)

                # Wait for a free slot, but not past one poll: the state
                # above has to stay fresh enough to abort stale groups.
                try:
                    await asyncio.wait_for(
                        generation_slots.acquire(), timeout=POLL_INTERVAL_SECONDS,
                    )
                except asyncio.TimeoutError:
                    continue

                window_key = (state.window_n, randomness, local_hash)
                if window_key != attempted_key:
                    attempted_key = window_key
                    attempted = {name: set() for name in self.envs}
                    try:
                        restored = attempt_journal.load(
                            window_n=state.window_n,
                            randomness=randomness,
                            checkpoint_hash=local_hash,
                        )
                    except Exception:
                        logger.warning(
                            "could not read the attempted-prompt journal",
                            exc_info=True,
                        )
                        restored = None
                    if restored:
                        for name, prompts in restored.items():
                            attempted.setdefault(name, set()).update(prompts)
                excluded = {
                    name: self._cooldown_per_env.get(name, set()) | attempted[name]
                    for name in self.envs
                }
                try:
                    env_name, prompt_idx = pick_env_and_prompt(
                        self.envs,
                        generation_mix,
                        excluded,
                        rng=rng,
                        randomness=randomness,
                        prompt_ranges=prompt_ranges or None,
                    )
                except RuntimeError:
                    generation_slots.release()
                    logger.info("all envs fully in cooldown; sleeping")
                    await asyncio.sleep(5)
                    continue
                attempted[env_name].add(prompt_idx)
                try:
                    attempt_journal.save(
                        window_n=state.window_n,
                        randomness=randomness,
                        checkpoint_hash=local_hash,
                        attempted=attempted,
                    )
                except Exception:
                    logger.warning(
                        "could not record attempted prompt %s/%d",
                        env_name, prompt_idx, exc_info=True,
                    )

                task = asyncio.create_task(generate_and_hand_off(
                    state=state,
                    env_name=env_name,
                    env=self.envs[env_name],
                    prompt_idx=prompt_idx,
                    randomness=randomness,
                    checkpoint_hash=local_hash,
                ))
                generating.add(task)
                task.add_done_callback(generation_finished)
                if concurrency == 1:
                    # One group at a time keeps the sequential contract:
                    # the next prompt is picked only after this one is queued.
                    await asyncio.wait({task})
                    if generation_failures:
                        raise generation_failures[0]

        return results

    def _generate_screened_group(
        self, *, env_name: str, env, problem, prompt_idx: int,
        randomness: str, checkpoint_hash: str, window_n: int,
    ) -> tuple[list[dict], list[float] | None] | None:
        """Generation-GPU half of one prompt: rollouts, then every local screen.

        Returns ``None`` (already logged) when the group should not be proved.
        Rewards are computed here for every locally-rewarded group so the proof
        thread never touches the tokenizer, which is not safe to share across
        threads.
        """
        with _cuda_device_scope(self.vllm_gpu):
            environment_spec = get_environment_spec(env_name)
            if environment_spec.interaction_mode == "episode":
                generations = self._generate_m_episode_rollouts(
                    env,
                    randomness,
                    prompt_idx=prompt_idx,
                    checkpoint_hash=checkpoint_hash,
                )
            elif self._local_zone_filter_applies(env_name, env):
                generations, screen_reason = self._generate_zone_screened_rollouts(
                    problem, randomness, env_name=env_name,
                    prompt_idx=prompt_idx, checkpoint_hash=checkpoint_hash,
                    env=env,
                )
                if screen_reason is not None:
                    logger.info(
                        "skipping before the tail: reason=%s window=%d "
                        "env=%s prompt=%d",
                        screen_reason, window_n, env_name, prompt_idx,
                    )
                    return None
            elif self._code_zone_screen_applies(env_name, env):
                generations, screen_reason = self._generate_code_screened_rollouts(
                    problem, randomness, env_name=env_name,
                    prompt_idx=prompt_idx, checkpoint_hash=checkpoint_hash,
                    env=env,
                )
                if screen_reason is not None:
                    logger.info(
                        "skipping before the tail: reason=%s window=%d "
                        "env=%s prompt=%d",
                        screen_reason, window_n, env_name, prompt_idx,
                    )
                    return None
            elif self._answer_screen_applies(env_name, env):
                generations, screen_reason = self._generate_answer_screened_rollouts(
                    problem, randomness, env_name=env_name,
                    prompt_idx=prompt_idx, checkpoint_hash=checkpoint_hash,
                    env=env,
                )
                if screen_reason is not None:
                    logger.info(
                        "skipping before the tail: reason=%s window=%d "
                        "env=%s prompt=%d",
                        screen_reason, window_n, env_name, prompt_idx,
                    )
                    return None
            else:
                generations = self._generate_m_rollouts(
                    problem, randomness, env_name=env_name,
                    prompt_idx=prompt_idx, checkpoint_hash=checkpoint_hash,
                )
        if len(generations) < M_ROLLOUTS:
            logger.warning(
                "generated %d/%d for prompt %d; skipping",
                len(generations), M_ROLLOUTS, prompt_idx,
            )
            return None

        if self._code_zone_screen_applies(env_name, env):
            code_skip = self._code_zone_screen_reason(env, problem, generations)
            if code_skip is not None:
                logger.info(
                    "skipping proof: reason=%s window=%d env=%s prompt=%d",
                    code_skip, window_n, env_name, prompt_idx,
                )
                return None
            if getattr(self, "generator", None) is not None:
                from reliquary.shared.modeling import resolve_eos_token_ids

                eos_ids = sorted(resolve_eos_token_ids(self.vllm_model, self.tokenizer))
                for index, generation in enumerate(generations):
                    generation["screen_termination"] = True
                    generation["rollout_index"] = index
                    generation["prompt_idx"] = prompt_idx
                    generation["checkpoint_hash"] = checkpoint_hash
                    generation["env_name"] = env_name
                    generation["eos_ids"] = eos_ids

        local_rewards = None
        if self._local_zone_filter_applies(env_name, env):
            from reliquary.shared.modeling import resolve_eos_token_ids

            completions, texts, local_rewards = self._score_generations(
                env, problem, generations,
            )
            skip_reason = _local_zone_skip_reason(
                env_name=env_name,
                rewards=local_rewards,
                completions=completions,
                texts=texts,
                eos_ids=resolve_eos_token_ids(self.vllm_model, self.tokenizer),
            )
            if skip_reason is not None:
                logger.info(
                    "skipping proof: reason=%s window=%d env=%s "
                    "prompt=%d correct=%d/%d",
                    skip_reason, window_n, env_name, prompt_idx,
                    sum(1 for r in local_rewards if r >= 0.5),
                    len(local_rewards),
                )
                return None
        elif (
            generations[0].get("trace") is None
            and not getattr(env, "validator_authoritative_reward", False)
        ):
            _, _, local_rewards = self._score_generations(env, problem, generations)
        return generations, local_rewards

    def _prove_group(self, group: _PreparedGroup, runtime_fingerprint):
        """Proof-GPU half: GRAIL commits for all rollouts, then the Code
        termination screen that needs the proof logits.

        Returns ``(request, None)`` to submit or ``(None, reason)`` to drop.
        """
        with _cuda_device_scope(self.proof_gpu):
            request = self.build_batch_request_from_generations(
                generations=group.generations,
                problem=group.problem,
                environment=group.env,
                randomness=group.state.randomness,
                prompt_idx=group.prompt_idx,
                window_number=group.state.window_n,
                checkpoint_revision=group.checkpoint_hash,
                runtime_fingerprint=runtime_fingerprint,
                rewards=group.rewards,
            )
        if group.env_name == "opencodeinstruct":
            upload_skip = _termination_upload_skip_reason(
                group.env_name,
                [g.get("termination_kind") for g in group.generations],
            )
            if upload_skip is not None:
                return None, upload_skip
        return request, None

    def _load_checkpoint(self, local_path: str):
        """Reload both hf_model and vllm_model from *local_path*.

        vllm_model is the fast-generation copy on ``self.vllm_gpu``;
        hf_model is the GRAIL-proof copy on ``self.proof_gpu``. The shared
        loader picks CausalLM for legacy text checkpoints and conditional
        text-only loading for Qwen3.5.
        """
        import torch

        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.shared.modeling import load_text_generation_model

        if getattr(self, "_loaded_checkpoint_path", None) == local_path:
            logger.debug("_load_checkpoint: already loaded from %s", local_path)
            return self.hf_model

        logger.info("Loading checkpoint from %s", local_path)

        def _load_one(device: str):
            return load_text_generation_model(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(device).eval()

        proof_device = f"cuda:{self.proof_gpu}"
        generation_device = (
            getattr(self, "generation_device", None) or f"cuda:{self.vllm_gpu}"
        )
        old_hf = self.hf_model
        old_gen = self.vllm_model

        # A single-device miner cannot hold both the old pair and a staged new
        # pair at once. Move the old pair to host memory before loading. Any
        # failure after that point is terminal for this process: a supervisor
        # restart reconstructs one coherent pair from the advertised revision.
        if self.proof_gpu == self.vllm_gpu:
            new_hf = None
            new_gen = None
            try:
                old_hf.to("cpu")
                old_gen.to("cpu")
                torch.cuda.empty_cache()
                new_hf = _load_one(proof_device)
                new_gen = _load_one(generation_device)
            except Exception as exc:
                del new_hf
                del new_gen
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise CheckpointActivationRestartRequired(
                    "single-device checkpoint activation requires restart"
                ) from exc

            self.hf_model = new_hf
            self.vllm_model = new_gen
            del old_hf
            del old_gen
            self._set_generator_weights(local_path)
            self._loaded_checkpoint_path = local_path
            logger.info("Checkpoint %s loaded into both models", local_path)
            return self.hf_model

        # Stage both copies before publishing either reference. If one load
        # fails, the previous generation/proof pair and checkpoint identity
        # remain active together.
        _release_cuda_memory()
        try:
            new_hf = _load_one(proof_device)
        except Exception:
            logger.exception(
                "Failed to reload hf_model from %s; keeping old model",
                local_path,
            )
            _release_cuda_memory()
            raise

        try:
            new_gen = _load_one(generation_device)
        except Exception:
            logger.exception(
                "Failed to stage vllm_model from %s; keeping the prior "
                "generation/proof pair",
                local_path,
            )
            del new_hf
            _release_cuda_memory()
            raise

        self.hf_model = new_hf
        self.vllm_model = new_gen
        del old_hf
        del old_gen
        _release_cuda_memory()

        self._set_generator_weights(local_path)
        self._loaded_checkpoint_path = local_path
        logger.info("Checkpoint %s loaded into both models", local_path)
        return self.hf_model

    def _host_turn(self) -> _HostTurn:
        # Engines built with object.__new__ (tests) skip __init__.
        return self.__dict__.setdefault("_host_turn_state", _HostTurn())

    def _environment_yield(self) -> _EnvironmentYield:
        return self.__dict__.setdefault("_environment_yield_state", _EnvironmentYield())

    def _resolved_generation_concurrency(self) -> int:
        if getattr(self, "generator", None) is None:
            return 1
        return max(1, int(getattr(self, "generation_concurrency", 1) or 1))

    def _generate_tagged_group(self, tag: _GenerationTag, **kwargs):
        """``_generate_screened_group`` on a worker thread, labelled for abort."""
        generator = getattr(self, "generator", None)
        tagged = (
            generator.tagged(tag)
            if generator is not None and hasattr(generator, "tagged")
            else contextlib.nullcontext()
        )
        with self._host_turn().hold(), tagged:
            return self._generate_screened_group(**kwargs)

    def _set_generator_weights(self, checkpoint_path: str | None = None) -> None:
        """Copy the freshly published proof weights into the vLLM engine.

        The engine is never rebuilt, so a failed copy leaves it neither on the
        old checkpoint nor the new one; only a restart gives a coherent pair.
        A generator with an engine in another process reads the checkpoint
        from ``checkpoint_path`` instead of these tensors.
        """
        generator = getattr(self, "generator", None)
        if generator is None:
            return
        try:
            if getattr(generator, "loads_checkpoint_path", False):
                generator.set_weights(
                    self.hf_model.named_parameters(),
                    checkpoint_path=checkpoint_path,
                )
            else:
                generator.set_weights(self.hf_model.named_parameters())
        except Exception as exc:
            raise CheckpointActivationRestartRequired(
                "vLLM weights could not be set in place"
            ) from exc

    def _generate_zone_screened_rollouts(
        self, problem, randomness, *, env_name: str, prompt_idx: int,
        checkpoint_hash: str, env,
    ) -> tuple[list[dict] | None, str | None]:
        """Finish a Math group only when its forced prefix is worth proving."""
        from reliquary.constants import max_new_tokens_for_environment
        from reliquary.shared.modeling import resolve_eos_token_ids

        eos_ids = resolve_eos_token_ids(self.vllm_model, self.tokenizer)
        max_new_tokens = min(
            int(self.max_new_tokens), max_new_tokens_for_environment(env_name),
        )

        def screen(generations):
            completions, texts, rewards = self._score_generations(
                env, problem, generations,
            )
            reason = _prefix_screen_reason(
                env_name=env_name,
                rewards=rewards,
                completions=completions,
                texts=texts,
                eos_ids=eos_ids,
                max_new_tokens=max_new_tokens,
            )
            if reason is not None:
                return reason, False
            unanimous = (
                len(set(rewards)) == 1
                and not _uncertain_indices(env_name, completions, texts, eos_ids)
            )
            return None, unanimous

        return self._generate_staged_rollouts(
            problem, randomness, env_name=env_name, prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash, screen=screen,
        )

    def _generate_code_screened_rollouts(
        self, problem, randomness, *, env_name: str, prompt_idx: int,
        checkpoint_hash: str, env,
    ) -> tuple[list[dict] | None, str | None]:
        """Finish a Code group only when its forced prefix can still split
        the public cases. Any failure to score generates the whole group."""
        from reliquary.constants import max_new_tokens_for_environment
        from reliquary.shared.modeling import resolve_eos_token_ids

        eos_ids = resolve_eos_token_ids(self.vllm_model, self.tokenizer)
        cap = max_new_tokens_for_environment(env_name)

        def screen(generations):
            kinds = [
                _structural_termination_kind(
                    list(generation["tokens"]),
                    int(generation["prompt_length"]),
                    eos_ids,
                    cap,
                )
                for generation in generations
            ]
            structural = _termination_upload_skip_reason(env_name, kinds)
            if structural is not None:
                return structural, False
            try:
                rewards = self._code_screen_rewards(env, problem, generations)
            except Exception:
                logger.warning("code prefix screen failed; finishing the group", exc_info=True)
                return None, False
            unanimous = (
                rewards is not None
                and len(set(rewards)) == 1
                and all(kind == "eos" for kind in kinds)
            )
            return None, unanimous

        return self._generate_staged_rollouts(
            problem, randomness, env_name=env_name, prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash, screen=screen,
        )

    def _generate_answer_screened_rollouts(
        self, problem, randomness, *, env_name: str, prompt_idx: int,
        checkpoint_hash: str, env,
    ) -> tuple[list[dict] | None, str | None]:
        """Finish a validator-scored JSON-answer group only while it can split.

        The validator grades ``tokenizer.decode(completion)`` with the same
        pinned checker and values a truncated rollout at whichever reward
        hurts the zone most, so only clean rollouts can split a prefix and the
        full group faces the same gate admission applies. Any scoring failure
        finishes the group.
        """
        from reliquary.constants import (
            MAX_TRUNCATED_PER_SUBMISSION,
            MAX_TRUNCATED_PER_SUBMISSION_BY_ENV,
        )
        from reliquary.shared.modeling import resolve_eos_token_ids

        eos = {int(t) for t in resolve_eos_token_ids(self.vllm_model, self.tokenizer)}
        allowance = MAX_TRUNCATED_PER_SUBMISSION_BY_ENV.get(
            env_name, MAX_TRUNCATED_PER_SUBMISSION,
        )

        def score(generations):
            try:
                return self._score_generations(env, problem, generations)
            except Exception:
                logger.warning(
                    "answer screen failed; finishing the group", exc_info=True,
                )
                return None

        def screen(generations):
            scored = score(generations)
            if scored is None:
                return None, False
            completions, _, rewards = scored
            clean = [
                reward for reward, completion in zip(rewards, completions)
                if any(int(token) in eos for token in completion)
            ]
            if len(rewards) - len(clean) > allowance:
                return "too_many_truncated", False
            return None, len(set(clean)) <= 1

        generations, reason = self._generate_staged_rollouts(
            problem, randomness, env_name=env_name, prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash, screen=screen,
        )
        if generations is not None and len(generations) >= M_ROLLOUTS:
            scored = score(generations)
            if scored is not None:
                completions, texts, rewards = scored
                reason = _local_zone_skip_reason(
                    env_name=env_name,
                    rewards=rewards,
                    completions=completions,
                    texts=texts,
                    eos_ids=eos,
                )
                if reason is not None:
                    return None, reason
        return generations, reason

    def _generate_staged_rollouts(
        self, problem, randomness, *, env_name: str, prompt_idx: int,
        checkpoint_hash: str, screen,
    ) -> tuple[list[dict] | None, str | None]:
        """One group in forced-index order, scored at each stop.

        Every stage draws the next rollout indices, so the finished group is
        the protocol group, not a second sample. ``screen(generations)``
        returns ``(reason, unanimous)``: a reason drops the group, a split
        result generates the rest at once, and a unanimous prefix that
        reaches ``MINER_UNANIMOUS_DROP_ROLLOUTS`` is dropped.
        """
        from reliquary.constants import MINER_UNANIMOUS_DROP_ROLLOUTS
        from reliquary.shared.modeling import resolve_eos_token_ids

        eos = {int(t) for t in resolve_eos_token_ids(self.vllm_model, self.tokenizer)}

        def extend(generations: list[dict], stop: int) -> list[dict]:
            if not generations and stop >= M_ROLLOUTS:
                return self._generate_m_rollouts(
                    problem, randomness, env_name=env_name, prompt_idx=prompt_idx,
                    checkpoint_hash=checkpoint_hash,
                )
            truncated = sum(
                not any(int(token) in eos for token in g["tokens"][g["prompt_length"]:])
                for g in generations
            )
            return generations + self._generate_m_rollouts(
                problem, randomness, env_name=env_name, prompt_idx=prompt_idx,
                checkpoint_hash=checkpoint_hash,
                rollout_indices=list(range(len(generations), stop)),
                truncated_so_far=truncated,
            )

        generations: list[dict] = []
        for stop in _unanimous_screen_stops(M_ROLLOUTS):
            generations = extend(generations, stop)
            if stop >= M_ROLLOUTS:
                break
            reason, unanimous = screen(generations)
            if reason is not None:
                return None, reason
            if not unanimous:
                return extend(generations, M_ROLLOUTS), None
            if MINER_UNANIMOUS_DROP_ROLLOUTS and stop >= MINER_UNANIMOUS_DROP_ROLLOUTS:
                return None, "unanimous_prefix"
        return generations, None

    def _generate_m_rollouts(
        self, problem, randomness, *, env_name: str | None = None,
        prompt_idx: int, checkpoint_hash: str,
        rollout_indices: list[int] | None = None,
        truncated_so_far: int = 0,
    ) -> list[dict]:
        """Generate M_ROLLOUTS completions at T_PROTO in one batched call.

        One .generate() with batch shape (M_ROLLOUTS, prompt_len) is ~5-7×
        faster on GPU than M_ROLLOUTS serial calls — the matmul tiling
        utilizes far more of the GPU's compute. Each row samples
        independently (do_sample=True), so GRPO-group semantics are
        preserved. Each output row is truncated at its first post-prompt
        EOS so trailing batch-padding (which HF pads with pad_token_id =
        eos_token_id) is not carried downstream — otherwise the validator's
        GRAIL forward pass would see extra EOS tokens the miner didn't
        "generate" in the usual sense.
        """
        import torch

        from reliquary.constants import (
            BFT_ANSWER_BUDGET,
            BFT_ENABLED,
            BFT_FORCE_ANSWER,
            BFT_THINKING_BUDGET,
            max_new_tokens_for_environment,
            max_truncated_for_environment,
            thinking_for_environment,
        )
        from reliquary.miner.forced_seed_sampler import (
            ForcedSeedLogitsProcessor, forced_seed_generate_kwargs,
        )
        from reliquary.protocol.tokens import encode_prompt
        from reliquary.shared.modeling import (
            first_eos_index,
            force_close_token_ids,
            resolve_eos_token_ids,
            think_close_token_ids,
        )

        hotkey = self.wallet.hotkey.ss58_address
        # Resolved before the prompt is encoded: whether the template opens a
        # reasoning block is per environment, and the validator renders this
        # same prompt with the same lookup.
        prompt_env_name = env_name
        if prompt_env_name is None:
            prompt_env_name = getattr(getattr(self, "env", None), "name", None)
        prompt_tokens = encode_prompt(
            self.tokenizer,
            problem["prompt"],
            thinking=thinking_for_environment(str(prompt_env_name or "")),
        )
        prompt_length = len(prompt_tokens)
        eos_ids = resolve_eos_token_ids(self.vllm_model, self.tokenizer)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None and eos_ids:
            pad_token_id = min(eos_ids)
        active_env_name = env_name
        if active_env_name is None:
            active_env_name = getattr(getattr(self, "env", None), "name", None)
        protocol_cap = max_new_tokens_for_environment(str(active_env_name or ""))
        environment_cap = min(int(self.max_new_tokens), protocol_cap)
        # The validator's cap counts prompt and completion together
        # (prompt_length + completion_length >= cap is "truncated"), so tokens
        # past cap - prompt_length only lengthen a rollout that is already a
        # valid truncation.
        completion_budget = min(environment_cap, max(1, protocol_cap - prompt_length))
        if active_env_name is None:
            # Legacy single-environment callers omitted env_name and were
            # always the active Math lane. Preserve that invocation contract.
            bft_applicable = BFT_ENABLED
        else:
            try:
                bft_applicable = BFT_ENABLED and (
                    get_environment_spec(
                        str(active_env_name)
                    ).termination_policy == "math_bft"
                )
            except ValueError:
                bft_applicable = False

        indices = (
            list(range(M_ROLLOUTS)) if rollout_indices is None else list(rollout_indices)
        )
        if not indices:
            return []
        if bft_applicable and rollout_indices is not None:
            raise RuntimeError("BFT generation does not screen a rollout prefix")

        generator = getattr(self, "generator", None)
        if generator is not None and not bft_applicable:
            # A rollout with no stop token is truncated or bad_termination;
            # past the allowance, admission and the upload screen both refuse
            # the group, so the generator abandons it at that point.
            max_truncated = (
                None if active_env_name is None
                else max(
                    0,
                    max_truncated_for_environment(str(active_env_name))
                    - int(truncated_so_far),
                )
            )
            completions = generator.generate(
                prompt_tokens,
                randomness=randomness,
                prompt_idx=prompt_idx,
                checkpoint_hash=checkpoint_hash,
                rollouts=len(indices),
                max_new_tokens=completion_budget,
                eos_ids=sorted(eos_ids),
                rollout_indices=indices,
                max_truncated=max_truncated,
                wait=self._host_turn().wait,
            )
            return [
                {
                    "tokens": prompt_tokens + list(completion),
                    "prompt_length": prompt_length,
                    "forced": False,
                }
                for completion in completions
            ]

        with torch.no_grad():
            input_tensor = torch.tensor(
                [prompt_tokens] * len(indices),
                device=getattr(self.vllm_model, "device", "cpu"),
            )
            attention_mask = torch.ones_like(input_tensor)
            # Force phase-1 sampling onto the protocol seed stream: the
            # processor applies the T_PROTO/top_k/top_p warp itself and picks
            # the inverse-CDF token, so HF's own warpers are stripped and
            # do_sample is off (see forced_seed_generate_kwargs). Row r is
            # rollout index r, resuming at completion offset 0.
            base_kwargs = {
                "max_new_tokens": (
                    min(environment_cap, BFT_THINKING_BUDGET)
                    if bft_applicable else completion_budget
                ),
                "pad_token_id": pad_token_id,
                "attention_mask": attention_mask,
            }
            if eos_ids:
                base_kwargs["eos_token_id"] = sorted(eos_ids)
            phase1_proc = ForcedSeedLogitsProcessor(
                randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
                checkpoint_hash=checkpoint_hash,
                rollout_indices=indices,
                base_offsets=[0] * len(indices), start_len=prompt_length,
            )
            outputs = self.vllm_model.generate(
                input_tensor,
                **forced_seed_generate_kwargs(base_kwargs, phase1_proc),
            )

            if bft_applicable and BFT_FORCE_ANSWER:
                # Phase-2 answer generation continues on the same forced stream
                # (identity threaded so it resumes at each row's offset). Skipped
                # under clean-cap (BFT_FORCE_ANSWER=False): a rollout that did not
                # close </think> within the phase-1 budget is left truncated and
                # grades as bad_termination instead of being force-answered.
                phase2_kwargs = {"pad_token_id": pad_token_id}
                if eos_ids:
                    phase2_kwargs["eos_token_id"] = sorted(eos_ids)
                return _bft_assemble_rollouts(
                    model=self.vllm_model,
                    phase1_tensor=outputs,
                    prompt_tokens=prompt_tokens,
                    think_close_ids=set(think_close_token_ids(self.tokenizer)),
                    force_ids=force_close_token_ids(self.tokenizer),
                    eos_ids=eos_ids,
                    answer_budget=BFT_ANSWER_BUDGET,
                    randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
                    checkpoint_hash=checkpoint_hash,
                    gen_kwargs=phase2_kwargs,
                )
        rollouts = []
        for row in range(len(indices)):
            seq = outputs[row].tolist()
            gen = seq[prompt_length:]
            first_eos = first_eos_index(gen, eos_ids)
            if first_eos is not None:
                gen = gen[: first_eos + 1]
            rollouts.append({
                "tokens": prompt_tokens + gen,
                "prompt_length": prompt_length,
                "forced": False,
            })
        return rollouts

    def _generate_m_episode_rollouts(
        self,
        env,
        randomness: str,
        *,
        prompt_idx: int,
        checkpoint_hash: str,
    ) -> list[dict]:
        """Generate M complete canonical episodes, one turn at a time."""

        from reliquary.environment.agentic.renderers import renderer_for
        from reliquary.environment.agentic.runner import EpisodeRunner
        from reliquary.miner.episode_policy import HFEpisodePolicy

        profile = ACTIVE_PROTOCOL_PROFILE.environments[env.name].episode
        if profile is None:
            raise RuntimeError(f"episode profile missing for {env.name}")

        def encode(text: str) -> list[int]:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            return list(getattr(encoded, "ids", encoded))

        # The environment's declared dialect — the same lookup the validator
        # uses to re-render this episode, so the two sides cannot disagree about
        # which renderer produced the transcript it compares byte for byte.
        renderer_id = str(get_environment_spec(env.name).renderer_id)
        renderer = renderer_for(renderer_id, encode)
        task = env.get_task(prompt_idx)
        hotkey = self.wallet.hotkey.ss58_address
        generations: list[dict] = []
        for rollout_index in range(M_ROLLOUTS):
            seed_material = (
                f"{randomness}:{hotkey}:{env.name}:{prompt_idx}:{rollout_index}"
            ).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
            policy = HFEpisodePolicy(
                model=self.vllm_model,
                tokenizer=self.tokenizer,
                randomness=randomness,
                hotkey=hotkey,
                prompt_idx=prompt_idx,
                checkpoint_hash=checkpoint_hash,
                rollout_index=rollout_index,
                max_action_tokens=profile.max_action_tokens,
                max_episode_tokens=profile.max_episode_tokens,
            )
            trace = EpisodeRunner(
                renderer=renderer,
                max_turns=min(profile.max_turns, env.max_turns),
                max_episode_tokens=profile.max_episode_tokens,
                max_observation_bytes=profile.max_observation_bytes,
            ).run(env, task, seed=seed, policy=policy)
            if not trace.assistant_spans:
                raise RuntimeError("episode generated no assistant actions")
            generations.append({
                "tokens": list(trace.tokens),
                "prompt_length": int(trace.assistant_spans[0][0]),
                "forced": False,
                "trace": trace,
                "renderer_id": renderer_id,
            })
        return generations

    def _code_zone_screen_applies(self, env_name: str, env) -> bool:
        """Code only. Math already has its own local zone screen.

        The screen reads public cases and does not touch the submitted
        reward, which stays the validator placeholder.
        """
        if env_name != "opencodeinstruct":
            return False
        return callable(getattr(env, "admission_reward_cases", None))

    def _code_zone_screen_reason(self, env, problem, generations: list[dict]) -> str | None:
        """``out_of_zone`` when every scored rollout agrees, else None.

        None also means the score could not be trusted, in which case the
        group is submitted unchanged.
        """
        from reliquary.shared.modeling import resolve_eos_token_ids

        try:
            from reliquary.constants import max_new_tokens_for_environment

            eos_ids = resolve_eos_token_ids(self.vllm_model, self.tokenizer)
            cap = max_new_tokens_for_environment("opencodeinstruct")
            for generation in generations:
                generation["termination_kind"] = _structural_termination_kind(
                    list(generation["tokens"]),
                    int(generation["prompt_length"]),
                    eos_ids,
                    cap,
                )
            structural = _termination_upload_skip_reason(
                "opencodeinstruct",
                [generation["termination_kind"] for generation in generations],
            )
            if structural is not None:
                return structural
            rewards = self._code_screen_rewards(env, problem, generations)
            if rewards is None:
                return None
            completions = [
                list(generation["tokens"][generation["prompt_length"]:])
                for generation in generations
            ]
            texts = [self.tokenizer.decode(completion) for completion in completions]
            return _local_zone_skip_reason(
                env_name="opencodeinstruct",
                rewards=rewards,
                completions=completions,
                texts=texts,
                eos_ids=eos_ids,
            )
        except Exception:
            logger.warning("code zone screen failed; submitting the group", exc_info=True)
            return None

    def _code_screen_rewards(self, env, problem, generations: list[dict]) -> list[float] | None:
        """Public-case pass fraction per rollout, or None when untrusted."""
        from reliquary.environment.opencodeinstruct import _entry_function_name, _extract_python
        from reliquary.miner.code_zone_screen import score_group

        cases = list(env.admission_reward_cases(problem) or [])
        if not cases:
            return None
        entry = _entry_function_name(cases)
        codes = [
            _extract_python(
                self.tokenizer.decode(generation["tokens"][generation["prompt_length"]:]),
                entry_name=entry,
            )
            for generation in generations
        ]
        # The cases run in a subprocess; other groups may use the tokenizer.
        with self._host_turn().released():
            rewards = score_group(codes, cases)
        if rewards is None or len(rewards) != len(generations):
            return None
        return rewards

    def _answer_screen_applies(self, env_name: str, env) -> bool:
        from reliquary.constants import BFT_ENABLED, MINER_ANSWER_SCREEN

        if not MINER_ANSWER_SCREEN or BFT_ENABLED:
            return False
        if not callable(getattr(env, "compute_reward", None)):
            return False
        try:
            spec = get_environment_spec(env_name)
        except ValueError:
            return False
        return (
            spec.validator_authoritative_reward
            and spec.external_distribution is not None
            and spec.interaction_mode == "single_turn"
            and spec.final_answer_policy == "json"
        )

    def _local_zone_filter_applies(self, env_name: str, env) -> bool:
        from reliquary.constants import BFT_ENABLED, MINER_LOCAL_ZONE_FILTER

        if not MINER_LOCAL_ZONE_FILTER or BFT_ENABLED:
            return False
        if getattr(env, "validator_authoritative_reward", False):
            return False
        try:
            spec = get_environment_spec(env_name)
        except ValueError:
            return False
        return (
            not spec.validator_authoritative_reward
            and spec.interaction_mode == "single_turn"
        )

    def _score_generations(self, env, problem, generations: list[dict]):
        """Completion tokens, decoded texts and local rewards, one per rollout.

        Decoding and scoring match ``_build_rollout_submission`` exactly, so
        these rewards are the claims that would be submitted.
        """
        completions = [
            list(g["tokens"][g["prompt_length"]:]) for g in generations
        ]
        texts = [self.tokenizer.decode(c) for c in completions]
        rewards = [float(env.compute_reward(problem, t)) for t in texts]
        return completions, texts, rewards

    def _build_rollout_submission(
        self, generation, problem, randomness, *, env=None, reward=None,
    ):
        """Build a RolloutSubmission: completion + claimed reward + GRAIL commit."""
        active_env = env if env is not None else self.env
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        if generation.get("trace") is not None:
            reward = 0.0
        elif getattr(active_env, "validator_authoritative_reward", False):
            reward = 0.0
        elif reward is not None:
            reward = float(reward)
        else:
            completion_text = self.tokenizer.decode(all_tokens[prompt_length:])
            reward = active_env.compute_reward(problem, completion_text)

        commit = self._build_grail_commit(generation, randomness)
        return RolloutSubmission(
            tokens=all_tokens,
            reward=reward,
            commit=commit,
            env_name=active_env.name,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_grail_commit(self, generation: dict, randomness: str) -> dict:
        """Construct a GRAIL proof commit dict from a generation dict.

        Reproduces the proof construction:
          - HF forward pass for hidden_states + logits
          - Commitment batch via GRAILVerifier
          - log-softmax token log-probs
          - Signature via sign_commit_binding
        """
        import torch

        from reliquary.constants import (
            GRAIL_EPISODE_PROOF_VERSION,
            GRAIL_PROOF_VERSION,
        )
        from reliquary.protocol.signatures import (
            sign_commit_binding,
            sign_episode_commit_binding,
        )
        from reliquary.shared.forward import forward_single_layer
        from reliquary.validator.verifier import (
            _LazyLogitRows,
            _lm_head_vocab_size,
        )

        all_tokens: list[int] = generation["tokens"]
        prompt_length: int = generation["prompt_length"]

        # The validator projects lm_head row by row instead of building the
        # [seq_len, vocab] block (~10 GB bf16 at the 32k cap); reading rows
        # the same way keeps the proof GPU's peak near the model's own size.
        lm_head = getattr(self.hf_model, "lm_head", None)
        vocab_size = None if lm_head is None else _lm_head_vocab_size(lm_head)
        proof_input = torch.tensor(
            [all_tokens], device=f"cuda:{self.proof_gpu}"
        )
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX,
                materialize_logits=vocab_size is None,
            )

        hidden_states = hidden_states[0]  # [seq_len, hidden_dim]
        logit_rows = (
            _LazyLogitRows(hidden_states, lm_head, vocab_size)
            if logits is None else logits[0]
        )

        # Build commitments
        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        trace = generation.get("trace")
        policy_positions = (
            [
                position
                for start, end in trace.assistant_spans
                for position in range(start, end)
            ]
            if trace is not None
            else list(range(prompt_length, len(all_tokens)))
        )
        # fp32 log_softmax to match the validator and reduce tail-token drift.
        token_logprobs = _policy_token_logprobs(
            logit_rows, all_tokens, policy_positions,
        )
        if generation.get("screen_termination"):
            try:
                from reliquary.constants import max_new_tokens_for_environment
                from reliquary.shared.modeling import resolve_eos_token_ids

                generation["termination_kind"] = _proof_termination_kind(
                    tokens=all_tokens,
                    prompt_length=prompt_length,
                    eos_ids=(
                        generation.get("eos_ids")
                        or resolve_eos_token_ids(self.hf_model, self.tokenizer)
                    ),
                    cap=max_new_tokens_for_environment(
                        str(generation.get("env_name") or "opencodeinstruct"),
                    ),
                    logits=logit_rows,
                    randomness=randomness,
                    prompt_idx=int(generation["prompt_idx"]),
                    checkpoint_hash=str(generation["checkpoint_hash"]),
                    rollout_index=int(generation["rollout_index"]),
                    token_logprobs=token_logprobs,
                )
            except Exception:
                logger.warning(
                    "code termination screen failed; submitting the group",
                    exc_info=True,
                )
                generation["termination_kind"] = None
        del logits, logit_rows

        model_name: str = getattr(self.hf_model, "name_or_path", "unknown")
        rollout_metadata = _rollout_metadata(generation, token_logprobs)
        if trace is not None:
            reward = trace.reward
            if reward is None:
                raise RuntimeError("episode trace has no reward report")
            episode = {
                "schema_version": trace.schema,
                # What this episode was actually rendered with, recorded when
                # the renderer was chosen rather than restated here.
                "renderer_id": generation["renderer_id"],
                "task_id": trace.task_id,
                "seed": trace.seed,
                "actions": [action.to_wire() for action in trace.actions],
                "assistant_spans": [list(span) for span in trace.assistant_spans],
                "observation_digests": list(trace.observation_digests),
                "termination_reason": trace.termination_reason,
                "state_digest": reward.state_digest,
                "trace_digest": trace.trace_digest,
            }
            rollout_metadata["episode"] = episode
            signature = sign_episode_commit_binding(
                all_tokens,
                randomness,
                model_name,
                LAYER_INDEX,
                commitments,
                episode,
                self.wallet,
            )
            proof_version = GRAIL_EPISODE_PROOF_VERSION
        else:
            signature = sign_commit_binding(
                all_tokens, randomness, model_name, LAYER_INDEX,
                commitments, self.wallet,
            )
            proof_version = GRAIL_PROOF_VERSION

        commit = {
            "tokens": all_tokens,
            "commitments": commitments,
            "proof_version": proof_version,
            "model": {"name": model_name, "layer_index": LAYER_INDEX},
            "signature": signature.hex(),
            "beacon": {"randomness": randomness},
            "rollout": rollout_metadata,
        }
        # The same final hidden states GRAIL just used: no extra forward.
        toploc = toploc_proof(ACTIVE_PROTOCOL_PROFILE)
        if toploc is not None:
            commit["toploc_proofs"] = completion_proofs_b64(
                hidden_states, prompt_length, len(all_tokens),
                chunk_tokens=toploc.chunk_tokens, topk=toploc.topk,
            )
        return commit
