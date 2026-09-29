"""Generate the protocol's forced draw on vLLM instead of transformers.

The draw is public: token t of rollout r is the inverse-CDF pick of ``u_at`` over
the warped policy, and the validator teacher-forces the same pick from its own
forward. Any engine may produce it, so long as it reads the same logits — which
is why this processor computes warp + pick itself and hands vLLM a one-hot,
with the request sampled greedily so it can select nothing else.

Measured on an H200 (20-09, Teutonic-I, code prompts): sequences generated here
and verified by the transformers validator agree on 0.964-0.980 of stochastic
positions per group, worst rollout 0.941, no group or rollout under its floor
and no terminal pick refused — level with the batched transformers path it
replaces (0.970-0.974). Throughput at 256 concurrent rollouts is 6,050 tok/s
against 507 for a 16-wide transformers batch, because a transformers decode step
costs ~31 ms whatever the batch, so the win is concurrency, not the engine.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from typing import Any

import torch

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import _warp_batch, u_at
from reliquary.shared.modeling import first_eos_index

logger = logging.getLogger(__name__)

try:  # vLLM is an optional miner dependency; the class must stay importable.
    from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
except ImportError:  # pragma: no cover - exercised on hosts without vLLM
    BatchUpdate = Any  # type: ignore[assignment,misc]
    LogitsProcessor = object  # type: ignore[assignment,misc]

FORCED_SEED_KEY = "forced_seed"


class ForcedSeedVLLMProcessor(LogitsProcessor):  # type: ignore[misc,valid-type]
    """One-hot the forced pick for every running request, in one batched pass.

    vLLM addresses requests by their slot in the persistent batch, and reuses a
    slot as soon as a request leaves it, so the per-slot state follows the
    engine's own removed/added/moved order rather than a request id.
    """

    def __init__(self, vllm_config: Any = None, device: Any = None,
                 is_pin_memory: bool = False) -> None:
        self._slots: dict[int, dict[str, Any]] = {}

    def is_argmax_invariant(self) -> bool:
        """The pick is not the argmax: this processor decides the token."""
        return False

    def update_state(self, batch_update: "BatchUpdate | None") -> None:
        if batch_update is None:
            return
        for index in getattr(batch_update, "removed", ()):
            self._slots.pop(_slot_index(index), None)
        for added in getattr(batch_update, "added", ()):
            index, params = added[0], added[1]
            slot = _slot_index(index)
            self._slots.pop(slot, None)
            forced = dict(getattr(params, "extra_args", None) or {}).get(FORCED_SEED_KEY)
            if forced:
                self._slots[slot] = {
                    "randomness": str(forced["randomness"]),
                    "prompt_idx": int(forced["prompt_idx"]),
                    "checkpoint_hash": str(forced["checkpoint_hash"]),
                    "rollout_index": int(forced["rollout_index"]),
                    "offset": int(forced.get("base_offset", 0)),
                }
        for moved in getattr(batch_update, "moved", ()):
            source, destination = _slot_index(moved[0]), _slot_index(moved[1])
            swap = "SWAP" in str(moved[2]).upper() if len(moved) > 2 else False
            from_state = self._slots.pop(source, None)
            to_state = self._slots.pop(destination, None)
            if from_state is not None:
                self._slots[destination] = from_state
            if swap and to_state is not None:
                self._slots[source] = to_state

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        slots = [slot for slot in sorted(self._slots) if slot < logits.shape[0]]
        if not slots:
            return logits
        draws = [
            u_at(state["randomness"], state["prompt_idx"], state["checkpoint_hash"],
                 state["rollout_index"], state["offset"])
            for state in (self._slots[slot] for slot in slots)
        ]
        rows = torch.tensor(slots, device=logits.device)
        probs = _warp_batch(logits[rows], t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
        cdf = torch.cumsum(probs, dim=-1)
        uniforms = torch.tensor(draws, device=cdf.device, dtype=cdf.dtype).unsqueeze(1)
        picks = torch.searchsorted(cdf, uniforms, right=True).squeeze(1)
        picks = picks.clamp(max=probs.shape[-1] - 1)
        forced = torch.full_like(logits, float("-inf"))
        forced[rows, picks] = 0.0
        for slot in slots:
            self._slots[slot]["offset"] += 1
        return forced


def _install_transformers5_tokenizer_compat() -> None:
    """vLLM 0.10 caches a tokenizer property transformers 5 removed.

    ``all_special_tokens_extended`` used to keep AddedToken objects. The
    string list ``all_special_tokens`` is what this engine actually reads.
    """
    import vllm.transformers_utils.tokenizer as tokenizer_mod

    original = tokenizer_mod.get_cached_tokenizer
    if getattr(original, "_reliquary_compat", False):
        return

    def get_cached_tokenizer(tokenizer):
        if not hasattr(tokenizer, "all_special_tokens_extended"):
            tokenizer.all_special_tokens_extended = list(tokenizer.all_special_tokens)
        return original(tokenizer)

    get_cached_tokenizer._reliquary_compat = True  # type: ignore[attr-defined]
    tokenizer_mod.get_cached_tokenizer = get_cached_tokenizer


def gpu_memory_utilization_for(
    device: int, reserve_bytes: float, *, mem_get_info: Any = None,
    cap: float = 0.92, floor: float = 0.2,
) -> float:
    """vLLM's ``gpu_memory_utilization`` sized from what is free right now.

    vLLM reads the fraction against the device's total memory and refuses to
    start when that exceeds free memory, so a fixed fraction either wastes a
    card that holds nothing else or fails beside a model already resident.
    """
    if mem_get_info is None:
        mem_get_info = torch.cuda.mem_get_info
    free, total = mem_get_info(device)
    if total <= 0:
        return floor
    fraction = (float(free) - float(reserve_bytes)) / float(total)
    return round(min(cap, max(floor, fraction)), 3)


def _slot_index(value: Any) -> int:
    return int(value[0] if isinstance(value, (tuple, list)) else value)


def forced_seed_extra_args(
    *, randomness: str, prompt_idx: int, checkpoint_hash: str,
    rollout_index: int, base_offset: int = 0,
) -> dict[str, Any]:
    """What one request tells the processor about its place in the draw."""
    return {
        FORCED_SEED_KEY: {
            "randomness": randomness,
            "prompt_idx": int(prompt_idx),
            "checkpoint_hash": checkpoint_hash,
            "rollout_index": int(rollout_index),
            "base_offset": int(base_offset),
        }
    }


def _load_weights_on_worker(worker: Any, weights: list[tuple[str, torch.Tensor]]) -> list[str]:
    """Copy ``weights`` into the worker's live parameters; return any left unset.

    vLLM's ``load_weights`` fuses q/k/v and gate/up into its stacked parameters
    and writes with ``copy_``, so parameter storage (which the captured CUDA
    graphs point at) never moves.
    """
    model = worker.model_runner.get_model()
    device = next(model.parameters()).device
    loaded = model.load_weights(
        (name, tensor.to(device, non_blocking=True)) for name, tensor in weights
    )
    torch.cuda.synchronize(device)
    expected = {name for name, _ in model.named_parameters()}
    return sorted(expected - set(loaded or ()))


class VLLMRolloutGenerator:
    """A group of rollouts for one prompt, generated concurrently on vLLM.

    The engine is built once for the life of the process. Every checkpoint,
    including the first when built with ``load_format="dummy"``, arrives through
    :meth:`set_weights` from tensors the miner already holds, so compiled
    graphs, the KV cache and the scheduler stay up and nothing is re-read from
    disk.
    """

    def __init__(
        self, model_path: str, *, revision: str | None = None,
        max_model_len: int | None = None, gpu_memory_utilization: float = 0.85,
        max_num_seqs: int = 256, enforce_eager: bool = False,
        load_format: str | None = None,
        engine: Any | None = None, sampling_params_class: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.revision = revision
        self._settings = {
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
            "load_format": load_format,
        }
        self._sampling_params_class = sampling_params_class
        self._llm = engine if engine is not None else self._build(model_path, revision)

    def _build(self, model_path: str, revision: str | None) -> Any:
        # set_weights hands live CUDA tensors to the worker. With the engine
        # core in a subprocess those would be pickled through ZMQ (~8 GB per
        # checkpoint); in-process they are passed by reference.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM  # local import: optional dependency

        _install_transformers5_tokenizer_compat()
        settings = {
            key: value for key, value in self._settings.items() if value is not None
        }
        engine = LLM(
            model=model_path, revision=revision, dtype="bfloat16",
            logits_processors=[ForcedSeedVLLMProcessor], **settings,
        )
        logger.info(
            "vLLM generation backend ready (%s%s)", model_path,
            f"@{revision}" if revision else "",
        )
        return engine

    def _require_in_process_engine(self) -> None:
        client = getattr(getattr(self._llm, "llm_engine", None), "engine_core", None)
        if client is not None and type(client).__name__ != "InprocClient":
            raise RuntimeError(
                "vLLM engine core runs in a subprocess "
                f"({type(client).__name__}); set_weights needs "
                "VLLM_ENABLE_V1_MULTIPROCESSING=0"
            )

    def set_weights(self, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Write a checkpoint's tensors into the running engine in place.

        ``named_tensors`` uses Hugging Face parameter names (what
        ``model.named_parameters()`` yields on the proof copy). Raises when any
        vLLM parameter was not covered, since generating from a half-updated
        model would produce proofs the validator rejects.
        """
        started = time.monotonic()
        self._require_in_process_engine()
        weights = [(name, tensor.detach()) for name, tensor in named_tensors]
        results = self._llm.collective_rpc(_load_weights_on_worker, args=(weights,))
        missing = sorted({name for result in results or () for name in result or ()})
        if missing:
            raise RuntimeError(
                f"vLLM weight update left {len(missing)} parameters unset "
                f"(first: {missing[:3]})"
            )
        # Cached prefixes were computed with the previous weights.
        self._llm.reset_prefix_cache()
        logger.info(
            "vLLM weights set in place: %d tensors in %.2fs",
            len(weights), time.monotonic() - started,
        )

    def generate(
        self, prompt_tokens: list[int], *, randomness: str, prompt_idx: int,
        checkpoint_hash: str, rollouts: int, max_new_tokens: int,
        eos_ids: list[int], rollout_indices: list[int] | None = None,
    ) -> list[list[int]]:
        """Completion token ids per rollout, truncated at their first stop token.

        vLLM keeps the token that stopped a request (measured 20-09), which the
        protocol needs: the terminal pick is checked on it. The truncation is the
        transformers path's, so a stop token appearing mid-batch cannot carry
        padding downstream either way.
        """
        sampling_params = self._sampling_params_class
        if sampling_params is None:
            from vllm import SamplingParams as sampling_params

        stop_ids = sorted(int(token) for token in eos_ids)
        indices = list(range(rollouts) if rollout_indices is None else rollout_indices)
        requests = [{"prompt_token_ids": list(prompt_tokens)} for _ in indices]
        sampling = [
            sampling_params(
                temperature=0.0, max_tokens=int(max_new_tokens), detokenize=False,
                # The checkpoint generation_config lists extra stop tokens.
                # Stopping on one the protocol does not treat as EOS leaves a
                # short completion the validator rejects as bad_termination.
                # ``ignore_eos`` keeps only the protocol stop set below.
                ignore_eos=True,
                stop_token_ids=stop_ids or None,
                extra_args=forced_seed_extra_args(
                    randomness=randomness, prompt_idx=prompt_idx,
                    checkpoint_hash=checkpoint_hash, rollout_index=index,
                ),
            )
            for index in indices
        ]
        outputs = self._llm.generate(requests, sampling, use_tqdm=False)
        completions: list[list[int]] = []
        for output in outputs:
            completion = list(output.outputs[0].token_ids)
            end = first_eos_index(completion, set(stop_ids))
            if end is not None:
                completion = completion[: end + 1]
            completions.append(completion)
        return completions
