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

import itertools
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
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
            # A request resumed after preemption re-enters with the tokens it
            # already generated; its next pick is at that position, not 0.
            generated = len(added[3]) if len(added) > 3 and added[3] is not None else 0
            slot = _slot_index(index)
            self._slots.pop(slot, None)
            forced = dict(getattr(params, "extra_args", None) or {}).get(FORCED_SEED_KEY)
            if forced:
                self._slots[slot] = {
                    "randomness": str(forced["randomness"]),
                    "prompt_idx": int(forced["prompt_idx"]),
                    "checkpoint_hash": str(forced["checkpoint_hash"]),
                    "rollout_index": int(forced["rollout_index"]),
                    "offset": int(forced.get("base_offset", 0)) + generated,
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


class GenerationAborted(RuntimeError):
    """A group stopped before all its rollouts finished; ``reason`` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def concurrent_groups_for(
    kv_cache_tokens: int | None, *, rollouts: int, tokens_per_rollout: int,
    max_num_seqs: int,
) -> int:
    """How many rollout groups to keep in flight on one engine.

    Sized so their expected KV footprint fits the cache; past that the
    scheduler preempts and recomputes, spending GPU time twice. Never more
    groups than ``max_num_seqs`` can run at once.
    """
    rollouts = max(1, int(rollouts))
    by_seqs = max(1, int(max_num_seqs) // rollouts)
    if not kv_cache_tokens or tokens_per_rollout <= 0:
        return 1
    by_cache = int(kv_cache_tokens) // (rollouts * int(tokens_per_rollout))
    return max(1, min(by_seqs, by_cache))


@dataclass(eq=False)
class _Group:
    future: Future
    request_ids: list[str]
    stop_ids: frozenset[int]
    max_truncated: int | None
    tag: Any
    completions: dict[str, list[int]] = field(default_factory=dict)
    truncated: int = 0


def _fail(future: Future, exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)


class VLLMRolloutGenerator:
    """Rollout groups generated on one vLLM engine, several prompts at once.

    The engine is built once for the life of the process. Every checkpoint,
    including the first when built with ``load_format="dummy"``, arrives through
    :meth:`set_weights` from tensors the miner already holds, so compiled
    graphs, the KV cache and the scheduler stay up and nothing is re-read from
    disk.

    One background thread owns the engine (add, step, abort, weight writes),
    so groups submitted from several threads share every decode step instead
    of each running a 16-wide batch that thins out as its rollouts finish. A
    group resolves its Future with completions, or fails with
    :class:`GenerationAborted` once abandoned.
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
        self._max_num_seqs = int(max_num_seqs)
        self._sampling_params_class = sampling_params_class
        self._device_index: int | None = None
        self._cv = threading.Condition()
        self._inbox: list[Callable[[], None]] = []
        self._live: set[_Group] = set()
        self._by_request: dict[str, _Group] = {}
        self._ids = itertools.count()
        self._local = threading.local()
        self._thread: threading.Thread | None = None
        self._closed = False
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
        # The in-process worker selected its device on this thread; the engine
        # loop thread has to run on the same one.
        if torch.cuda.is_available():
            self._device_index = torch.cuda.current_device()
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

    def kv_cache_tokens(self) -> int | None:
        """Tokens the engine's KV cache holds across all running sequences."""
        config = getattr(getattr(self._llm, "llm_engine", None), "vllm_config", None)
        cache = getattr(config, "cache_config", None)
        blocks = getattr(cache, "num_gpu_blocks", None)
        block_size = getattr(cache, "block_size", None)
        if not blocks or not block_size:
            return None
        return int(blocks) * int(block_size)

    def concurrent_groups(self, *, rollouts: int, tokens_per_rollout: int) -> int:
        return concurrent_groups_for(
            self.kv_cache_tokens(), rollouts=rollouts,
            tokens_per_rollout=tokens_per_rollout, max_num_seqs=self._max_num_seqs,
        )

    @contextmanager
    def tagged(self, tag: Any) -> Iterator[None]:
        """Label groups this thread submits, so :meth:`abort` can select them."""
        previous = getattr(self._local, "tag", None)
        self._local.tag = tag
        try:
            yield
        finally:
            self._local.tag = previous

    def set_weights(self, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Write a checkpoint's tensors into the running engine in place.

        ``named_tensors`` uses Hugging Face parameter names (what
        ``model.named_parameters()`` yields on the proof copy). Raises when any
        vLLM parameter was not covered, since generating from a half-updated
        model would produce proofs the validator rejects. Groups still in
        flight were drawn from the old checkpoint and are aborted first.
        """
        started = time.monotonic()
        self._require_in_process_engine()
        weights = [(name, tensor.detach()) for name, tensor in named_tensors]
        self.abort(reason="checkpoint_changed")
        self._call_on_loop(lambda: self._apply_weights(weights))
        logger.info(
            "vLLM weights set in place: %d tensors in %.2fs",
            len(weights), time.monotonic() - started,
        )

    def _apply_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        results = self._llm.collective_rpc(_load_weights_on_worker, args=(weights,))
        missing = sorted({name for result in results or () for name in result or ()})
        if missing:
            raise RuntimeError(
                f"vLLM weight update left {len(missing)} parameters unset "
                f"(first: {missing[:3]})"
            )
        # Cached prefixes were computed with the previous weights. vLLM refuses
        # the reset while any block is held, which would leave them reusable.
        if self._llm.reset_prefix_cache() is False:
            raise RuntimeError("vLLM prefix cache still in use after abort")

    def _sampling(self) -> tuple[Any, dict[str, Any]]:
        if self._sampling_params_class is not None:
            return self._sampling_params_class, {}
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        # Only the final output is read; the default re-sends every request's
        # whole token list on each step.
        return SamplingParams, {"output_kind": RequestOutputKind.FINAL_ONLY}

    def submit(
        self, prompt_tokens: list[int], *, randomness: str, prompt_idx: int,
        checkpoint_hash: str, rollouts: int, max_new_tokens: int,
        eos_ids: list[int], rollout_indices: list[int] | None = None,
        max_truncated: int | None = None,
    ) -> Future:
        """Queue one group; the Future yields completions in ``rollout_indices`` order.

        With ``max_truncated`` set, the group is abandoned (Future fails with
        ``GenerationAborted("too_many_truncated")``) as soon as more rollouts
        than that have finished without a stop token, since admission would
        refuse it however the rest ended.
        """
        sampling_params, sampling_extra = self._sampling()
        stop_ids = sorted(int(token) for token in eos_ids)
        indices = list(range(rollouts) if rollout_indices is None else rollout_indices)
        future: Future = Future()
        if not indices:
            future.set_result([])
            return future
        batch = next(self._ids)
        request_ids = [f"g{batch}-r{index}" for index in indices]
        requests = [
            (
                request_id,
                {"prompt_token_ids": list(prompt_tokens)},
                sampling_params(
                    temperature=0.0, max_tokens=int(max_new_tokens), detokenize=False,
                    # The checkpoint generation_config lists extra stop tokens.
                    # Stopping on one the protocol does not treat as EOS leaves
                    # a short completion the validator rejects as
                    # bad_termination. ``ignore_eos`` keeps only the protocol
                    # stop set below.
                    ignore_eos=True,
                    stop_token_ids=stop_ids or None,
                    extra_args=forced_seed_extra_args(
                        randomness=randomness, prompt_idx=prompt_idx,
                        checkpoint_hash=checkpoint_hash, rollout_index=index,
                    ),
                    **sampling_extra,
                ),
            )
            for request_id, index in zip(request_ids, indices)
        ]
        group = _Group(
            future=future, request_ids=request_ids, stop_ids=frozenset(stop_ids),
            max_truncated=None if max_truncated is None else int(max_truncated),
            tag=getattr(self._local, "tag", None),
        )
        with self._cv:
            if self._closed:
                raise RuntimeError("vLLM generator is closed")
            self._live.add(group)
            for request_id in request_ids:
                self._by_request[request_id] = group
            self._inbox.append(lambda: self._add_requests(requests))
            self._cv.notify()
        self._ensure_loop()
        return future

    def generate(
        self, prompt_tokens: list[int], *, randomness: str, prompt_idx: int,
        checkpoint_hash: str, rollouts: int, max_new_tokens: int,
        eos_ids: list[int], rollout_indices: list[int] | None = None,
        max_truncated: int | None = None,
        wait: Callable[[Future], list[list[int]]] | None = None,
    ) -> list[list[int]]:
        """Completion token ids per rollout, truncated at their first stop token.

        vLLM keeps the token that stopped a request (measured 20-09), which the
        protocol needs: the terminal pick is checked on it. ``wait`` blocks on
        the group's Future (default ``Future.result``); callers pass one that
        releases their own locks meanwhile.
        """
        future = self.submit(
            prompt_tokens, randomness=randomness, prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash, rollouts=rollouts,
            max_new_tokens=max_new_tokens, eos_ids=eos_ids,
            rollout_indices=rollout_indices, max_truncated=max_truncated,
        )
        return wait(future) if wait is not None else future.result()

    def abort(self, predicate: Callable[[Any], bool] | None = None, *, reason: str) -> int:
        """Abandon in-flight groups whose tag matches ``predicate`` (all if None)."""
        with self._cv:
            doomed = [
                group for group in self._live
                if predicate is None or predicate(group.tag)
            ]
            for group in doomed:
                self._drop(group)
        for group in doomed:
            _fail(group.future, GenerationAborted(reason))
        return len(doomed)

    def in_flight(self) -> int:
        with self._cv:
            return len(self._live)

    def close(self) -> None:
        self.abort(reason="closed")
        with self._cv:
            self._closed = True
            self._cv.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=10)

    def _drop(self, group: _Group) -> None:
        """Forget ``group`` and queue its unfinished requests for abort. Holds ``_cv``."""
        self._live.discard(group)
        pending = []
        for request_id in group.request_ids:
            self._by_request.pop(request_id, None)
            if request_id not in group.completions:
                pending.append(request_id)
        if pending:
            self._inbox.append(lambda: self._llm.llm_engine.abort_request(pending))
            self._cv.notify()

    def _add_requests(self, requests: list[tuple[str, dict, Any]]) -> None:
        engine = self._llm.llm_engine
        for request_id, prompt, params in requests:
            with self._cv:
                alive = request_id in self._by_request
            if alive:
                engine.add_request(request_id, prompt, params)

    def _call_on_loop(self, fn: Callable[[], Any]) -> Any:
        future: Future = Future()

        def run() -> None:
            try:
                future.set_result(fn())
            except BaseException as exc:
                future.set_exception(exc)

        with self._cv:
            self._inbox.append(run)
            self._cv.notify()
        self._ensure_loop()
        return future.result()

    def _ensure_loop(self) -> None:
        with self._cv:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run, name="vllm-generation", daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        if self._device_index is not None:
            torch.cuda.set_device(self._device_index)
        while True:
            with self._cv:
                while not self._closed and not self._inbox and not self._live:
                    self._cv.wait()
                if self._closed:
                    return
                work, self._inbox = self._inbox, []
            try:
                for item in work:
                    item()
                # An idle loop would otherwise keep the last weight list alive.
                work = item = None
                with self._cv:
                    if not self._live:
                        continue
                engine = self._llm.llm_engine
                if not engine.has_unfinished_requests():
                    self._fail_stranded()
                    continue
                outputs = engine.step()
            except Exception as exc:
                logger.exception("vLLM engine loop failed")
                self._fail_all(exc)
                continue
            self._collect(outputs)

    def _collect(self, outputs: Iterable[Any]) -> None:
        resolved: list[tuple[_Group, Any]] = []
        with self._cv:
            for output in outputs or ():
                if not getattr(output, "finished", True):
                    continue
                group = self._by_request.get(output.request_id)
                if group is None:
                    continue
                completion = list(output.outputs[0].token_ids)
                end = first_eos_index(completion, set(group.stop_ids))
                if end is not None:
                    completion = completion[: end + 1]
                else:
                    group.truncated += 1
                group.completions[output.request_id] = completion
                if group.max_truncated is not None and group.truncated > group.max_truncated:
                    self._drop(group)
                    resolved.append((group, GenerationAborted("too_many_truncated")))
                elif len(group.completions) == len(group.request_ids):
                    self._drop(group)
                    resolved.append(
                        (group, [group.completions[rid] for rid in group.request_ids])
                    )
        for group, result in resolved:
            if isinstance(result, BaseException):
                _fail(group.future, result)
            elif not group.future.done():
                group.future.set_result(result)

    def _fail_stranded(self) -> None:
        """Groups whose requests all left the engine without every output."""
        with self._cv:
            if self._inbox:
                return
            stranded = list(self._live)
            for group in stranded:
                self._drop(group)
        for group in stranded:
            _fail(group.future, RuntimeError("vLLM finished a group without all outputs"))

    def _fail_all(self, exc: BaseException) -> None:
        with self._cv:
            groups = list(self._live)
            self._live.clear()
            self._by_request.clear()
        for group in groups:
            _fail(group.future, exc)


