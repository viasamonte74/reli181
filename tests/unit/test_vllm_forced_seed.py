"""The vLLM path must draw exactly what the protocol draws.

vLLM addresses requests by their slot in a persistent batch and recycles a slot
the moment a request leaves it, so these tests pin the bookkeeping as much as
the arithmetic: a stale slot would silently draw another rollout's stream.
"""

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import pick, u_at, warp
from reliquary.miner.vllm_generation import (
    ForcedSeedVLLMProcessor,
    GenerationAborted,
    VLLMRolloutGenerator,
    concurrent_groups_for,
    forced_seed_extra_args,
)

RANDOMNESS = "ab" * 32
CHECKPOINT = "c" * 40
VOCAB = 512


def _params(rollout_index: int, prompt_idx: int = 7, base_offset: int = 0):
    return SimpleNamespace(extra_args=forced_seed_extra_args(
        randomness=RANDOMNESS, prompt_idx=prompt_idx, checkpoint_hash=CHECKPOINT,
        rollout_index=rollout_index, base_offset=base_offset,
    ))


def _update(added=(), removed=(), moved=()):
    return SimpleNamespace(batch_size=8, added=list(added),
                           removed=list(removed), moved=list(moved))


def _expected(logits: torch.Tensor, rollout_index: int, step: int,
              prompt_idx: int = 7) -> int:
    probs = warp(logits, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    return pick(probs, u_at(RANDOMNESS, prompt_idx, CHECKPOINT, rollout_index, step))


def test_the_processor_picks_what_the_protocol_picks():
    torch.manual_seed(0)
    logits = torch.randn(3, VOCAB)
    processor = ForcedSeedVLLMProcessor()
    processor.update_state(_update(added=[
        (row, _params(rollout_index=row * 5), [], []) for row in range(3)
    ]))

    forced = processor.apply(logits.clone())

    assert forced.shape == logits.shape
    for row in range(3):
        chosen = int(forced[row].argmax())
        assert chosen == _expected(logits[row], rollout_index=row * 5, step=0)
        assert forced[row, chosen] == 0.0
        assert torch.isinf(forced[row]).sum() == VOCAB - 1


def test_each_call_advances_that_slot_in_the_draw():
    torch.manual_seed(1)
    processor = ForcedSeedVLLMProcessor()
    processor.update_state(_update(added=[(0, _params(rollout_index=2), [], [])]))

    for step in range(3):
        logits = torch.randn(1, VOCAB)
        chosen = int(processor.apply(logits.clone())[0].argmax())
        assert chosen == _expected(logits[0], rollout_index=2, step=step)


def test_a_request_resumes_at_the_offset_it_declares():
    torch.manual_seed(2)
    logits = torch.randn(1, VOCAB)
    processor = ForcedSeedVLLMProcessor()
    processor.update_state(_update(added=[
        (0, _params(rollout_index=1, base_offset=9), [], []),
    ]))

    chosen = int(processor.apply(logits.clone())[0].argmax())

    assert chosen == _expected(logits[0], rollout_index=1, step=9)


def test_a_freed_slot_stops_drawing_and_a_new_one_takes_over():
    torch.manual_seed(3)
    logits = torch.randn(2, VOCAB)
    processor = ForcedSeedVLLMProcessor()
    processor.update_state(_update(added=[
        (0, _params(rollout_index=0), [], []),
        (1, _params(rollout_index=1), [], []),
    ]))
    processor.apply(logits.clone())

    processor.update_state(_update(removed=[0]))
    forced = processor.apply(logits.clone())
    # Slot 0 is nobody's: it must be left alone rather than drawn for.
    assert torch.equal(forced[0], torch.full((VOCAB,), float("-inf")))
    assert int(forced[1].argmax()) == _expected(logits[1], rollout_index=1, step=1)

    processor.update_state(_update(added=[(0, _params(rollout_index=4), [], [])]))
    forced = processor.apply(logits.clone())
    assert int(forced[0].argmax()) == _expected(logits[0], rollout_index=4, step=0)


def test_a_moved_request_keeps_its_place_in_the_draw():
    torch.manual_seed(4)
    logits = torch.randn(3, VOCAB)
    processor = ForcedSeedVLLMProcessor()
    processor.update_state(_update(added=[(2, _params(rollout_index=6), [], [])]))
    processor.apply(logits.clone())

    processor.update_state(_update(moved=[(2, 0, "UNIDIRECTIONAL")]))
    forced = processor.apply(logits.clone())

    assert int(forced[0].argmax()) == _expected(logits[0], rollout_index=6, step=1)
    assert torch.equal(forced[2], torch.full((VOCAB,), float("-inf")))


def test_an_untracked_batch_is_returned_untouched():
    logits = torch.randn(2, VOCAB)
    assert torch.equal(ForcedSeedVLLMProcessor().apply(logits.clone()), logits)


class _FakeParams(SimpleNamespace):
    """vLLM's SamplingParams, reduced to what the generator sets on it."""


class _FakeEngine:
    """vLLM's LLM reduced to the V1 engine calls the generator loop makes.

    ``completions[rollout_index]`` is what that rollout generates. Each step
    finishes the oldest running request; with ``gate`` set, a step waits for it,
    which holds requests in flight.
    """

    def __init__(self, completions, *, gate=None, hold_until_added=0):
        self.completions = completions
        self.gate = gate
        self.hold_until_added = hold_until_added
        self.added = []
        self.aborted = []
        self.running = []
        self.llm_engine = self
        self._lock = threading.Lock()

    def add_request(self, request_id, prompt, params):
        with self._lock:
            self.added.append((request_id, prompt, params))
            self.running.append((request_id, params))

    def abort_request(self, request_ids):
        with self._lock:
            self.aborted.extend(request_ids)
            self.running = [r for r in self.running if r[0] not in set(request_ids)]

    def has_unfinished_requests(self):
        with self._lock:
            return bool(self.running)

    def step(self):
        if self.gate is not None:
            assert self.gate.wait(timeout=5)
        with self._lock:
            if not self.running or len(self.added) < self.hold_until_added:
                return []
            request_id, params = self.running.pop(0)
        index = params.extra_args["forced_seed"]["rollout_index"]
        return [SimpleNamespace(
            request_id=request_id, finished=True,
            outputs=[SimpleNamespace(token_ids=list(self.completions[index]))],
        )]

    def requests(self):
        return [prompt for _, prompt, _ in self.added]

    def sampling(self):
        return [params for _, _, params in self.added]


def _generator(engine):
    return VLLMRolloutGenerator("model", engine=engine, sampling_params_class=_FakeParams)


def _submit(generator, prompt_idx, rollouts, **kwargs):
    return generator.submit(
        [prompt_idx, 1], randomness=RANDOMNESS, prompt_idx=prompt_idx,
        checkpoint_hash=CHECKPOINT, rollouts=rollouts, max_new_tokens=64,
        eos_ids=[99], **kwargs,
    )


def test_the_generator_truncates_at_the_first_stop_token():
    engine = _FakeEngine([[11, 12, 99, 13], [21, 22, 23]])
    generator = _generator(engine)

    completions = generator.generate(
        [1, 2, 3], randomness=RANDOMNESS, prompt_idx=7, checkpoint_hash=CHECKPOINT,
        rollouts=2, max_new_tokens=64, eos_ids=[99],
    )

    assert completions == [[11, 12, 99], [21, 22, 23]]


def test_every_rollout_asks_for_its_own_stream():
    engine = _FakeEngine([[1], [2], [3]])
    generator = _generator(engine)

    generator.generate(
        [5, 6], randomness=RANDOMNESS, prompt_idx=42, checkpoint_hash=CHECKPOINT,
        rollouts=3, max_new_tokens=8, eos_ids=[99],
    )

    sampling = engine.sampling()
    assert [r["prompt_token_ids"] for r in engine.requests()] == [[5, 6]] * 3
    forced = [p.extra_args["forced_seed"] for p in sampling]
    assert [f["rollout_index"] for f in forced] == [0, 1, 2]
    assert {f["prompt_idx"] for f in forced} == {42}
    assert all(p.temperature == 0.0 for p in sampling)
    assert all(p.max_tokens == 8 for p in sampling)
    assert all(p.ignore_eos is True for p in sampling)


def test_rollout_indices_select_the_forced_stream():
    engine = _FakeEngine({4: [1], 5: [2]})
    generator = _generator(engine)

    completions = generator.generate(
        [5, 6], randomness=RANDOMNESS, prompt_idx=42, checkpoint_hash=CHECKPOINT,
        rollouts=2, max_new_tokens=8, eos_ids=[99], rollout_indices=[4, 5],
    )

    forced = [p.extra_args["forced_seed"] for p in engine.sampling()]
    assert [f["rollout_index"] for f in forced] == [4, 5]
    assert completions == [[1], [2]]


def test_groups_from_several_prompts_share_the_engine():
    # No request finishes until both groups are inside the engine.
    engine = _FakeEngine([[10, 99], [20, 99], [30, 99]], hold_until_added=5)
    generator = _generator(engine)

    first = _submit(generator, 1, 3)
    second = _submit(generator, 2, 2)

    assert first.result(timeout=5) == [[10, 99], [20, 99], [30, 99]]
    assert second.result(timeout=5) == [[10, 99], [20, 99]]
    assert {p["prompt_token_ids"][0] for p in engine.requests()} == {1, 2}
    generator.close()


def test_a_group_with_too_many_truncated_rollouts_is_abandoned():
    engine = _FakeEngine([[1, 2], [3, 4], [5, 99], [6, 99]])
    generator = _generator(engine)

    future = _submit(generator, 3, 4, max_truncated=1)

    with pytest.raises(GenerationAborted) as caught:
        future.result(timeout=5)
    assert caught.value.reason == "too_many_truncated"
    deadline = time.monotonic() + 5
    while engine.running and time.monotonic() < deadline:
        time.sleep(0.01)
    # The two rollouts still running were not generated to the end.
    assert sorted(engine.aborted) == sorted(r for r, _, _ in engine.added[2:])
    generator.close()


def test_truncation_within_the_allowance_still_completes():
    engine = _FakeEngine([[1, 2], [5, 99], [6, 99]])
    generator = _generator(engine)

    assert _submit(generator, 3, 3, max_truncated=1).result(timeout=5) == [
        [1, 2], [5, 99], [6, 99],
    ]
    generator.close()


def test_abort_drops_only_the_tagged_groups():
    gate = threading.Event()
    engine = _FakeEngine([[1, 99], [2, 99]], gate=gate)
    generator = _generator(engine)
    with generator.tagged("stale"):
        stale = _submit(generator, 1, 2)
    with generator.tagged("live"):
        live = _submit(generator, 2, 2)

    assert generator.abort(lambda tag: tag == "stale", reason="window_changed") == 1
    gate.set()

    with pytest.raises(GenerationAborted, match="window_changed"):
        stale.result(timeout=5)
    assert live.result(timeout=5) == [[1, 99], [2, 99]]
    # Stale requests not yet added are skipped, the rest aborted; none of the
    # live group's are touched.
    assert engine.aborted and all(r.startswith("g0-") for r in engine.aborted)
    assert not any(
        r.startswith("g0-") and r not in engine.aborted for r, _, _ in engine.added[1:]
    )
    assert generator.in_flight() == 0
    generator.close()


def test_generate_waits_through_the_callers_wait():
    engine = _FakeEngine([[7, 99]])
    generator = _generator(engine)
    waited = []

    def wait(future):
        waited.append(future)
        return future.result(timeout=5)

    assert generator.generate(
        [1], randomness=RANDOMNESS, prompt_idx=1, checkpoint_hash=CHECKPOINT,
        rollouts=1, max_new_tokens=8, eos_ids=[99], wait=wait,
    ) == [[7, 99]]
    assert len(waited) == 1
    generator.close()


def test_a_preempted_request_resumes_at_the_tokens_it_already_has():
    torch.manual_seed(5)
    logits = torch.randn(1, VOCAB)
    processor = ForcedSeedVLLMProcessor()
    processor.update_state(_update(added=[
        (0, _params(rollout_index=3, base_offset=2), [1, 2], [8, 9, 10]),
    ]))

    chosen = int(processor.apply(logits.clone())[0].argmax())

    assert chosen == _expected(logits[0], rollout_index=3, step=5)


@pytest.mark.parametrize(
    ("kv_tokens", "expected"),
    [
        (251_808, 6),     # 16 rollouts x 2,560 expected tokens each
        (40_000, 1),      # less than one group's footprint still runs one
        (None, 1),        # unknown capacity: one group at a time
        (10_000_000, 16), # bounded by max_num_seqs // rollouts
    ],
)
def test_concurrent_groups_follow_kv_capacity(kv_tokens, expected):
    assert concurrent_groups_for(
        kv_tokens, rollouts=16, tokens_per_rollout=2_560, max_num_seqs=256,
    ) == expected


def test_the_generator_reads_capacity_from_the_engine_config():
    engine = _FakeEngine([])
    engine.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks=15_738, block_size=16),
    )
    generator = _generator(engine)

    assert generator.kv_cache_tokens() == 251_808
    assert generator.concurrent_groups(rollouts=16, tokens_per_rollout=2_560) == 6


class _StackedModel(torch.nn.Module):
    """A vLLM-style model: q/k/v arrive separately and land in one parameter."""

    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Parameter(torch.zeros(3, 2))
        self.norm = torch.nn.Parameter(torch.zeros(2))

    def load_weights(self, weights):
        loaded = set()
        for name, tensor in weights:
            if name.endswith(("q_proj", "k_proj", "v_proj")):
                row = ("q_proj", "k_proj", "v_proj").index(name.rsplit(".", 1)[-1])
                self.qkv.data[row].copy_(tensor)
                loaded.add("qkv")
            elif name == "norm":
                self.norm.data.copy_(tensor)
                loaded.add("norm")
        return loaded


class _InProcessEngine:
    """vLLM's LLM with the engine core in-process: RPC runs on the worker."""

    def __init__(self, model):
        worker = SimpleNamespace(model_runner=SimpleNamespace(get_model=lambda: model))
        self._worker = worker
        self.calls = []
        self.llm_engine = SimpleNamespace(engine_core=type("InprocClient", (), {})())

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls.append("collective_rpc")
        return [method(self._worker, *args, **(kwargs or {}))]

    def reset_prefix_cache(self, device=None):
        self.calls.append("reset_prefix_cache")


def _weights(scale: float):
    return [
        ("attn.q_proj", torch.full((2,), 1.0 * scale)),
        ("attn.k_proj", torch.full((2,), 2.0 * scale)),
        ("attn.v_proj", torch.full((2,), 3.0 * scale)),
        ("norm", torch.full((2,), 4.0 * scale)),
    ]


def test_set_weights_writes_into_the_live_parameters(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    model = _StackedModel()
    engine = _InProcessEngine(model)
    generator = VLLMRolloutGenerator("m", engine=engine, sampling_params_class=_FakeParams)
    storage = model.qkv.data_ptr()

    generator.set_weights(_weights(1.0))
    generator.set_weights(_weights(10.0))

    assert generator._llm is engine
    assert model.qkv.data_ptr() == storage
    assert model.qkv.tolist() == [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]
    assert model.norm.tolist() == [40.0, 40.0]
    assert engine.calls == ["collective_rpc", "reset_prefix_cache"] * 2


def test_set_weights_refuses_a_partial_update(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    engine = _InProcessEngine(_StackedModel())
    generator = VLLMRolloutGenerator("m", engine=engine, sampling_params_class=_FakeParams)

    with pytest.raises(RuntimeError, match="1 parameters unset"):
        generator.set_weights(_weights(1.0)[:3])

    assert "reset_prefix_cache" not in engine.calls


def test_set_weights_needs_the_engine_core_in_process():
    engine = _InProcessEngine(_StackedModel())
    engine.llm_engine.engine_core = type("SyncMPClient", (), {})()
    generator = VLLMRolloutGenerator("m", engine=engine, sampling_params_class=_FakeParams)

    with pytest.raises(RuntimeError, match="VLLM_ENABLE_V1_MULTIPROCESSING=0"):
        generator.set_weights(_weights(1.0))

    assert engine.calls == []


def test_the_engine_is_only_built_when_none_is_supplied(monkeypatch):
    real_import = __import__

    def _blocked(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise ImportError("vllm is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocked)
    with pytest.raises(ImportError):
        VLLMRolloutGenerator("model")
