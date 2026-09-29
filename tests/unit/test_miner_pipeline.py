"""Generation/proof pipeline in ``MiningEngine.mine_window``.

Generation (vllm_gpu) and proof (proof_gpu) are separate stages joined by a
bounded queue. These tests drive the real loop with fake validator I/O and
timed fake stages, and check that the stages overlap only when they sit on
different devices.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from reliquary.miner.engine import (
    MiningEngine,
    _PreparedGroup,
    _resolve_pipeline_depth,
    _stale_group_reason,
)
from reliquary.protocol.submission import WindowState

STAGE_SECONDS = 0.15
REV1 = "1" * 40
REV2 = "2" * 40


class _Lane:
    accepting_submissions = True
    admission_remaining = 100
    prompt_range = (0, 10_000)

    def cooldown_prompts(self):
        return set()


class _Env:
    name = "openmathinstruct"
    validator_authoritative_reward = False

    def __len__(self):
        return 10_000

    def get_problem(self, idx):
        return {"prompt": f"p{idx}"}


def _state(window_n=1, randomness="ab" * 32, revision=REV1):
    return SimpleNamespace(
        state=WindowState.OPEN,
        window_n=window_n,
        randomness=randomness,
        checkpoint_n=1,
        checkpoint_repo_id="repo",
        checkpoint_revision=revision,
        protocol_version=6,
        generation_profile_id="profile",
        generation_contract={},
        submission_deadline_at=None,
        environments={"openmathinstruct": _Lane()},
    )


class _Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.generation: list[tuple[float, float]] = []
        self.proof: list[tuple[float, float]] = []
        self.proof_threads: set[int] = set()
        self.uploads: list[tuple[float, float]] = []
        self.submitted = 0


def _engine(*, vllm_gpu, proof_gpu, recorder, prove_error=None):
    eng = object.__new__(MiningEngine)
    eng.envs = {"openmathinstruct": _Env()}
    eng.mix = [("openmathinstruct", 1)]
    eng._cooldown_per_env = {"openmathinstruct": set()}
    eng.vllm_gpu = vllm_gpu
    eng.proof_gpu = proof_gpu
    eng.pipeline_depth = None
    eng.validator_url_override = "http://validator"
    eng.vllm_model = MagicMock()
    eng.hf_model = MagicMock()
    eng.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Hot"))
    eng._initial_checkpoint_identity = SimpleNamespace(
        checkpoint_n=1, repo_id="repo", oid=REV1,
    )
    eng._checkpoint_identity_store = MagicMock()
    eng._checkpoint_identity_store.load.return_value = None

    def _generate(**kwargs):
        start = time.monotonic()
        time.sleep(STAGE_SECONDS)
        with recorder.lock:
            recorder.generation.append((start, time.monotonic()))
        return [{"tokens": [1, 2], "prompt_length": 1}] * 16, [1.0] * 16

    def _prove(group, runtime_fingerprint):
        if prove_error is not None:
            raise prove_error
        start = time.monotonic()
        time.sleep(STAGE_SECONDS)
        with recorder.lock:
            recorder.proof.append((start, time.monotonic()))
            recorder.proof_threads.add(threading.get_ident())
        request = SimpleNamespace(
            window_start=group.state.window_n, prompt_idx=group.prompt_idx,
        )
        return request, None

    eng._generate_screened_group = _generate
    eng._prove_group = _prove
    return eng


def _run_until(eng, recorder, *, submissions, timeout=10.0, monitor_kwargs=None,
               requests=None, state=None, submit_seconds=0.0, retry_after=None,
               pull=None):
    from reliquary.miner import engine as engine_mod
    from reliquary.miner import submitter

    state = state or _state()

    async def _miner_state(url, *, client, etag=None):
        return state, "etag"

    async def _contract(url, *, client):
        raise submitter.SubmissionError("no telemetry")

    async def _submit(url, request, **kwargs):
        start = time.monotonic()
        await asyncio.sleep(submit_seconds)
        with recorder.lock:
            recorder.uploads.append((start, time.monotonic()))
        recorder.submitted += 1
        if requests is not None:
            requests.append(request)
        if retry_after is not None and recorder.submitted == 1:
            return SimpleNamespace(
                accepted=False, reason="rate_limited",
                _retry_after_seconds=retry_after,
            )
        return SimpleNamespace(
            accepted=True, reason="submitted", _retry_after_seconds=None,
        )

    @contextlib.asynccontextmanager
    async def _monitor(*args, **kwargs):
        if monitor_kwargs is not None:
            monitor_kwargs.update(kwargs)
        yield

    async def _no_pull(*, state, local_n, local_hash, local_repo_id, local_model, **_):
        return local_n, local_repo_id, local_hash, local_model

    async def _drive():
        task = asyncio.create_task(eng.mine_window(None))
        deadline = time.monotonic() + timeout
        try:
            while recorder.submitted < submissions:
                if task.done():
                    task.result()
                if time.monotonic() > deadline:
                    raise AssertionError("pipeline did not make progress")
                await asyncio.sleep(0.01)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(submitter, "get_miner_state_v1", _miner_state))
        stack.enter_context(patch.object(submitter, "get_runtime_contract_v1", _contract))
        stack.enter_context(patch.object(submitter, "submit_batch_v2", _submit))
        stack.enter_context(patch.object(submitter, "monitor_submission_verdicts", _monitor))
        stack.enter_context(patch.object(
            engine_mod, "maybe_pull_checkpoint", pull or _no_pull,
        ))
        stack.enter_context(patch.object(
            engine_mod, "_state_matches_active_protocol", lambda s: True,
        ))
        stack.enter_context(patch.object(
            engine_mod, "_release_state_mismatch_reason", lambda **kw: None,
        ))
        stack.enter_context(patch.object(
            engine_mod, "_attempt_journal_path", lambda hotkey: Path(tmp) / "a.json",
        ))
        stack.enter_context(patch.object(
            engine_mod, "checkpoint_identity_from_state",
            lambda s: SimpleNamespace(
                checkpoint_n=s.checkpoint_n, repo_id="repo",
                oid=s.checkpoint_revision,
            ),
        ))
        asyncio.run(_drive())


def _overlaps(first, second):
    return any(
        a_start < b_end and b_start < a_end
        for a_start, a_end in first
        for b_start, b_end in second
    )


def test_pipeline_depth_defaults_follow_device_layout():
    with patch("reliquary.constants.MINER_PIPELINE_DEPTH", None):
        assert _resolve_pipeline_depth(None, generation_gpu=0, proof_gpu=1) == 1
        assert _resolve_pipeline_depth(None, generation_gpu=0, proof_gpu=0) == 0
        assert _resolve_pipeline_depth(3, generation_gpu=0, proof_gpu=0) == 3
    with patch("reliquary.constants.MINER_PIPELINE_DEPTH", 2):
        assert _resolve_pipeline_depth(None, generation_gpu=0, proof_gpu=1) == 2


def _verdict(request, reason):
    return SimpleNamespace(
        window_n=request.window_start, prompt_idx=request.prompt_idx,
        merkle_root="r", accepted=True, selected_for_batch=False,
        selection_status="not_selected", outcome_code=reason,
        selection_reason=None, reason=None, proof_reason=None,
        reason_details=None,
    )


def test_unpaid_verdicts_take_back_a_kept_group_once():
    recorder = _Recorder()
    eng = _engine(vllm_gpu=0, proof_gpu=1, recorder=recorder)
    monitor, requests = {}, []
    _run_until(eng, recorder, submissions=3, monitor_kwargs=monitor,
               requests=requests)
    environment_yield = eng._environment_yield()
    kept = environment_yield._kept["openmathinstruct"]
    on_verdict = monitor["on_verdict"]

    on_verdict(_verdict(requests[0], "proof_not_needed_target_reached"))
    on_verdict(_verdict(requests[0], "proof_not_needed_target_reached"))
    on_verdict(_verdict(requests[1], "batch_filled"))
    on_verdict(_verdict(requests[2], "selected_fifo"))

    assert environment_yield._kept["openmathinstruct"] == kept - 2


def test_checkpoint_swap_releases_the_replaced_proof_copy():
    """The loop drops its references to the old proof copy and frees it once
    the swap lands, so the next reload can stage beside the current copy."""
    import gc
    import weakref

    from reliquary.miner import engine as engine_mod

    class _Model:
        pass

    recorder = _Recorder()
    eng = _engine(vllm_gpu=0, proof_gpu=1, recorder=recorder)
    eng.hf_model = _Model()
    old_ref = weakref.ref(eng.hf_model)
    new_model = _Model()
    releases = []

    async def _pull(*, state, local_n, local_hash, local_repo_id, local_model, **_):
        if local_hash == state.checkpoint_revision:
            return local_n, local_repo_id, local_hash, local_model
        eng.hf_model = new_model
        return state.checkpoint_n, "repo", state.checkpoint_revision, new_model

    def _release():
        releases.append(old_ref() is None)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with patch.object(engine_mod, "_release_cuda_memory", _release), \
                patch.object(engine_mod, "_log_proof_memory", lambda gpu: None):
            _run_until(eng, recorder, submissions=1, state=_state(revision=REV2),
                       pull=_pull)
    finally:
        if was_enabled:
            gc.enable()

    assert eng.hf_model is new_model
    assert releases == [True]


def _record_envs(eng):
    picked = []
    generate = eng._generate_screened_group

    def _generate(**kwargs):
        picked.append(kwargs["env_name"])
        return generate(**kwargs)

    eng._generate_screened_group = _generate
    return picked


def test_a_lane_whose_prompt_source_is_still_building_is_not_picked():
    gate = threading.Event()

    class _Building(_Env):
        name = "reliquary_logic_v2"

        def __len__(self):
            gate.wait(5)
            return 10_000

    recorder = _Recorder()
    eng = _engine(vllm_gpu=0, proof_gpu=1, recorder=recorder)
    eng.envs = {"openmathinstruct": _Env(), "reliquary_logic_v2": _Building()}
    eng.mix = [("openmathinstruct", 1), ("reliquary_logic_v2", 1)]
    eng._cooldown_per_env = {name: set() for name in eng.envs}
    state = _state()
    state.environments = {name: _Lane() for name in eng.envs}
    picked = _record_envs(eng)
    try:
        _run_until(eng, recorder, submissions=4, state=state)
    finally:
        gate.set()
    assert set(picked) == {"openmathinstruct"}


def test_a_failed_prompt_fetch_skips_the_prompt():
    class _Flaky(_Env):
        failures = 1

        def get_problem(self, idx):
            if self.failures:
                self.failures -= 1
                raise RuntimeError("row-group fetch failed")
            return super().get_problem(idx)

    recorder = _Recorder()
    eng = _engine(vllm_gpu=0, proof_gpu=1, recorder=recorder)
    eng.envs = {"openmathinstruct": _Flaky()}
    _run_until(eng, recorder, submissions=2)
    assert eng.envs["openmathinstruct"].failures == 0


def test_the_proof_gpu_does_not_wait_for_uploads():
    recorder = _Recorder()
    eng = _engine(vllm_gpu=0, proof_gpu=1, recorder=recorder)
    _run_until(eng, recorder, submissions=4, submit_seconds=1.0)
    assert _overlaps(recorder.proof, recorder.uploads)
    assert _overlaps(recorder.uploads[:1], recorder.uploads[1:])


def test_a_retry_after_holds_back_the_uploads_that_follow():
    recorder = _Recorder()
    eng = _engine(vllm_gpu=0, proof_gpu=1, recorder=recorder)
    _run_until(eng, recorder, submissions=3, retry_after=0.6)
    first_end = recorder.uploads[0][1]
    assert all(start >= first_end + 0.5 for start, _ in recorder.uploads[1:])


def test_two_devices_overlap_generation_with_proof():
    recorder = _Recorder()
    eng = _engine(vllm_gpu=0, proof_gpu=1, recorder=recorder)
    _run_until(eng, recorder, submissions=4)
    assert _overlaps(recorder.generation, recorder.proof)
    assert threading.get_ident() not in recorder.proof_threads


def test_shared_device_stays_sequential():
    recorder = _Recorder()
    eng = _engine(vllm_gpu=0, proof_gpu=0, recorder=recorder)
    _run_until(eng, recorder, submissions=3)
    assert not _overlaps(recorder.generation, recorder.proof)


def test_proof_stage_failure_stops_the_miner():
    recorder = _Recorder()
    eng = _engine(
        vllm_gpu=0, proof_gpu=1, recorder=recorder,
        prove_error=RuntimeError("proof OOM"),
    )
    try:
        _run_until(eng, recorder, submissions=1, timeout=5.0)
    except RuntimeError as exc:
        assert "proof OOM" in str(exc)
    else:
        raise AssertionError("a proof failure must terminate mine_window")


def _group(**state_overrides):
    return _PreparedGroup(
        state=_state(**state_overrides), env_name="openmathinstruct",
        env=_Env(), prompt_idx=7, problem={"prompt": "p7"},
        generations=[], rewards=None, checkpoint_hash=REV1,
    )


def test_stale_group_reason_catches_moved_state():
    group = _group()
    assert _stale_group_reason(
        group, latest_state=_state(), checkpoint_hash=REV1,
    ) is None
    assert _stale_group_reason(
        group, latest_state=_state(), checkpoint_hash=REV2,
    ) == "checkpoint_changed"
    assert _stale_group_reason(
        group, latest_state=_state(window_n=2), checkpoint_hash=REV1,
    ) == "window_changed"
    assert _stale_group_reason(
        group, latest_state=_state(randomness="cd" * 32), checkpoint_hash=REV1,
    ) == "randomness_changed"


class _LoadedModel:
    def __init__(self, path):
        self.path = path
        self.weights = [(f"{path}.w", object())]

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def named_parameters(self):
        return iter(self.weights)


def _checkpoint_engine(generator):
    eng = object.__new__(MiningEngine)
    eng.hf_model = _LoadedModel("old")
    eng.vllm_model = _LoadedModel("old")
    eng.proof_gpu, eng.vllm_gpu = 1, 0
    eng.generation_device = "cpu"
    eng.generator = generator
    return eng


def test_checkpoint_change_sets_vllm_weights_from_the_proof_copy():
    pushed = []
    generator = SimpleNamespace(
        set_weights=lambda named: pushed.append(list(named)),
        reload=MagicMock(side_effect=AssertionError("must not rebuild or re-read")),
    )
    eng = _checkpoint_engine(generator)

    with patch("reliquary.shared.modeling.load_text_generation_model", _load_model):
        eng._load_checkpoint("/snapshots/next")

    assert eng.hf_model.device == "cuda:1"
    assert pushed == [eng.hf_model.weights]
    generator.reload.assert_not_called()


def test_failed_vllm_weight_set_requires_a_restart():
    from reliquary.miner.engine import CheckpointActivationRestartRequired

    def _fail(named):
        raise RuntimeError("worker died")

    eng = _checkpoint_engine(SimpleNamespace(set_weights=_fail))

    with patch("reliquary.shared.modeling.load_text_generation_model", _load_model):
        try:
            eng._load_checkpoint("/snapshots/next")
        except CheckpointActivationRestartRequired:
            pass
        else:
            raise AssertionError("a failed in-place weight set must stop the miner")


def _load_model(path, **_kwargs):
    return _LoadedModel(path)


def test_vllm_memory_fraction_tracks_free_memory():
    from reliquary.miner.vllm_generation import gpu_memory_utilization_for

    gib = 2**30
    empty = lambda device: (47 * gib, 48 * gib)  # noqa: E731
    beside_proof = lambda device: (38 * gib, 48 * gib)  # noqa: E731
    assert gpu_memory_utilization_for(0, 2 * gib, mem_get_info=empty) == 0.92
    assert gpu_memory_utilization_for(
        0, 12 * gib, mem_get_info=beside_proof,
    ) == round(26 / 48, 3)
    assert gpu_memory_utilization_for(
        0, 40 * gib, mem_get_info=beside_proof,
    ) == 0.2


class _AbortRecorder:
    """The generator surface mine_window drives: tags and predicate aborts."""

    def __init__(self):
        self.aborts = []

    def tagged(self, tag):
        return contextlib.nullcontext()

    def abort(self, predicate=None, *, reason):
        self.aborts.append((reason, predicate))
        return 0


def _gpu_bound_engine(recorder, *, concurrency):
    from concurrent.futures import Future

    eng = _engine(vllm_gpu=0, proof_gpu=1, recorder=recorder)
    eng.generator = _AbortRecorder()
    eng.generation_concurrency = concurrency

    def _generate(**kwargs):
        # Host work, then the group sits on the GPU: the host turn is handed
        # back meanwhile, as VLLMRolloutGenerator.generate does via ``wait``.
        start = time.monotonic()
        done = Future()
        threading.Timer(STAGE_SECONDS, done.set_result, args=(None,)).start()
        eng._host_turn().wait(done)
        with recorder.lock:
            recorder.generation.append((start, time.monotonic()))
        return [{"tokens": [1, 2], "prompt_length": 1}] * 16, [1.0] * 16

    eng._generate_screened_group = _generate
    return eng


def test_several_prompt_groups_generate_at_once():
    recorder = _Recorder()
    eng = _gpu_bound_engine(recorder, concurrency=3)
    _run_until(eng, recorder, submissions=6)
    pairs = [
        (a, b) for i, a in enumerate(recorder.generation)
        for b in recorder.generation[i + 1:]
    ]
    assert any(_overlaps([a], [b]) for a, b in pairs)


def test_one_group_at_a_time_without_concurrency():
    recorder = _Recorder()
    eng = _gpu_bound_engine(recorder, concurrency=1)
    _run_until(eng, recorder, submissions=3)
    ordered = sorted(recorder.generation)
    assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:]))


def test_in_flight_groups_are_aborted_only_when_stale():
    from reliquary.miner.engine import _GenerationTag

    recorder = _Recorder()
    eng = _gpu_bound_engine(recorder, concurrency=2)
    _run_until(eng, recorder, submissions=2)

    live = _GenerationTag(
        window_n=1, randomness="ab" * 32, env_name="openmathinstruct",
        checkpoint_hash=REV1,
    )
    by_reason = {}
    for reason, predicate in eng.generator.aborts:
        by_reason.setdefault(reason, []).append(predicate)
    assert {"window_changed", "checkpoint_changed", "admission_closed"} <= set(by_reason)
    for reason, predicates in by_reason.items():
        if reason == "miner_stopped":
            continue
        assert not any(p(live) for p in predicates), reason
    stale = {
        "window_changed": live.__class__(2, live.randomness, live.env_name, REV1),
        "checkpoint_changed": live.__class__(1, live.randomness, live.env_name, REV2),
        "admission_closed": live.__class__(1, live.randomness, "reliquary_logic_v2", REV1),
    }
    for reason, tag in stale.items():
        assert by_reason[reason][-1](tag), reason
    # Shutting the loop down releases whatever was still on the engine.
    assert eng.generator.aborts[-1][0] == "miner_stopped"


class _BudgetGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, prompt_tokens, **kwargs):
        self.calls.append(kwargs)
        return [[5, 999]] * kwargs["rollouts"]


def _budget_engine(generator):
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Hot"))
    with patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=128):
        return MiningEngine(
            vllm_model=MagicMock(), hf_model=MagicMock(), tokenizer=MagicMock(),
            wallet=wallet, envs={"openmathinstruct": _Env()},
            mix=[("openmathinstruct", 1)], max_new_tokens=1_000_000,
            generator=generator,
        )


@patch("reliquary.constants.BFT_ENABLED", False)
@patch("reliquary.shared.modeling.resolve_eos_token_ids", return_value={999})
@patch("reliquary.protocol.tokens.encode_prompt", return_value=list(range(300)))
def test_completion_budget_stops_at_the_protocol_cap(_enc, _eos):
    from reliquary.constants import (
        max_new_tokens_for_environment,
        max_truncated_for_environment,
    )

    generator = _BudgetGenerator()
    eng = _budget_engine(generator)

    eng._generate_m_rollouts(
        {"prompt": "p"}, "ab" * 32, env_name="openmathinstruct",
        prompt_idx=7, checkpoint_hash=REV1,
    )
    eng._generate_m_rollouts(
        {"prompt": "p"}, "ab" * 32, env_name="openmathinstruct",
        prompt_idx=7, checkpoint_hash=REV1, rollout_indices=[4, 5],
        truncated_so_far=1,
    )

    cap = max_new_tokens_for_environment("openmathinstruct")
    allowance = max_truncated_for_environment("openmathinstruct")
    # prompt + completion reaching the cap is already a valid truncation.
    assert [c["max_new_tokens"] for c in generator.calls] == [cap - 300] * 2
    assert [c["max_truncated"] for c in generator.calls] == [
        allowance, max(0, allowance - 1),
    ]
    assert all(callable(c["wait"]) for c in generator.calls)


def test_host_turn_is_handed_over_while_waiting_on_the_gpu():
    from concurrent.futures import Future

    from reliquary.miner.engine import _HostTurn

    turn = _HostTurn()
    done = Future()
    entered = threading.Event()

    def other():
        with turn.hold():
            entered.set()
            done.set_result("tokens")

    with turn.hold():
        worker = threading.Thread(target=other)
        worker.start()
        assert not entered.wait(0.05)
        assert turn.wait(done) == "tokens"
        assert entered.is_set()
    worker.join(timeout=5)
