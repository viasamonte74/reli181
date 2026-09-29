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


def _run_until(eng, recorder, *, submissions, timeout=10.0):
    from reliquary.miner import engine as engine_mod
    from reliquary.miner import submitter

    state = _state()

    async def _miner_state(url, *, client, etag=None):
        return state, "etag"

    async def _contract(url, *, client):
        raise submitter.SubmissionError("no telemetry")

    async def _submit(url, request, **kwargs):
        recorder.submitted += 1
        return SimpleNamespace(
            accepted=True, reason="submitted", _retry_after_seconds=None,
        )

    @contextlib.asynccontextmanager
    async def _monitor(*args, **kwargs):
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
        stack.enter_context(patch.object(engine_mod, "maybe_pull_checkpoint", _no_pull))
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
            lambda s: SimpleNamespace(checkpoint_n=1, repo_id="repo", oid=REV1),
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
