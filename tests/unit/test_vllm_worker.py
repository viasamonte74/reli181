import multiprocessing
import threading
from concurrent.futures import Future

import pytest

from reliquary.miner import vllm_generation, vllm_worker
from reliquary.miner.vllm_generation import GenerationAborted
from reliquary.miner.vllm_worker import (
    PooledGenerator,
    RemoteVLLMGenerator,
    _physical_device,
    _worker_main,
    checkpoint_tensors,
)

KWARGS = dict(
    randomness="ab", prompt_idx=3, checkpoint_hash="h", rollouts=2,
    max_new_tokens=8, eos_ids=[1],
)


class _Local:
    """In-process generator double: groups stay pending until resolved."""

    def __init__(self):
        self.tag = None
        self.groups: list[tuple[Future, object, list]] = []
        self.weights = None

    def tagged(self, tag):
        from contextlib import contextmanager

        @contextmanager
        def scope():
            previous, self.tag = self.tag, tag
            try:
                yield
            finally:
                self.tag = previous

        return scope()

    def submit(self, prompt_tokens, **kwargs):
        future = Future()
        self.groups.append((future, self.tag, prompt_tokens))
        return future

    def in_flight(self):
        return sum(1 for future, _, _ in self.groups if not future.done())

    def abort(self, predicate=None, *, reason):
        count = 0
        for future, tag, _ in self.groups:
            if not future.done() and (predicate is None or predicate(tag)):
                future.set_exception(GenerationAborted(reason))
                count += 1
        return count

    def kv_cache_tokens(self):
        return 100

    def set_weights(self, named_tensors):
        self.weights = list(named_tensors)


class _Remote(_Local):
    loads_checkpoint_path = True

    def __init__(self, *, fail_load=False):
        super().__init__()
        self.alive = True
        self.fail_load = fail_load
        self.loaded = None
        self.closed = False

    def set_weights(self, named_tensors=None, *, checkpoint_path=None):
        if self.fail_load:
            raise RuntimeError("no such file")
        self.loaded = checkpoint_path

    def close(self):
        self.closed = True
        self.alive = False


def test_the_pool_sends_each_group_to_the_least_loaded_engine():
    local, remote = _Local(), _Remote()
    pool = PooledGenerator([(local, 4), (remote, 2)])
    for _ in range(6):
        pool.submit([5], **KWARGS)
    assert (local.in_flight(), remote.in_flight()) == (4, 2)


def test_the_pool_forwards_the_tag_and_aborts_across_engines():
    local, remote = _Local(), _Remote()
    pool = PooledGenerator([(local, 1), (remote, 1)])
    with pool.tagged("w1"):
        first = pool.submit([5], **KWARGS)
    with pool.tagged("w2"):
        second = pool.submit([5], **KWARGS)
    assert {local.groups[0][1], remote.groups[0][1]} == {"w1", "w2"}

    assert pool.abort(lambda tag: tag == "w1", reason="window_closed") == 1
    assert pool.abort(reason="stop") == 1
    for future in (first, second):
        with pytest.raises(GenerationAborted):
            future.result(timeout=1)


def test_the_pool_skips_an_engine_that_stopped():
    local, remote = _Local(), _Remote()
    remote.alive = False
    pool = PooledGenerator([(local, 1), (remote, 8)])
    pool.submit([5], **KWARGS)
    assert local.in_flight() == 1 and remote.in_flight() == 0

    local_only = PooledGenerator([(remote, 1)])
    with pytest.raises(GenerationAborted) as info:
        local_only.submit([5], **KWARGS).result(timeout=1)
    assert info.value.reason == "engine_lost"


def test_pool_weights_go_as_tensors_locally_and_as_a_path_remotely():
    local, remote = _Local(), _Remote()
    pool = PooledGenerator([(local, 1), (remote, 1)])
    pending = pool.submit([5], **KWARGS)

    pool.set_weights(iter([("w", 1)]), checkpoint_path="/snap/abc")

    assert local.weights == [("w", 1)]
    assert remote.loaded == "/snap/abc"
    with pytest.raises(GenerationAborted):
        pending.result(timeout=1)


def test_a_remote_engine_that_cannot_follow_a_checkpoint_is_closed():
    local, remote = _Local(), _Remote(fail_load=True)
    pool = PooledGenerator([(local, 1), (remote, 1)])

    pool.set_weights([("w", 1)], checkpoint_path="/snap/abc")

    assert remote.closed and not remote.alive
    assert local.weights == [("w", 1)]


def test_the_engine_passes_the_checkpoint_path_to_a_pool():
    from reliquary.miner.engine import MiningEngine

    class _Model:
        def named_parameters(self):
            return iter([("w", 1)])

    local, remote = _Local(), _Remote()
    engine = MiningEngine.__new__(MiningEngine)
    engine.generator = PooledGenerator([(local, 1), (remote, 1)])
    engine.hf_model = _Model()

    engine._set_generator_weights("/snap/new")

    assert remote.loaded == "/snap/new" and local.weights == [("w", 1)]


class _FakeProcess:
    exitcode = None

    def __init__(self):
        self.joined = False

    def is_alive(self):
        return not self.joined

    def join(self, timeout=None):
        self.joined = True

    def terminate(self):
        self.joined = True

    kill = terminate


def _remote_with_worker(handler):
    """A RemoteVLLMGenerator whose child end is served by ``handler`` on a thread."""
    parent, child = multiprocessing.Pipe()
    remote = RemoteVLLMGenerator(_FakeProcess(), parent, device=1, max_num_seqs=64)

    def serve():
        child.send(("ready", 4096, 0.5))
        try:
            while True:
                message = child.recv()
                if message[0] == "close":
                    break
                handler(child, message)
        except (EOFError, OSError):
            pass
        child.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    remote.wait_ready(timeout=5)
    return remote, child, thread


def test_a_remote_group_resolves_with_the_worker_completions():
    seen = []

    def handler(child, message):
        seen.append(message)
        if message[0] == "submit":
            child.send(("done", message[1], [[7, 1], [8, 1]]))

    remote, _, _ = _remote_with_worker(handler)
    try:
        assert remote.kv_cache_tokens() == 4096
        assert remote.concurrent_groups(rollouts=16, tokens_per_rollout=128) == 2
        with remote.tagged("tag"):
            completions = remote.generate([5, 6], **KWARGS)
        assert completions == [[7, 1], [8, 1]]
        assert seen[0][2] == [5, 6] and seen[0][3]["prompt_idx"] == 3
        assert remote.in_flight() == 0
    finally:
        remote.close()


def test_a_remote_abort_selects_by_tag_on_the_miner_side():
    messages = []
    remote, _, _ = _remote_with_worker(lambda child, message: messages.append(message))
    try:
        with remote.tagged(("w", 1)):
            doomed = remote.submit([5], **KWARGS)
        with remote.tagged(("w", 2)):
            kept = remote.submit([5], **KWARGS)

        assert remote.abort(lambda tag: tag == ("w", 1), reason="window_closed") == 1

        with pytest.raises(GenerationAborted) as info:
            doomed.result(timeout=1)
        assert info.value.reason == "window_closed"
        assert not kept.done()
    finally:
        remote.close()
    with pytest.raises(GenerationAborted):
        kept.result(timeout=1)


def test_a_worker_that_dies_fails_its_groups_as_engine_lost():
    def handler(child, message):
        child.close()
        raise EOFError

    remote, _, thread = _remote_with_worker(handler)
    future = remote.submit([5], **KWARGS)
    with pytest.raises(GenerationAborted) as info:
        future.result(timeout=5)
    assert info.value.reason == "engine_lost"
    assert not remote.alive
    with pytest.raises(GenerationAborted):
        remote.submit([5], **KWARGS).result(timeout=1)
    remote.close()


def test_a_remote_load_waits_for_the_worker_and_reports_errors():
    def handler(child, message):
        if message[0] == "load":
            error = None if message[2] == "/good" else "FileNotFoundError"
            child.send(("loaded", message[1], error))

    remote, _, _ = _remote_with_worker(handler)
    try:
        remote.set_weights(checkpoint_path="/good")
        with pytest.raises(RuntimeError, match="FileNotFoundError"):
            remote.set_weights(checkpoint_path="/bad")
        with pytest.raises(RuntimeError, match="checkpoint path"):
            remote.set_weights([("w", 1)])
    finally:
        remote.close()


class _WorkerEngine(_Local):
    instances: list = []

    def __init__(self, model_path, **settings):
        super().__init__()
        self.model_path = model_path
        self.settings = settings
        self.closed = False
        _WorkerEngine.instances.append(self)

    def submit(self, prompt_tokens, **kwargs):
        future = super().submit(prompt_tokens, **kwargs)
        if prompt_tokens == [9]:
            future.set_exception(GenerationAborted("too_many_truncated"))
        else:
            future.set_result([[len(prompt_tokens)]])
        return future

    def set_weights(self, named_tensors):
        self.weights = list(named_tensors)

    def kv_cache_tokens(self):
        return 2048

    def close(self):
        self.closed = True


def test_the_worker_loop_serves_submits_aborts_and_loads(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(vllm_generation, "VLLMRolloutGenerator", _WorkerEngine)
    monkeypatch.setattr(
        vllm_generation, "gpu_memory_utilization_for", lambda device, reserve: 0.61,
    )
    monkeypatch.setattr(
        vllm_worker, "checkpoint_tensors", lambda path, revision: iter([("w", path)]),
    )
    parent, child = multiprocessing.Pipe()
    config = {
        "physical_device": "1", "model_path": "m", "revision": None,
        "max_model_len": 100, "max_num_seqs": 32, "reserve_bytes": 0.0,
    }
    thread = threading.Thread(target=_worker_main, args=(child, config), daemon=True)
    thread.start()

    assert parent.recv() == ("ready", 2048, 0.61)
    parent.send(("submit", 0, [1, 2, 3], KWARGS))
    assert parent.recv() == ("done", 0, [[3]])
    parent.send(("submit", 1, [9], KWARGS))
    assert parent.recv() == ("aborted", 1, "too_many_truncated")
    parent.send(("load", 5, "/snap", None))
    assert parent.recv() == ("loaded", 5, None)
    parent.send(("close",))
    thread.join(timeout=5)

    engine = _WorkerEngine.instances[-1]
    assert engine.settings["gpu_memory_utilization"] == 0.61
    assert engine.groups[0][1] == 0  # the worker tags each group with its id
    assert engine.weights == [("w", "/snap")]
    assert engine.closed


def test_the_worker_reports_a_failed_start(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    def boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(vllm_generation, "VLLMRolloutGenerator", boom)
    monkeypatch.setattr(
        vllm_generation, "gpu_memory_utilization_for", lambda device, reserve: 0.5,
    )
    parent, child = multiprocessing.Pipe()
    _worker_main(child, {"physical_device": "1", "model_path": "m", "reserve_bytes": 0})
    kind, detail = parent.recv()
    assert kind == "start_failed" and "out of memory" in detail


def test_physical_device_follows_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _physical_device(1) == "1"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    assert _physical_device(1) == "6"


def test_checkpoint_tensors_reads_every_safetensors_file(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    save_file({"a": torch.ones(2)}, str(tmp_path / "model-00001.safetensors"))
    save_file({"b": torch.zeros(3)}, str(tmp_path / "model-00002.safetensors"))

    names = {name: tensor.shape[0] for name, tensor in checkpoint_tensors(str(tmp_path))}

    assert names == {"a": 2, "b": 3}
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        list(checkpoint_tensors(str(empty)))
