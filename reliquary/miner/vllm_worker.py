"""A second vLLM engine, in its own process, on the proof GPU.

vLLM runs one in-process engine per process, and the proof copy sits idle on
its GPU most of the time. This module runs another :class:`VLLMRolloutGenerator`
in a spawned process pinned to that GPU and pools it with the in-process one.

Tensors do not cross the process boundary: the worker loads each checkpoint
from the snapshot directory the miner has just activated. Tags stay on the
miner side too, so ``abort(predicate)`` keeps working on arbitrary objects.
"""

from __future__ import annotations

import atexit
import itertools
import logging
import multiprocessing
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from typing import Any

from reliquary.miner.vllm_generation import GenerationAborted, concurrent_groups_for

logger = logging.getLogger(__name__)

ENGINE_LOST = "engine_lost"
_ENV_LOCK = threading.Lock()


def _fail(future: Future, exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)


def _physical_device(device: int) -> str:
    """The CUDA_VISIBLE_DEVICES entry that is ``cuda:<device>`` here."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return str(device)
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    return entries[device] if device < len(entries) else str(device)


def checkpoint_tensors(path: str, revision: str | None = None) -> Iterator[tuple[str, Any]]:
    """(name, CPU tensor) for every tensor in a checkpoint's safetensors files."""
    from pathlib import Path

    from safetensors import safe_open

    root = Path(path)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(
            path, revision=revision, allow_patterns=["*.safetensors", "*.json"],
        ))
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files under {root}")
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


def _worker_main(conn: Any, config: dict[str, Any]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["physical_device"])
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    logging.basicConfig(
        level=config.get("log_level", logging.INFO),
        format=(
            f"%(asctime)s %(levelname)s [gpu{config['physical_device']} engine] "
            "%(name)s: %(message)s"
        ),
    )
    send_lock = threading.Lock()

    def send(*message: Any) -> None:
        with send_lock:
            try:
                conn.send(message)
            except (OSError, EOFError, BrokenPipeError):
                pass

    try:
        from reliquary.miner.vllm_generation import (
            VLLMRolloutGenerator,
            gpu_memory_utilization_for,
        )

        utilization = gpu_memory_utilization_for(0, float(config["reserve_bytes"]))
        generator = VLLMRolloutGenerator(
            config["model_path"], revision=config.get("revision"),
            max_model_len=config.get("max_model_len"),
            max_num_seqs=int(config.get("max_num_seqs", 256)),
            gpu_memory_utilization=utilization,
        )
    except BaseException as exc:  # noqa: BLE001 - reported to the miner
        logging.getLogger(__name__).exception("second vLLM engine failed to start")
        send("start_failed", repr(exc))
        return
    send("ready", generator.kv_cache_tokens(), utilization)

    def report(gid: int, future: Future) -> None:
        exc = future.exception()
        if exc is None:
            send("done", gid, future.result())
        elif isinstance(exc, GenerationAborted):
            send("aborted", gid, exc.reason)
        else:
            send("failed", gid, repr(exc))

    try:
        while True:
            try:
                message = conn.recv()
            except (EOFError, OSError):
                break
            kind = message[0]
            if kind == "submit":
                _, gid, prompt_tokens, kwargs = message
                try:
                    with generator.tagged(gid):
                        future = generator.submit(prompt_tokens, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    send("failed", gid, repr(exc))
                    continue
                future.add_done_callback(lambda f, gid=gid: report(gid, f))
            elif kind == "abort":
                wanted = set(message[1])
                generator.abort(lambda tag: tag in wanted, reason=message[2])
            elif kind == "load":
                _, request_id, path, revision = message
                try:
                    generator.set_weights(checkpoint_tensors(path, revision))
                except Exception as exc:  # noqa: BLE001
                    logging.getLogger(__name__).exception(
                        "second vLLM engine could not load %s", path,
                    )
                    send("loaded", request_id, repr(exc))
                else:
                    send("loaded", request_id, None)
            elif kind == "close":
                break
    except KeyboardInterrupt:
        pass
    finally:
        generator.close()


class RemoteVLLMGenerator:
    """The generator surface the miner uses, served by an engine in a subprocess."""

    loads_checkpoint_path = True

    def __init__(
        self, process: Any, conn: Any, *, device: int, max_num_seqs: int,
    ) -> None:
        self._process = process
        self._conn = conn
        self.device = int(device)
        self._max_num_seqs = int(max_num_seqs)
        self._kv_cache_tokens: int | None = None
        self.gpu_memory_utilization: float | None = None
        self._pending: dict[int, tuple[Future, Any]] = {}
        self._loads: dict[int, Future] = {}
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._ids = itertools.count()
        self._local = threading.local()
        self._alive = False
        self._closing = False
        self._reader: threading.Thread | None = None

    @classmethod
    def spawn(
        cls, model_path: str, *, device: int, revision: str | None = None,
        max_model_len: int | None = None, max_num_seqs: int = 256,
        reserve_bytes: float = 0.0, context: Any = None,
    ) -> "RemoteVLLMGenerator":
        """Start the worker; :meth:`wait_ready` blocks until its engine is up."""
        ctx = context or multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        physical = _physical_device(int(device))
        config = {
            "physical_device": physical, "model_path": model_path,
            "revision": revision, "max_model_len": max_model_len,
            "max_num_seqs": int(max_num_seqs), "reserve_bytes": float(reserve_bytes),
            "log_level": logging.getLogger().getEffectiveLevel(),
        }
        process = ctx.Process(
            target=_worker_main, args=(child, config),
            name=f"vllm-cuda{device}", daemon=False,
        )
        # Anything the child imports before _worker_main runs must already see
        # only its own GPU.
        with _ENV_LOCK:
            previous = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = physical
            try:
                process.start()
            finally:
                if previous is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = previous
        child.close()
        remote = cls(process, parent, device=device, max_num_seqs=max_num_seqs)
        atexit.register(remote.close)
        return remote

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        message: tuple | None = None
        try:
            while message is None:
                if self._conn.poll(1.0):
                    message = self._conn.recv()
                elif not self._process.is_alive():
                    raise RuntimeError(
                        f"second vLLM engine exited with code {self._process.exitcode}"
                    )
                elif time.monotonic() > deadline:
                    raise TimeoutError(f"second vLLM engine not ready after {timeout:.0f}s")
        except EOFError as exc:
            self.close()
            raise RuntimeError("second vLLM engine exited during start") from exc
        except BaseException:
            self.close()
            raise
        if message[0] != "ready":
            self.close()
            raise RuntimeError(f"second vLLM engine failed to start: {message[1]}")
        self._kv_cache_tokens = message[1]
        self.gpu_memory_utilization = message[2]
        self._alive = True
        self._reader = threading.Thread(
            target=self._read, name=f"vllm-cuda{self.device}-reader", daemon=True,
        )
        self._reader.start()

    @property
    def alive(self) -> bool:
        return self._alive

    def kv_cache_tokens(self) -> int | None:
        return self._kv_cache_tokens

    def concurrent_groups(self, *, rollouts: int, tokens_per_rollout: int) -> int:
        return concurrent_groups_for(
            self._kv_cache_tokens, rollouts=rollouts,
            tokens_per_rollout=tokens_per_rollout, max_num_seqs=self._max_num_seqs,
        )

    @contextmanager
    def tagged(self, tag: Any) -> Iterator[None]:
        previous = getattr(self._local, "tag", None)
        self._local.tag = tag
        try:
            yield
        finally:
            self._local.tag = previous

    def in_flight(self) -> int:
        with self._lock:
            return len(self._pending)

    def _send(self, message: tuple) -> bool:
        with self._send_lock:
            try:
                self._conn.send(message)
                return True
            except (OSError, EOFError, BrokenPipeError, ValueError):
                pass
        self._lost()
        return False

    def submit(
        self, prompt_tokens: list[int], *, randomness: str, prompt_idx: int,
        checkpoint_hash: str, rollouts: int, max_new_tokens: int,
        eos_ids: list[int], rollout_indices: list[int] | None = None,
        max_truncated: int | None = None,
    ) -> Future:
        future: Future = Future()
        if not self._alive:
            future.set_exception(GenerationAborted(ENGINE_LOST))
            return future
        gid = next(self._ids)
        kwargs = {
            "randomness": randomness, "prompt_idx": int(prompt_idx),
            "checkpoint_hash": checkpoint_hash, "rollouts": int(rollouts),
            "max_new_tokens": int(max_new_tokens),
            "eos_ids": [int(token) for token in eos_ids],
            "rollout_indices": None if rollout_indices is None else list(rollout_indices),
            "max_truncated": max_truncated,
        }
        with self._lock:
            self._pending[gid] = (future, getattr(self._local, "tag", None))
        self._send(("submit", gid, list(prompt_tokens), kwargs))
        return future

    def generate(
        self, prompt_tokens: list[int], *, wait: Callable[[Future], Any] | None = None,
        **kwargs: Any,
    ) -> list[list[int]]:
        future = self.submit(prompt_tokens, **kwargs)
        return wait(future) if wait is not None else future.result()

    def abort(self, predicate: Callable[[Any], bool] | None = None, *, reason: str) -> int:
        with self._lock:
            doomed = [
                (gid, future) for gid, (future, tag) in self._pending.items()
                if predicate is None or predicate(tag)
            ]
            for gid, _ in doomed:
                del self._pending[gid]
        for _, future in doomed:
            _fail(future, GenerationAborted(reason))
        if doomed and self._alive:
            self._send(("abort", [gid for gid, _ in doomed], reason))
        return len(doomed)

    def set_weights(
        self, named_tensors: Iterable[Any] | None = None, *,
        checkpoint_path: str | None = None, revision: str | None = None,
        timeout: float = 900.0,
    ) -> None:
        """Load ``checkpoint_path`` in the worker; groups in flight are aborted."""
        if not checkpoint_path:
            raise RuntimeError("the second vLLM engine loads from a checkpoint path")
        if not self._alive:
            raise RuntimeError("second vLLM engine is not running")
        started = time.monotonic()
        self.abort(reason="checkpoint_changed")
        request_id = next(self._ids)
        done: Future = Future()
        with self._lock:
            self._loads[request_id] = done
        if not self._send(("load", request_id, str(checkpoint_path), revision)):
            raise RuntimeError("second vLLM engine is not running")
        error = done.result(timeout=timeout)
        if error:
            raise RuntimeError(f"second vLLM engine could not load weights: {error}")
        logger.info(
            "second vLLM engine (cuda:%d) loaded %s in %.2fs",
            self.device, checkpoint_path, time.monotonic() - started,
        )

    def _read(self) -> None:
        while True:
            try:
                message = self._conn.recv()
            except (EOFError, OSError):
                break
            kind = message[0]
            if kind == "loaded":
                with self._lock:
                    done = self._loads.pop(message[1], None)
                if done is not None and not done.done():
                    done.set_result(message[2])
                continue
            with self._lock:
                entry = self._pending.pop(message[1], None)
            if entry is None:
                continue
            future = entry[0]
            if kind == "done":
                if not future.done():
                    future.set_result(message[2])
            elif kind == "aborted":
                _fail(future, GenerationAborted(message[2]))
            else:
                logger.warning("second vLLM engine failed a group: %s", message[2])
                _fail(future, GenerationAborted("engine_failed"))
        self._lost()

    def _lost(self) -> None:
        with self._lock:
            was_alive, self._alive = self._alive, False
            pending = [future for future, _ in self._pending.values()]
            loads = list(self._loads.values())
            self._pending.clear()
            self._loads.clear()
        if was_alive and not self._closing:
            logger.error(
                "second vLLM engine (cuda:%d) stopped; generating on the rest",
                self.device,
            )
        for future in pending:
            _fail(future, GenerationAborted(ENGINE_LOST))
        for done in loads:
            if not done.done():
                done.set_result(ENGINE_LOST)

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.abort(reason="closed")
        if self._alive:
            self._send(("close",))
        self._lost()
        process = self._process
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        if process.is_alive():
            process.kill()
        try:
            self._conn.close()
        except OSError:
            pass


class PooledGenerator:
    """Several generators behind one; each group goes to the least loaded."""

    loads_checkpoint_path = True

    def __init__(self, members: list[tuple[Any, int]]) -> None:
        if not members:
            raise ValueError("PooledGenerator needs at least one generator")
        self._members = [(generator, max(1, int(capacity))) for generator, capacity in members]
        self._local = threading.local()
        self._lock = threading.Lock()

    @property
    def members(self) -> list[Any]:
        return [generator for generator, _ in self._members]

    def _pick(self) -> Any | None:
        live = [
            (generator, capacity) for generator, capacity in self._members
            if getattr(generator, "alive", True)
        ]
        if not live:
            return None
        return min(live, key=lambda member: member[0].in_flight() / member[1])[0]

    @contextmanager
    def tagged(self, tag: Any) -> Iterator[None]:
        previous = getattr(self._local, "tag", None)
        self._local.tag = tag
        try:
            yield
        finally:
            self._local.tag = previous

    def submit(self, prompt_tokens: list[int], **kwargs: Any) -> Future:
        tag = getattr(self._local, "tag", None)
        with self._lock:
            generator = self._pick()
            if generator is None:
                future: Future = Future()
                future.set_exception(GenerationAborted(ENGINE_LOST))
                return future
            with generator.tagged(tag):
                return generator.submit(prompt_tokens, **kwargs)

    def generate(
        self, prompt_tokens: list[int], *, wait: Callable[[Future], Any] | None = None,
        **kwargs: Any,
    ) -> list[list[int]]:
        future = self.submit(prompt_tokens, **kwargs)
        return wait(future) if wait is not None else future.result()

    def abort(self, predicate: Callable[[Any], bool] | None = None, *, reason: str) -> int:
        return sum(
            generator.abort(predicate, reason=reason) for generator in self.members
        )

    def in_flight(self) -> int:
        return sum(generator.in_flight() for generator in self.members)

    def kv_cache_tokens(self) -> int | None:
        sizes = [generator.kv_cache_tokens() for generator in self.members]
        known = [int(size) for size in sizes if size]
        return sum(known) if known else None

    def set_weights(
        self, named_tensors: Iterable[Any], *, checkpoint_path: str | None = None,
    ) -> None:
        """In-process engines take the tensors; engines elsewhere read the path.

        A remote engine that cannot follow is closed rather than left
        generating from the previous checkpoint.
        """
        self.abort(reason="checkpoint_changed")
        remote = [
            generator for generator in self.members
            if getattr(generator, "loads_checkpoint_path", False)
            and getattr(generator, "alive", True)
        ]
        loads: list[tuple[Any, Future]] = []
        for generator in remote:
            done: Future = Future()

            def load(generator: Any = generator, done: Future = done) -> None:
                try:
                    generator.set_weights(checkpoint_path=checkpoint_path)
                except BaseException as exc:  # noqa: BLE001
                    done.set_exception(exc)
                else:
                    done.set_result(None)

            threading.Thread(target=load, name="vllm-remote-load", daemon=True).start()
            loads.append((generator, done))
        tensors = list(named_tensors)
        for generator in self.members:
            if generator not in remote and not getattr(generator, "loads_checkpoint_path", False):
                generator.set_weights(tensors)
        for generator, done in loads:
            try:
                done.result()
            except Exception:
                logger.exception(
                    "second vLLM engine could not follow the checkpoint; closing it",
                )
                generator.close()

    def close(self) -> None:
        for generator in self.members:
            close = getattr(generator, "close", None)
            if close is not None:
                close()
