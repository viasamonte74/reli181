"""Miner checkpoint activation binds number, repository, and immutable OID."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from reliquary.miner.engine import (
    CheckpointIdentityError,
    _checkpoint_prefetch,
    _CheckpointPrefetcher,
    _prune_checkpoint_cache,
    maybe_pull_checkpoint,
)


REPO = "aivolutionedge/reliquary-sn"
OTHER_REPO = "aivolutionedge/reliquary-sn-mirror"
REV_OLD = "4" * 40
REV_5 = "5" * 40
REV_7 = "7" * 40
REV_NEW = "a" * 40


@pytest.mark.asyncio
async def test_pull_when_remote_n_higher():
    state = MagicMock(
        checkpoint_n=5,
        checkpoint_repo_id=REPO,
        checkpoint_revision=REV_5,
    )
    download_fn = AsyncMock(return_value="/hf_cache/model_5")
    load_fn = MagicMock(return_value="loaded_model_5")

    result = await maybe_pull_checkpoint(
        state=state,
        local_n=4,
        local_repo_id=REPO,
        local_hash=REV_OLD,
        local_model="old_model",
        download_fn=download_fn,
        load_fn=load_fn,
    )

    assert result == (5, REPO, REV_5, "loaded_model_5")
    download_fn.assert_awaited_once_with(REPO, REV_5)
    load_fn.assert_called_once_with("/hf_cache/model_5")


@pytest.mark.asyncio
async def test_no_pull_when_local_identity_is_exact():
    state = MagicMock(
        checkpoint_n=5,
        checkpoint_repo_id=REPO,
        checkpoint_revision=REV_5,
    )
    download_fn = AsyncMock()

    result = await maybe_pull_checkpoint(
        state=state,
        local_n=5,
        local_repo_id=REPO,
        local_hash=REV_5,
        local_model="cached",
        download_fn=download_fn,
        load_fn=MagicMock(),
    )

    assert result == (5, REPO, REV_5, "cached")
    download_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_pull_before_first_publish():
    state = MagicMock(
        checkpoint_n=0,
        checkpoint_repo_id=None,
        checkpoint_revision=None,
    )
    download_fn = AsyncMock()

    result = await maybe_pull_checkpoint(
        state=state,
        local_n=0,
        local_repo_id="",
        local_hash="",
        local_model="initial_model",
        download_fn=download_fn,
        load_fn=MagicMock(),
    )

    assert result == (0, "", "", "initial_model")
    download_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_checkpoint_identity_is_rejected():
    state = MagicMock(
        checkpoint_n=3,
        checkpoint_repo_id=REPO,
        checkpoint_revision=None,
    )
    download_fn = AsyncMock()

    with pytest.raises(CheckpointIdentityError, match="invalid"):
        await maybe_pull_checkpoint(
            state=state,
            local_n=2,
            local_repo_id=REPO,
            local_hash=REV_OLD,
            local_model="local",
            download_fn=download_fn,
            load_fn=MagicMock(),
        )

    download_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_fresh_miner_joins_a_later_checkpoint():
    state = MagicMock(
        checkpoint_n=7,
        checkpoint_repo_id=REPO,
        checkpoint_revision=REV_7,
    )

    result = await maybe_pull_checkpoint(
        state=state,
        local_n=0,
        local_repo_id="",
        local_hash="",
        local_model=None,
        download_fn=AsyncMock(return_value="/hf_cache/model_7"),
        load_fn=MagicMock(return_value="model_7"),
    )

    assert result == (7, REPO, REV_7, "model_7")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("remote_repo", "remote_revision"),
    [(REPO, REV_NEW), (OTHER_REPO, REV_5)],
)
async def test_same_number_identity_rebinding_fails_closed(
    remote_repo,
    remote_revision,
):
    state = MagicMock(
        checkpoint_n=5,
        checkpoint_repo_id=remote_repo,
        checkpoint_revision=remote_revision,
    )
    download_fn = AsyncMock()

    with pytest.raises(CheckpointIdentityError, match="rebound"):
        await maybe_pull_checkpoint(
            state=state,
            local_n=5,
            local_repo_id=REPO,
            local_hash=REV_5,
            local_model="old_model",
            download_fn=download_fn,
            load_fn=MagicMock(),
        )

    download_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_number_can_initialize_an_unknown_local_identity():
    state = MagicMock(
        checkpoint_n=5,
        checkpoint_repo_id=REPO,
        checkpoint_revision=REV_5,
    )

    result = await maybe_pull_checkpoint(
        state=state,
        local_n=5,
        local_repo_id="",
        local_hash="",
        local_model="initial_model",
        download_fn=AsyncMock(return_value="/hf_cache/model_5"),
        load_fn=MagicMock(return_value="loaded_model_5"),
    )

    assert result == (5, REPO, REV_5, "loaded_model_5")


@pytest.mark.asyncio
async def test_mutable_revision_is_rejected_before_download():
    state = MagicMock(
        checkpoint_n=5,
        checkpoint_repo_id=REPO,
        checkpoint_revision="main",
    )
    download_fn = AsyncMock()

    with pytest.raises(CheckpointIdentityError, match="invalid"):
        await maybe_pull_checkpoint(
            state=state,
            local_n=4,
            local_repo_id=REPO,
            local_hash=REV_OLD,
            local_model="old_model",
            download_fn=download_fn,
            load_fn=MagicMock(),
        )

    download_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_loader_failure_does_not_advance_checkpoint_identity():
    state = MagicMock(
        checkpoint_n=5,
        checkpoint_repo_id=REPO,
        checkpoint_revision=REV_NEW,
    )

    with pytest.raises(RuntimeError, match="no activated model"):
        await maybe_pull_checkpoint(
            state=state,
            local_n=4,
            local_repo_id=REPO,
            local_hash=REV_OLD,
            local_model="old_model",
            download_fn=AsyncMock(return_value="/hf_cache/new"),
            load_fn=MagicMock(return_value=None),
        )


class _SlowDownloads:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.release = asyncio.Event()
        self.fail = False

    async def __call__(self, repo_id, revision):
        self.calls.append((repo_id, revision))
        await self.release.wait()
        if self.fail:
            raise OSError("connection reset")
        return f"/hf_cache/{revision[:4]}"


def _prefetcher(downloads, head=REV_NEW):
    return _CheckpointPrefetcher(
        interval=10, download_fn=downloads, head_fn=lambda repo_id: head,
    )


@pytest.mark.asyncio
async def test_pull_joins_a_prefetch_already_in_flight():
    downloads = _SlowDownloads()
    prefetcher = _prefetcher(downloads)
    prefetcher.watch(REPO, REV_OLD)

    await prefetcher.poll()
    pull = asyncio.ensure_future(prefetcher.download(REPO, REV_NEW))
    await asyncio.sleep(0)
    downloads.release.set()

    assert await pull == "/hf_cache/aaaa"
    assert downloads.calls == [(REPO, REV_NEW)]


@pytest.mark.asyncio
async def test_prefetch_skips_active_and_known_heads():
    downloads = _SlowDownloads()
    downloads.release.set()
    prefetcher = _prefetcher(downloads)

    await prefetcher.poll()
    prefetcher.watch(REPO, REV_NEW)
    await prefetcher.poll()
    assert downloads.calls == []

    prefetcher.watch(REPO, REV_OLD)
    await prefetcher.poll()
    await prefetcher.poll()
    await asyncio.sleep(0)
    assert downloads.calls == [(REPO, REV_NEW)]


@pytest.mark.asyncio
async def test_failed_prefetch_is_retried_by_the_pull():
    downloads = _SlowDownloads()
    downloads.fail = True
    prefetcher = _prefetcher(downloads)
    prefetcher.watch(REPO, REV_OLD)

    await prefetcher.poll()
    downloads.release.set()
    for _ in range(3):
        await asyncio.sleep(0)
    downloads.fail = False

    assert await prefetcher.download(REPO, REV_NEW) == "/hf_cache/aaaa"
    assert downloads.calls == [(REPO, REV_NEW), (REPO, REV_NEW)]


@pytest.mark.asyncio
async def test_cancelled_pull_keeps_the_download_running():
    downloads = _SlowDownloads()
    prefetcher = _prefetcher(downloads)

    pull = asyncio.ensure_future(prefetcher.download(REPO, REV_NEW))
    await asyncio.sleep(0)
    pull.cancel()
    await asyncio.gather(pull, return_exceptions=True)
    downloads.release.set()

    assert await prefetcher.download(REPO, REV_NEW) == "/hf_cache/aaaa"
    assert downloads.calls == [(REPO, REV_NEW)]


@pytest.mark.asyncio
async def test_head_lookup_failure_is_quiet():
    def head_fn(repo_id):
        raise OSError("dns")

    downloads = _SlowDownloads()
    prefetcher = _CheckpointPrefetcher(
        interval=10, download_fn=downloads, head_fn=head_fn,
    )
    prefetcher.watch(REPO, REV_OLD)
    await prefetcher.poll()
    assert downloads.calls == []


def _cache_revision(cache, repo_id, commit, files, mtime):
    """Lay out one revision in the Hub cache format, blobs stamped ``mtime``."""
    root = cache / ("models--" + repo_id.replace("/", "--"))
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    snapshot = root / "snapshots" / commit
    snapshot.mkdir(parents=True)
    for name, blob in files.items():
        blob_path = root / "blobs" / blob
        if not blob_path.exists():
            blob_path.write_bytes(blob.encode() * 100)
            os.utime(blob_path, (mtime, mtime))
        (snapshot / name).symlink_to(f"../../blobs/{blob}")
    return root


def test_prune_deletes_only_revisions_older_than_active(tmp_path):
    old, pinned, active, newer = "1" * 40, "2" * 40, "3" * 40, "4" * 40
    root = _cache_revision(tmp_path, REPO, old, {"w": "b_old", "c": "b_cfg"}, 100)
    _cache_revision(tmp_path, REPO, pinned, {"w": "b_pin"}, 200)
    _cache_revision(tmp_path, REPO, active, {"w": "b_act", "c": "b_cfg"}, 300)
    _cache_revision(tmp_path, REPO, newer, {"w": "b_new"}, 400)
    other = _cache_revision(tmp_path, OTHER_REPO, old, {"w": "b_other"}, 50)

    deleted, freed = _prune_checkpoint_cache(
        REPO, active, {pinned}, cache_dir=tmp_path,
    )

    assert deleted == 1 and freed > 0
    assert sorted(p.name for p in (root / "snapshots").iterdir()) == [
        pinned, active, newer,
    ]
    assert not (root / "blobs" / "b_old").exists()
    assert (root / "blobs" / "b_cfg").exists()
    assert (other / "snapshots" / old).exists()


def test_prune_does_nothing_when_active_is_not_cached(tmp_path):
    root = _cache_revision(tmp_path, REPO, "1" * 40, {"w": "b1"}, 100)

    assert _prune_checkpoint_cache(REPO, "9" * 40, set(), cache_dir=tmp_path) == (0, 0)
    assert (root / "snapshots" / ("1" * 40)).exists()


@pytest.mark.asyncio
async def test_pinned_revisions_are_in_flight_downloads_and_latest_head():
    downloads = _SlowDownloads()
    prefetcher = _prefetcher(downloads)
    prefetcher.watch(REPO, REV_OLD)

    await prefetcher.poll()
    assert prefetcher.pinned(REPO) == {REV_NEW}
    assert prefetcher.pinned(OTHER_REPO) == set()

    downloads.release.set()
    await prefetcher.download(REPO, REV_NEW)
    await prefetcher.download(REPO, REV_5)
    assert prefetcher.pinned(REPO) == {REV_NEW}


@pytest.mark.asyncio
async def test_prefetch_loop_is_off_when_interval_is_zero():
    prefetcher = _CheckpointPrefetcher(
        interval=0, download_fn=_SlowDownloads(),
        head_fn=MagicMock(side_effect=AssertionError("polled")),
    )
    prefetcher.watch(REPO, REV_OLD)
    async with _checkpoint_prefetch(prefetcher):
        await asyncio.sleep(0.01)
