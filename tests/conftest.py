"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_reliquary_state_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep durable-state tests isolated from host/container defaults."""

    monkeypatch.setenv("RELIQUARY_STATE_DIR", str(tmp_path / "state"))


@pytest.fixture(autouse=True)
def no_checkpoint_cache_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep miner loops under test off the Hub and out of the local HF cache."""

    monkeypatch.setattr(
        "reliquary.constants.MINER_CHECKPOINT_PREFETCH_SECONDS", 0.0,
    )
    monkeypatch.setattr("reliquary.constants.MINER_PRUNE_CHECKPOINTS", False)
    monkeypatch.setattr("reliquary.constants.MINER_LIVE_LANE_PRICES", False)
    monkeypatch.setattr("reliquary.constants.MINER_LANE_VALUES", "")
