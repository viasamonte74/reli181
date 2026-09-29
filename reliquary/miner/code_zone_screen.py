"""Miner-local Code screen. It never changes a submitted reward.

OpenCode rewards stay validator-authoritative (the submission still carries
placeholder zeros). This only decides whether a finished group is worth the
local GRAIL forward and the upload. A group whose public cases all pass or
all fail is ``out_of_zone`` on the validator; scoring it here drops that work
before the proof step so the next prompt can start.

Scoring uses the same restricted worker the grader runs. One subprocess
covers the whole group. Any failure to score is a pass-through: the group is
submitted exactly as it would have been without the screen.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any


SCREEN_TIMEOUT_SECONDS = 45.0


def score_group(codes: list[str], cases: list[dict[str, Any]]) -> list[float] | None:
    """Pass fraction per completion, or None when the score cannot be trusted."""

    if not codes or not cases:
        return None
    payload = json.dumps({"codes": codes, "cases": cases}).encode()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "reliquary.miner.code_zone_screen"],
            input=payload,
            capture_output=True,
            timeout=SCREEN_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        parsed = json.loads(completed.stdout)
        rewards = parsed["rewards"]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if (
        not isinstance(rewards, list)
        or len(rewards) != len(codes)
        or any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in rewards)
    ):
        return None
    return [float(item) for item in rewards]


def _score_one(code: str, cases: list[dict[str, Any]]) -> float:
    """Match the grader server: a non-ok status other than bad_output scores 0."""

    from reliquary.environment.grader.server import GraderServer
    from reliquary.environment.grader.worker import evaluate_call

    passed = 0
    for case in cases:
        entry = case.get("entry") or {}
        output, status = evaluate_call(
            code,
            entry,
            list(case.get("args") or []),
            dict(case.get("kwargs") or {}),
            5.0,
        )
        if status == "bad_output":
            continue
        if status != "ok":
            return 0.0
        if GraderServer._outputs_match(
            output,
            case.get("expected"),
            str(case.get("compare") or "exact"),
        ):
            passed += 1
    return passed / len(cases)


def main() -> None:
    request = json.load(sys.stdin)
    codes = request["codes"]
    cases = request["cases"]
    rewards = [_score_one(str(code), cases) for code in codes]
    json.dump({"rewards": rewards}, sys.stdout)


if __name__ == "__main__":
    main()
