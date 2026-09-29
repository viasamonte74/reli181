"""Code groups that cannot clear the zone are dropped before the GRAIL proof."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from reliquary.constants import M_ROLLOUTS
from reliquary.miner.code_zone_screen import score_group
from reliquary.miner.engine import (
    MiningEngine,
    _EnvironmentYield,
    _eligible_generation_mix,
    _proof_termination_kind,
    _structural_termination_kind,
    _termination_upload_skip_reason,
    _unanimous_screen_stops,
)


def test_open_lanes_are_weighted_by_seats_left():
    state = SimpleNamespace(environments={
        "openmathinstruct": SimpleNamespace(
            accepting_submissions=True, admission_remaining=2,
        ),
        "opencodeinstruct": SimpleNamespace(
            accepting_submissions=True, admission_remaining=40,
        ),
        "reliquary_logic_v2": SimpleNamespace(
            accepting_submissions=False, admission_remaining=0,
        ),
    })
    assert _eligible_generation_mix(
        [
            ("openmathinstruct", 1),
            ("opencodeinstruct", 1),
            ("reliquary_logic_v2", 1),
        ],
        state,
    ) == [("openmathinstruct", 2), ("opencodeinstruct", 40)]


def test_score_group_uses_public_cases():
    cases = [{
        "entry": {"kind": "function", "name": "add"},
        "args": [1, 2],
        "kwargs": {},
        "expected": 3,
        "compare": "exact",
    }]
    good = "def add(a, b):\n    return a + b\n"
    bad = "def add(a, b):\n    return 0\n"
    assert score_group([good, bad], cases) == [1.0, 0.0]


def test_score_group_fails_open():
    assert score_group([], [{"entry": {}}]) is None


def _engine():
    engine = MiningEngine.__new__(MiningEngine)
    engine.tokenizer = SimpleNamespace(
        eos_token_id=1,
        decode=lambda tokens: "```python\ndef add(a, b):\n    return a + b\n```",
    )
    engine.vllm_model = None
    return engine


def _generations():
    return [
        {"tokens": [0, 1], "prompt_length": 1}
        for _ in range(M_ROLLOUTS)
    ]


def test_unanimous_code_group_skips_proof():
    engine = _engine()
    env = SimpleNamespace(admission_reward_cases=lambda problem: [{"entry": {
        "kind": "function", "name": "add",
    }}])
    with patch(
        "reliquary.miner.code_zone_screen.score_group",
        return_value=[1.0] * M_ROLLOUTS,
    ):
        assert engine._code_zone_screen_reason(env, {}, _generations()) == "out_of_zone"


def test_short_code_completion_without_eos_skips_before_scoring():
    engine = _engine()
    engine.tokenizer.eos_token_id = 9
    generations = [
        {"tokens": [0, 2], "prompt_length": 1} for _ in range(M_ROLLOUTS)
    ]
    env = SimpleNamespace(admission_reward_cases=lambda problem: [{"entry": {}}])
    with patch("reliquary.miner.code_zone_screen.score_group") as score:
        assert engine._code_zone_screen_reason(env, {}, generations) == "bad_termination"
    score.assert_not_called()


def test_cap_without_eos_is_truncated_and_three_are_allowed():
    kind = _structural_termination_kind(
        [0, 0, 2, 2, 2], prompt_length=2, eos_ids={9}, cap=4,
    )
    assert kind == "truncated"
    assert _termination_upload_skip_reason(
        "opencodeinstruct", ["truncated"] * 3,
    ) is None
    assert _termination_upload_skip_reason(
        "opencodeinstruct", ["truncated"] * 4,
    ) == "too_many_truncated"


def test_eos_that_is_not_the_forced_pick_is_not_uploaded():
    # Token 3 is the stop. The proof forward puts essentially all mass on 0,
    # so the inverse-CDF pick is not the stop the engine emitted.
    logits = torch.zeros(2, 4)
    logits[0, 0] = 20.0
    kind = _proof_termination_kind(
        tokens=[1, 3],
        prompt_length=1,
        eos_ids={3},
        cap=8192,
        logits=logits,
        randomness="ab" * 32,
        prompt_idx=7,
        checkpoint_hash="c" * 40,
        rollout_index=0,
        token_logprobs=[-0.1],
    )
    assert kind == "bad_termination"


def test_eos_that_is_the_forced_pick_is_submitted():
    logits = torch.zeros(2, 4)
    logits[0, 3] = 20.0
    kind = _proof_termination_kind(
        tokens=[1, 3],
        prompt_length=1,
        eos_ids={3},
        cap=8192,
        logits=logits,
        randomness="ab" * 32,
        prompt_idx=7,
        checkpoint_hash="c" * 40,
        rollout_index=0,
        token_logprobs=[-0.1],
    )
    assert kind == "ok"


_CASES = [{"entry": {"kind": "function", "name": "add"}}]


STAGED_ROLLOUTS = 16


def _staged_code_engine(monkeypatch, *, tokens=(0, 1)):
    monkeypatch.setattr("reliquary.constants.MINER_UNANIMOUS_DROP_ROLLOUTS", 8)
    monkeypatch.setattr("reliquary.miner.engine.M_ROLLOUTS", STAGED_ROLLOUTS)
    engine = _engine()
    seen: list[list[int]] = []

    def _generate(*args, **kwargs):
        indices = kwargs["rollout_indices"]
        seen.append(list(indices))
        return [{"tokens": list(tokens), "prompt_length": 1} for _ in indices]

    engine._generate_m_rollouts = _generate
    return engine, seen


def _screen_code(engine):
    env = SimpleNamespace(admission_reward_cases=lambda problem: _CASES)
    return engine._generate_code_screened_rollouts(
        {}, "rand", env_name="opencodeinstruct", prompt_idx=3,
        checkpoint_hash="abc", env=env,
    )


def test_code_prefix_that_agrees_for_eight_rollouts_is_dropped(monkeypatch):
    engine, seen = _staged_code_engine(monkeypatch)
    with patch(
        "reliquary.miner.code_zone_screen.score_group",
        side_effect=lambda codes, cases: [1.0] * len(codes),
    ) as score:
        generations, reason = _screen_code(engine)

    assert generations is None and reason == "unanimous_prefix"
    assert seen == [list(range(4)), list(range(4, 8))]
    assert [len(call.args[0]) for call in score.call_args_list] == [4, 8]


def test_code_prefix_that_splits_generates_the_rest_at_once(monkeypatch):
    engine, seen = _staged_code_engine(monkeypatch)
    with patch(
        "reliquary.miner.code_zone_screen.score_group",
        return_value=[1.0, 0.0, 1.0, 1.0],
    ):
        generations, reason = _screen_code(engine)

    assert reason is None and len(generations) == STAGED_ROLLOUTS
    assert seen == [list(range(4)), list(range(4, STAGED_ROLLOUTS))]


def test_code_prefix_that_cannot_be_scored_finishes_the_group(monkeypatch):
    engine, seen = _staged_code_engine(monkeypatch)
    with patch("reliquary.miner.code_zone_screen.score_group", return_value=None):
        generations, reason = _screen_code(engine)

    assert reason is None and len(generations) == STAGED_ROLLOUTS
    assert seen == [list(range(4)), list(range(4, STAGED_ROLLOUTS))]


def test_code_prefix_with_a_bad_termination_is_dropped_unscored(monkeypatch):
    engine, seen = _staged_code_engine(monkeypatch, tokens=(0, 2))
    with patch("reliquary.miner.code_zone_screen.score_group") as score:
        generations, reason = _screen_code(engine)

    assert generations is None and reason == "bad_termination"
    assert seen == [list(range(4))]
    score.assert_not_called()


def test_code_cases_run_without_the_host_turn():
    engine = _engine()
    turn = engine._host_turn()
    free_while_scoring = []

    def _score(codes, cases):
        free_while_scoring.append(turn._lock.acquire(blocking=False))
        if free_while_scoring[-1]:
            turn._lock.release()
        return [1.0] * len(codes)

    env = SimpleNamespace(admission_reward_cases=lambda problem: _CASES)
    with turn.hold(), patch("reliquary.miner.code_zone_screen.score_group", _score):
        engine._code_screen_rewards(env, {}, _generations())
        assert not turn._lock.acquire(blocking=False)
    assert free_while_scoring == [True]


def test_yield_weighting_favours_the_lane_keeping_groups_fastest():
    tracker = _EnvironmentYield()
    tracker.start_window(1)
    for _ in range(6):
        tracker.observe("reliquary_logic_v2", 30.0, kept=True)
    for _ in range(6):
        tracker.observe("opencodeinstruct", 30.0, kept=False)

    weights = dict(tracker.weigh([
        ("reliquary_logic_v2", 10), ("opencodeinstruct", 10),
    ]))

    assert weights["reliquary_logic_v2"] > 5 * weights["opencodeinstruct"]
    assert tracker.weigh([("opencodeinstruct", 10)]) == [("opencodeinstruct", 10)]


def test_yield_history_decays_per_window_and_takes_back_refusals():
    tracker = _EnvironmentYield()
    tracker.start_window(1)
    tracker.observe("reliquary_logic_v2", 100.0, kept=True)
    tracker.observe("reliquary_logic_v2", 100.0, kept=True)
    before = tracker.rate("reliquary_logic_v2")

    tracker.retract("reliquary_logic_v2")
    assert tracker.rate("reliquary_logic_v2") < before

    assert tracker.start_window(1) is False
    assert tracker.start_window(2) is True
    decayed = tracker.rate("reliquary_logic_v2")
    assert decayed == pytest.approx(
        (0.8 + tracker.PRIOR_KEPT) / (160.0 + tracker.PRIOR_SECONDS),
    )


@pytest.mark.parametrize(
    ("drop", "stops"),
    [(8, [4, 8, 16]), (4, [4, 16]), (0, [4, 16]), (32, [4, 16])],
)
def test_unanimous_screen_stops(monkeypatch, drop, stops):
    monkeypatch.setattr("reliquary.constants.MINER_UNANIMOUS_DROP_ROLLOUTS", drop)
    assert _unanimous_screen_stops(16) == stops


def test_mixed_code_group_is_submitted():
    engine = _engine()
    env = SimpleNamespace(admission_reward_cases=lambda problem: [{"entry": {
        "kind": "function", "name": "add",
    }}])
    half = M_ROLLOUTS // 2
    rewards = [0.0] * half + [1.0] * (M_ROLLOUTS - half)
    with patch(
        "reliquary.miner.code_zone_screen.score_group",
        return_value=rewards,
    ):
        assert engine._code_zone_screen_reason(env, {}, _generations()) is None
