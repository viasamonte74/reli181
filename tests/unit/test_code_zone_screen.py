"""Code groups that cannot clear the zone are dropped before the GRAIL proof."""

from types import SimpleNamespace
from unittest.mock import patch

from reliquary.constants import M_ROLLOUTS
from reliquary.miner.code_zone_screen import score_group
from reliquary.miner.engine import MiningEngine, _eligible_generation_mix


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
