"""Code groups that cannot clear the zone are dropped before the GRAIL proof."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from reliquary.constants import M_ROLLOUTS
from reliquary.miner.code_zone_screen import score_group
from reliquary.miner.engine import (
    MiningEngine,
    _eligible_generation_mix,
    _proof_termination_kind,
    _structural_termination_kind,
    _termination_upload_skip_reason,
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
