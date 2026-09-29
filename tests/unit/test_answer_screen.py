"""Staged screen for validator-scored JSON-answer environments (logic)."""

from unittest.mock import MagicMock

import pytest

from reliquary.miner.engine import MiningEngine

EOS = 99
ROLLOUTS = 16
LOGIC = "reliquary_logic_v2"


class _Env:
    validator_authoritative_reward = True

    def compute_reward(self, problem, completion):
        return 1.0


@pytest.fixture
def answer_screen(monkeypatch):
    monkeypatch.setattr("reliquary.constants.BFT_ENABLED", False)
    monkeypatch.setattr("reliquary.constants.MINER_ANSWER_SCREEN", True)
    monkeypatch.setattr("reliquary.constants.MINER_UNANIMOUS_DROP_ROLLOUTS", 8)
    monkeypatch.setattr(
        "reliquary.constants.MAX_TRUNCATED_PER_SUBMISSION_BY_ENV", {LOGIC: 2},
    )
    monkeypatch.setattr("reliquary.constants.SIGMA_MIN", 0.24)
    monkeypatch.setattr(
        "reliquary.constants.ROBUST_TRUNCATION_UTILITY_ENABLED", True,
    )
    monkeypatch.setattr("reliquary.miner.engine.M_ROLLOUTS", ROLLOUTS)
    monkeypatch.setattr(
        "reliquary.shared.modeling.resolve_eos_token_ids", lambda *args: {EOS},
    )


def _engine(rewards, *, truncated=(), score_error=None):
    """Rollout ``i`` scores ``rewards[i]``; indices in ``truncated`` never stop."""
    eng = object.__new__(MiningEngine)
    eng.vllm_model = MagicMock()
    eng.tokenizer = MagicMock()
    seen: list[list[int]] = []

    def _generate(*args, **kwargs):
        indices = kwargs.get("rollout_indices") or list(range(ROLLOUTS))
        seen.append(list(indices))
        return [
            {
                "tokens": [0, 1] if i in truncated else [0, 1, EOS],
                "prompt_length": 1,
                "index": i,
            }
            for i in indices
        ]

    def _score(env, problem, generations):
        if score_error is not None:
            raise score_error
        return (
            [list(g["tokens"][1:]) for g in generations],
            ["{}"] * len(generations),
            [rewards[g["index"]] for g in generations],
        )

    eng._generate_m_rollouts = _generate
    eng._score_generations = _score
    return eng, seen


def _screen(eng):
    return eng._generate_answer_screened_rollouts(
        {"prompt": "p"}, "rand", env_name=LOGIC, prompt_idx=3,
        checkpoint_hash="abc", env=_Env(),
    )


def test_applies_only_to_validator_scored_json_answer_environments(answer_screen):
    eng = object.__new__(MiningEngine)
    assert eng._answer_screen_applies(LOGIC, _Env())
    assert not eng._answer_screen_applies("openmathinstruct", _Env())
    assert not eng._answer_screen_applies("opencodeinstruct", _Env())
    assert not eng._answer_screen_applies(LOGIC, object())
    assert not eng._answer_screen_applies("no_such_environment", _Env())


@pytest.mark.parametrize(
    "setting", ["reliquary.constants.MINER_ANSWER_SCREEN",
                "reliquary.constants.BFT_ENABLED"],
)
def test_switches_turn_the_screen_off(answer_screen, monkeypatch, setting):
    enabled = setting.endswith("MINER_ANSWER_SCREEN")
    monkeypatch.setattr(setting, not enabled)
    assert not object.__new__(MiningEngine)._answer_screen_applies(LOGIC, _Env())


def test_drops_a_unanimous_prefix_of_eight(answer_screen):
    eng, seen = _engine([0.0] * ROLLOUTS)

    generations, reason = _screen(eng)

    assert generations is None and reason == "unanimous_prefix"
    assert seen == [list(range(4)), list(range(4, 8))]


def test_finishes_once_the_second_stage_splits(answer_screen):
    rewards = [0.0] * ROLLOUTS
    rewards[5] = 1.0
    eng, seen = _engine(rewards)

    generations, reason = _screen(eng)

    assert reason is None and len(generations) == ROLLOUTS
    assert seen == [list(range(4)), list(range(4, 8)), list(range(8, ROLLOUTS))]


def test_a_truncated_rollout_cannot_split_a_prefix(answer_screen):
    rewards = [0.0] * ROLLOUTS
    rewards[2] = 1.0
    eng, seen = _engine(rewards, truncated={2})

    generations, reason = _screen(eng)

    assert generations is None and reason == "unanimous_prefix"
    assert seen == [list(range(4)), list(range(4, 8))]


def test_a_clean_split_beside_a_truncated_rollout_is_finished(answer_screen):
    rewards = [0.0] * ROLLOUTS
    rewards[1] = 1.0
    eng, seen = _engine(rewards, truncated={2})

    generations, reason = _screen(eng)

    assert reason is None and len(generations) == ROLLOUTS
    assert seen == [list(range(4)), list(range(4, ROLLOUTS))]


def test_full_group_unanimous_but_for_a_truncated_rollout_is_not_proved(
    answer_screen, monkeypatch,
):
    monkeypatch.setattr("reliquary.constants.MINER_UNANIMOUS_DROP_ROLLOUTS", 0)
    monkeypatch.setattr(
        "reliquary.constants.MAX_TRUNCATED_PER_SUBMISSION_BY_ENV", {LOGIC: 1},
    )
    rewards = [0.0] * ROLLOUTS
    rewards[12] = 1.0
    eng, _ = _engine(rewards, truncated={12})

    generations, reason = _screen(eng)

    assert generations is None and reason == "out_of_zone"


def test_too_many_truncated_is_dropped_early(answer_screen):
    eng, seen = _engine([0.0] * ROLLOUTS, truncated={0, 1, 2})

    generations, reason = _screen(eng)

    assert generations is None and reason == "too_many_truncated"
    assert seen == [list(range(4))]


def test_scoring_failure_finishes_the_group(answer_screen):
    eng, seen = _engine([0.0] * ROLLOUTS, score_error=RuntimeError("checker"))

    generations, reason = _screen(eng)

    assert reason is None and len(generations) == ROLLOUTS
    assert seen == [list(range(4)), list(range(4, ROLLOUTS))]


def test_unanimous_full_group_is_not_proved(answer_screen, monkeypatch):
    monkeypatch.setattr("reliquary.constants.MINER_UNANIMOUS_DROP_ROLLOUTS", 0)
    eng, seen = _engine([1.0] * ROLLOUTS)

    generations, reason = _screen(eng)

    assert generations is None and reason == "out_of_zone"
    assert seen == [list(range(4)), list(range(4, ROLLOUTS))]


def test_one_correct_answer_in_sixteen_is_in_zone(answer_screen, monkeypatch):
    monkeypatch.setattr("reliquary.constants.MINER_UNANIMOUS_DROP_ROLLOUTS", 0)
    rewards = [0.0] * ROLLOUTS
    rewards[15] = 1.0
    eng, _ = _engine(rewards)

    generations, reason = _screen(eng)

    assert reason is None and len(generations) == ROLLOUTS
