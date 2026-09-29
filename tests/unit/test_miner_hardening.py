"""Miner-side hardening: window polling, proof log-probs, local zone gate."""

from unittest.mock import MagicMock

import httpx
import pytest
import torch

from reliquary.miner.engine import (
    AttemptedPromptJournal,
    MiningEngine,
    NO_ACTIVE_WINDOW_POLL_SECONDS,
    ZONE_SCREEN_ROLLOUTS,
    _local_zone_skip_reason,
    _no_active_window_delay,
    _policy_token_logprobs,
    _prefix_screen_reason,
)
from reliquary.miner.submitter import (
    NoActiveWindowError,
    SubmissionError,
    get_miner_state_v1,
    get_window_state_v2,
)

EOS = 99
CORRECT = "reasoning ... \\boxed{1}"
WRONG = "reasoning ... \\boxed{2}"


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [(None, NO_ACTIVE_WINDOW_POLL_SECONDS), (0, 1.0), (3, 3.0), (600, 10.0),
     ("junk", NO_ACTIVE_WINDOW_POLL_SECONDS), (float("nan"), 1.0)],
)
def test_no_active_window_delay_is_fast_and_bounded(retry_after, expected):
    exc = NoActiveWindowError("between windows", retry_after=retry_after)
    assert _no_active_window_delay(exc) == expected


def test_no_active_window_error_is_a_submission_error():
    assert issubclass(NoActiveWindowError, SubmissionError)


@pytest.mark.asyncio
async def test_miner_state_503_raises_no_active_window(monkeypatch):
    async def _get(self, url, headers=None, timeout=None):
        return httpx.Response(503, headers={"Retry-After": "2"})

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)
    async with httpx.AsyncClient() as client:
        with pytest.raises(NoActiveWindowError) as caught:
            await get_miner_state_v1("http://fake", client=client)
    assert caught.value.retry_after == 2


@pytest.mark.asyncio
async def test_state_503_raises_no_active_window(monkeypatch):
    async def _get(self, url, timeout=None):
        return httpx.Response(503, json={"detail": "no_active_window"})

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)
    async with httpx.AsyncClient() as client:
        with pytest.raises(NoActiveWindowError):
            await get_window_state_v2("http://fake", client=client)


@pytest.mark.parametrize("workspace_bytes", [4 * 1024 * 1024, 1])
def test_policy_logprobs_match_full_sequence_log_softmax(workspace_bytes):
    generator = torch.Generator().manual_seed(0)
    seq_len, vocab = 37, 1024
    logits = torch.randn(seq_len, vocab, generator=generator).to(torch.bfloat16)
    tokens = torch.randint(0, vocab, (seq_len,), generator=generator).tolist()
    positions = list(range(5, seq_len))

    full = torch.log_softmax(logits.float(), dim=-1)
    expected = [full[i - 1, tokens[i]].item() for i in positions]

    got = _policy_token_logprobs(
        logits, tokens, positions, workspace_bytes=workspace_bytes,
    )
    assert got == expected


def test_policy_logprobs_empty_positions():
    assert _policy_token_logprobs(torch.zeros(3, 8), [1, 2, 3], []) == []


@pytest.fixture
def v1_math_gates(monkeypatch):
    monkeypatch.setattr("reliquary.constants.MATH_ANSWER_FORMAT", "boxed")
    monkeypatch.setattr("reliquary.constants.SIGMA_MIN", 0.24)
    monkeypatch.setattr("reliquary.constants.MAX_TRUNCATED_PER_SUBMISSION", 1)
    monkeypatch.setattr(
        "reliquary.constants.MAX_TRUNCATED_PER_SUBMISSION_BY_ENV",
        {"opencodeinstruct": 3},
    )
    monkeypatch.setattr(
        "reliquary.constants.ROBUST_TRUNCATION_UTILITY_ENABLED", True,
    )


def _group(texts, *, truncated=()):
    completions = [
        [1, 2, 3] if index in truncated else [1, 2, EOS]
        for index in range(len(texts))
    ]
    rewards = [1.0 if text == CORRECT else 0.0 for text in texts]
    return dict(
        env_name="openmathinstruct",
        rewards=rewards,
        completions=completions,
        texts=list(texts),
        eos_ids={EOS},
    )


@pytest.mark.parametrize("correct", [1, 8, 15])
def test_zone_gate_passes_every_nondegenerate_binary_group(v1_math_gates, correct):
    texts = [CORRECT] * correct + [WRONG] * (16 - correct)
    assert _local_zone_skip_reason(**_group(texts)) is None


@pytest.mark.parametrize("text", [CORRECT, WRONG])
def test_zone_gate_skips_degenerate_groups(v1_math_gates, text):
    assert _local_zone_skip_reason(**_group([text] * 16)) == "out_of_zone"


def test_zone_gate_skips_beyond_truncation_allowance(v1_math_gates):
    texts = [CORRECT] * 8 + [WRONG] * 8
    assert _local_zone_skip_reason(
        **_group(texts, truncated={0, 9})
    ) == "too_many_truncated"


def test_zone_gate_skips_malformed_final_box(v1_math_gates):
    texts = [CORRECT] * 8 + [WRONG] * 7 + ["\\boxed{<|im_end|>"]
    assert _local_zone_skip_reason(**_group(texts)) == "malformed_final_answer"


def test_zone_gate_values_unboxed_rollout_at_every_outcome(v1_math_gates):
    # Unboxed -> 0 gives 15/16; unboxed -> 1 gives 16/16, which is degenerate.
    texts = [CORRECT] * 15 + ["the answer is 1"]
    assert _local_zone_skip_reason(**_group(texts)) == "out_of_zone"
    # Either outcome of the unboxed rollout still leaves 1 or 2 of 16 correct.
    texts = [CORRECT] + [WRONG] * 14 + ["the answer is 1"]
    assert _local_zone_skip_reason(**_group(texts)) is None


def test_zone_gate_values_truncated_rollout_at_every_outcome(v1_math_gates):
    texts = [WRONG] * 15 + [CORRECT]
    # The only success is truncated: valued at 0 the group is 0/16.
    assert _local_zone_skip_reason(
        **_group(texts, truncated={15})
    ) == "out_of_zone"


def test_rollout_submission_uses_precomputed_reward():
    class _MathEnv:
        name = "openmathinstruct"
        validator_authoritative_reward = False
        compute_reward = MagicMock(side_effect=AssertionError("already scored"))

    eng = object.__new__(MiningEngine)
    eng.env = _MathEnv()
    eng.tokenizer = MagicMock()
    eng._build_grail_commit = MagicMock(return_value={"tokens": [1, 2, 3]})

    rollout = eng._build_rollout_submission(
        {"tokens": [1, 2, 3], "prompt_length": 1}, {"prompt": "p"}, "r",
        env=eng.env, reward=1.0,
    )
    assert rollout.reward == 1.0
    eng.env.compute_reward.assert_not_called()


def test_authoritative_env_ignores_precomputed_reward():
    class _CodeEnv:
        name = "opencodeinstruct"
        validator_authoritative_reward = True

    eng = object.__new__(MiningEngine)
    eng.env = _CodeEnv()
    eng.tokenizer = MagicMock()
    eng._build_grail_commit = MagicMock(return_value={"tokens": [1, 2, 3]})

    rollout = eng._build_rollout_submission(
        {"tokens": [1, 2, 3], "prompt_length": 1}, {"prompt": "p"}, "r",
        env=eng.env, reward=1.0,
    )
    assert rollout.reward == 0.0


def test_local_zone_filter_scope(monkeypatch):
    monkeypatch.setattr("reliquary.constants.BFT_ENABLED", False)
    monkeypatch.setattr("reliquary.constants.MINER_LOCAL_ZONE_FILTER", True)
    eng = object.__new__(MiningEngine)
    math_env = type("E", (), {"validator_authoritative_reward": False})()
    code_env = type("E", (), {"validator_authoritative_reward": True})()

    assert eng._local_zone_filter_applies("openmathinstruct", math_env)
    assert not eng._local_zone_filter_applies("opencodeinstruct", code_env)

    monkeypatch.setattr("reliquary.constants.MINER_LOCAL_ZONE_FILTER", False)
    assert not eng._local_zone_filter_applies("openmathinstruct", math_env)

    monkeypatch.setattr("reliquary.constants.MINER_LOCAL_ZONE_FILTER", True)
    monkeypatch.setattr("reliquary.constants.BFT_ENABLED", True)
    assert not eng._local_zone_filter_applies("openmathinstruct", math_env)


def test_score_generations_matches_submission_decoding():
    env = MagicMock()
    env.compute_reward.side_effect = lambda problem, text: float(text == "ok")
    eng = object.__new__(MiningEngine)
    eng.tokenizer = MagicMock()
    eng.tokenizer.decode.side_effect = lambda ids: "ok" if ids == [7, EOS] else "no"

    completions, texts, rewards = eng._score_generations(
        env, {"prompt": "p"},
        [{"tokens": [1, 2, 7, EOS], "prompt_length": 2},
         {"tokens": [1, 2, 8, EOS], "prompt_length": 2}],
    )
    assert completions == [[7, EOS], [8, EOS]]
    assert texts == ["ok", "no"]
    assert rewards == [1.0, 0.0]


def test_prefix_screen_finishes_a_mixed_group(v1_math_gates):
    texts = [CORRECT, WRONG, CORRECT, WRONG]
    assert _prefix_screen_reason(
        **_group(texts), max_new_tokens=8192,
    ) is None


def test_prefix_screen_finishes_a_short_unanimous_group(v1_math_gates):
    assert _prefix_screen_reason(
        **_group([CORRECT] * 4), max_new_tokens=8192,
    ) is None


def test_prefix_screen_drops_a_long_unanimous_group(v1_math_gates):
    group = _group([CORRECT] * 4)
    group["completions"] = [[1] * 4999 + [EOS] for _ in range(4)]
    assert _prefix_screen_reason(**group, max_new_tokens=8192) == "expensive_extreme"


def test_prefix_screen_drops_a_second_truncation(v1_math_gates):
    assert _prefix_screen_reason(
        **_group([CORRECT, WRONG, CORRECT, WRONG], truncated={0, 1}),
        max_new_tokens=8192,
    ) == "too_many_truncated"


def test_zone_screen_skips_the_tail_and_keeps_forced_indices(monkeypatch):
    monkeypatch.setattr("reliquary.constants.BFT_ENABLED", False)
    monkeypatch.setattr("reliquary.constants.MINER_LOCAL_ZONE_FILTER", True)
    eng = object.__new__(MiningEngine)
    eng.vllm_model = MagicMock()
    eng.tokenizer = MagicMock()
    eng.max_new_tokens = 8192
    seen: list[list[int]] = []

    def _generate(*args, **kwargs):
        indices = kwargs["rollout_indices"]
        seen.append(indices)
        return [
            {"tokens": [1, EOS], "prompt_length": 1} for _ in indices
        ]

    eng._generate_m_rollouts = _generate
    eng._score_generations = MagicMock(return_value=(
        [[1] * 4999 + [EOS]] * ZONE_SCREEN_ROLLOUTS,
        [CORRECT] * ZONE_SCREEN_ROLLOUTS,
        [1.0] * ZONE_SCREEN_ROLLOUTS,
    ))
    monkeypatch.setattr(
        "reliquary.shared.modeling.resolve_eos_token_ids", lambda *args: {EOS},
    )

    generations, reason = eng._generate_zone_screened_rollouts(
        {"prompt": "p"}, "rand", env_name="openmathinstruct", prompt_idx=3,
        checkpoint_hash="abc", env=MagicMock(),
    )
    assert generations is None and reason == "expensive_extreme"
    assert seen == [list(range(ZONE_SCREEN_ROLLOUTS))]


def test_zone_screen_finishes_the_remaining_forced_indices(monkeypatch):
    monkeypatch.setattr("reliquary.constants.BFT_ENABLED", False)
    monkeypatch.setattr("reliquary.constants.MINER_LOCAL_ZONE_FILTER", True)
    from reliquary.constants import M_ROLLOUTS

    eng = object.__new__(MiningEngine)
    eng.vllm_model = MagicMock()
    eng.tokenizer = MagicMock()
    eng.max_new_tokens = 8192
    seen: list[list[int]] = []

    def _generate(*args, **kwargs):
        indices = kwargs["rollout_indices"]
        seen.append(list(indices))
        return [{"tokens": [1, EOS], "prompt_length": 1} for _ in indices]

    eng._generate_m_rollouts = _generate
    eng._score_generations = MagicMock(return_value=(
        [[1, EOS]] * ZONE_SCREEN_ROLLOUTS,
        [CORRECT, WRONG, CORRECT, WRONG][:ZONE_SCREEN_ROLLOUTS],
        [1.0, 0.0, 1.0, 0.0][:ZONE_SCREEN_ROLLOUTS],
    ))
    monkeypatch.setattr(
        "reliquary.shared.modeling.resolve_eos_token_ids", lambda *args: {EOS},
    )

    generations, reason = eng._generate_zone_screened_rollouts(
        {"prompt": "p"}, "rand", env_name="openmathinstruct", prompt_idx=3,
        checkpoint_hash="abc", env=MagicMock(),
    )
    assert reason is None and len(generations) == M_ROLLOUTS
    assert seen == [
        list(range(ZONE_SCREEN_ROLLOUTS)),
        list(range(ZONE_SCREEN_ROLLOUTS, M_ROLLOUTS)),
    ]


def test_attempted_prompt_journal_is_window_scoped(tmp_path):
    journal = AttemptedPromptJournal(tmp_path / "attempted.json")
    journal.save(
        window_n=5, randomness="r", checkpoint_hash="c",
        attempted={"openmathinstruct": {3, 1}},
    )
    assert journal.load(
        window_n=5, randomness="r", checkpoint_hash="c",
    ) == {"openmathinstruct": {1, 3}}
    assert journal.load(
        window_n=6, randomness="r", checkpoint_hash="c",
    ) is None
