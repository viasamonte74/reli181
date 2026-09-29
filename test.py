"""Miner preflight: prove this host is on the validator's live contract.

Loads no model and needs no wallet. Run it before `reliquary mine`:

    RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-base-dapo-reliquary-v1 \
    RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED=1 \
    python test.py --validator-url http://62.238.81.36:8000

Exits 0 only when `_state_matches_active_protocol(state)` is True against the
live `/state`. Between windows the validator answers 503; the check waits for
the next state up to `--wait-seconds` rather than judging a missing state.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from importlib import metadata

EXPECTED_PROFILE = "qwen3-4b-base-dapo-reliquary-v1"
REQUIRED_ENV = {
    "RELIQUARY_PROTOCOL_PROFILE": EXPECTED_PROFILE,
    "RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED": "1",
}
PACKAGES = {
    "torch_version": "torch",
    "transformers_version": "transformers",
    "fla_version": "flash-linear-attention",
    "fla_core_version": "fla-core",
    "causal_conv1d_version": "causal-conv1d",
    "flash_attn_version": "flash-attn",
}


def _installed(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _same_version(local: str | None, remote: str | None) -> bool:
    # torch reports "2.7.0+cu128" locally; the fingerprint may carry either form.
    if local is None or remote is None:
        return local == remote
    return local.split("+")[0] == remote.split("+")[0]


def check_environment() -> list[str]:
    problems = []
    for name, expected in REQUIRED_ENV.items():
        value = os.environ.get(name)
        status = "ok" if value == expected else "WRONG"
        print(f"  env {name}={value!r} [{status}]")
        if value != expected:
            problems.append(f"{name} must be {expected!r}")
    return problems


def check_constants() -> list[str]:
    from reliquary.constants import FILL_CLOSED_ENABLED, M_ROLLOUTS, PROMPT_RANGE_SIZE
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE

    profile = ACTIVE_PROTOCOL_PROFILE
    print(
        f"  active profile={profile.profile_id} protocol_v{profile.protocol_version} "
        f"model={profile.model_id}@{profile.model_revision[:12]}"
    )
    print(
        f"  fill_closed={FILL_CLOSED_ENABLED} rollouts={M_ROLLOUTS} "
        f"prompt_range_size={PROMPT_RANGE_SIZE} envs={sorted(profile.environments)}"
    )
    problems = []
    if profile.profile_id != EXPECTED_PROFILE:
        problems.append(f"active profile is {profile.profile_id}, not {EXPECTED_PROFILE}")
    if not FILL_CLOSED_ENABLED:
        problems.append("fill-closed windows are not enabled")
    return problems


def check_gpu() -> list[str]:
    try:
        import torch
    except ImportError:
        return ["torch is not installed"]
    print(f"  torch={torch.__version__} cuda={torch.version.cuda} "
          f"available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        return ["CUDA is not available to torch"]
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        print(f"  cuda:{index} {torch.cuda.get_device_name(index)} "
              f"free={free / 2**30:.1f}GiB total={total / 2**30:.1f}GiB")
    if torch.cuda.device_count() == 1:
        print("  single device: generation and proof share cuda:0")
    return []


async def check_runtime(url: str, client) -> list[str]:
    from reliquary.miner.submitter import SubmissionError, get_runtime_contract_v1

    try:
        contract = await get_runtime_contract_v1(url, client=client)
    except (SubmissionError, ValueError) as exc:
        print(f"  /runtime-contract unavailable: {exc}")
        return []
    remote = contract.validator_profile
    problems = []
    for field, distribution in PACKAGES.items():
        want = getattr(remote, field, None)
        have = _installed(distribution)
        ok = _same_version(have, want)
        # Qwen3-4B-Base is a softmax transformer. causal-conv1d only feeds the
        # Qwen3.5 GatedDeltaNet path, which this profile never executes, and
        # the miner plan keeps it uninstalled.
        unused = distribution == "causal-conv1d"
        label = "ok" if ok else ("unused" if unused else "DIFFERS")
        print(f"  {distribution}: local={have} validator={want} [{label}]")
        if not ok and not unused:
            problems.append(f"{distribution} local={have} validator={want}")
    print(f"  validator attention={remote.proof_attention_implementation} "
          f"dtype={remote.proof_dtype} gpu={remote.gpu_name}")
    return problems


def _contract_diff(local: dict, remote: dict | None, prefix: str = "") -> list[str]:
    if not isinstance(remote, dict):
        return [f"{prefix or 'generation_contract'}: remote={remote!r}"]
    lines = []
    for key in sorted(set(local) | set(remote)):
        path = f"{prefix}.{key}" if prefix else key
        a, b = local.get(key), remote.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            lines.extend(_contract_diff(a, b, path))
        elif a != b:
            lines.append(f"{path}: local={a!r} remote={b!r}")
    return lines


async def check_state(url: str, client, wait_seconds: float) -> list[str]:
    from reliquary.miner.engine import _state_matches_active_protocol
    from reliquary.miner.submitter import NoActiveWindowError, get_window_state_v2
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, to_generation_contract
    from reliquary.protocol.submission import WindowState

    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            state = await get_window_state_v2(url, client=client)
        except NoActiveWindowError:
            state = None
        if state is not None:
            break
        if time.monotonic() >= deadline:
            return [f"validator served no window state within {wait_seconds:.0f}s (503)"]
        await asyncio.sleep(1.0)

    if state.state != WindowState.OPEN or not state.randomness:
        print("  note: window is not OPEN with randomness yet; the miner will wait for it")
    print(f"  window={state.window_n} state={state.state.value} "
          f"checkpoint_n={state.checkpoint_n} "
          f"{state.checkpoint_repo_id}@{(state.checkpoint_revision or '')[:12]}")
    print(f"  remote profile={state.generation_profile_id} protocol_v{state.protocol_version}")
    matches = _state_matches_active_protocol(state)
    print(f"  _state_matches_active_protocol(state) -> {matches}")
    if matches:
        return []
    for line in _contract_diff(
        to_generation_contract(ACTIVE_PROTOCOL_PROFILE), state.generation_contract,
    ):
        print(f"    {line}")
    return ["validator generation contract does not match the local profile"]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--validator-url", default="http://62.238.81.36:8000")
    parser.add_argument("--wait-seconds", type=float, default=300.0)
    args = parser.parse_args()

    problems: list[str] = []
    print("[environment]")
    problems += check_environment()
    if problems:
        # The profile is resolved at import time; importing now would only
        # report the stale default.
        print("\nFAIL:\n  " + "\n  ".join(problems))
        return 1
    print("[constants]")
    problems += check_constants()
    print("[gpu]")
    problems += check_gpu()

    import httpx

    async with httpx.AsyncClient(timeout=15) as client:
        print("[runtime]")
        problems += await check_runtime(args.validator_url, client)
        print("[state]")
        problems += await check_state(args.validator_url, client, args.wait_seconds)

    if problems:
        print("\nFAIL:\n  " + "\n  ".join(problems))
        return 1
    print("\nPASS: safe to start `reliquary mine`")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
