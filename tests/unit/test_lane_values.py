"""Price-aware lane weighting and prompt-source warm-up in the miner."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from reliquary.miner.engine import (
    _EnvironmentWarmup,
    _EnvironmentYield,
    _LaneValues,
    _live_lane_values,
    _parse_lane_values,
    _prefetch_prompt_range,
    _weigh_lanes,
)

MATH, LOGIC, CODE = "openmathinstruct", "reliquary_logic_v2", "opencodeinstruct"


def _tasks(price, task_id="reliquary"):
    return {"tasks": [{"task_id": task_id, "price": price}]}


def test_parse_lane_values_keeps_positive_numbers_only():
    assert _parse_lane_values(
        f" {MATH}=2.66, {LOGIC}=1 ,bad, {CODE}=0,x=nan,=3,y=abc",
    ) == {MATH: 2.66, LOGIC: 1.0}
    assert _parse_lane_values("") == {}


def test_values_scale_the_mix_by_value():
    weighed = dict(_weigh_lanes([(MATH, 1), (LOGIC, 1)], values={MATH: 3.0, LOGIC: 1.0}))
    assert weighed[MATH] == pytest.approx(3 * weighed[LOGIC])


def test_value_and_yield_multiply():
    environment_yield = _EnvironmentYield()
    environment_yield.observe(LOGIC, 60, kept=True)
    environment_yield.observe(LOGIC, 60, kept=True)
    environment_yield.observe(LOGIC, 60, kept=True)
    environment_yield.observe(MATH, 180, kept=True)
    # Logic keeps twice as often per second; a math seat pays three times more.
    weighed = dict(environment_yield.weigh(
        [(MATH, 1), (LOGIC, 1)], values={MATH: 3.0, LOGIC: 1.0},
    ))
    assert weighed[MATH] / weighed[LOGIC] == pytest.approx(1.5)
    assert environment_yield.weigh([(MATH, 1), (LOGIC, 1)]) == _weigh_lanes(
        [(MATH, 1), (LOGIC, 1)],
        rates={MATH: environment_yield.rate(MATH), LOGIC: environment_yield.rate(LOGIC)},
    )


def test_single_lane_and_missing_values_leave_the_mix_alone():
    assert _weigh_lanes([(MATH, 5)], values={MATH: 9.0}) == [(MATH, 5)]
    assert _weigh_lanes([(MATH, 1), (LOGIC, 2)], values=None) == [(MATH, 1), (LOGIC, 2)]


def test_live_lane_values_read_the_per_environment_price():
    price = {
        "applied": True,
        "value": 0.4,
        "by_environment": {
            MATH: {"value": 0.53, "regime": "hold"},
            LOGIC: {"value": 0.2, "regime": "descend"},
            CODE: {"value": None},
        },
    }
    assert _live_lane_values(_tasks(price)) == {MATH: 0.53, LOGIC: 0.2}


def test_live_lane_values_pick_the_matching_task():
    payload = {"tasks": [
        {"task_id": "other", "price": {"applied": True, "by_environment": {MATH: {"value": 9}}}},
        {"task_id": "mine", "price": {"applied": True, "by_environment": {MATH: {"value": 2}}}},
    ]}
    assert _live_lane_values(payload, "mine") == {MATH: 2.0}
    assert _live_lane_values(payload, "absent") == {MATH: 9.0}


def test_an_unapplied_or_single_price_pays_every_lane_alike():
    assert _live_lane_values(_tasks({"applied": False, "value": 0.3})) == {}
    assert _live_lane_values(_tasks({"applied": True, "value": 0.3})) == {}


@pytest.mark.parametrize("payload", [None, {}, {"tasks": []}, _tasks(None), {"tasks": "x"}])
def test_no_usable_price(payload):
    assert _live_lane_values(payload) is None


def test_configured_values_fill_in_until_a_live_price_arrives():
    lanes = _LaneValues({MATH: 2.66}, live=True)
    assert lanes.values([MATH, LOGIC]) == {MATH: 2.66, LOGIC: 2.66}
    lanes = _LaneValues({MATH: 2.66, LOGIC: 1.0}, live=True)
    assert lanes.values([MATH, LOGIC]) == {MATH: 2.66, LOGIC: 1.0}
    assert lanes.source == "configured"
    lanes._live = {MATH: 0.2, LOGIC: 0.4}
    assert lanes.values([MATH, LOGIC]) == {MATH: 0.2, LOGIC: 0.4}
    lanes._live = {}
    assert lanes.values([MATH, LOGIC]) is None
    assert lanes.source == "live-uniform"
    assert _LaneValues({}, live=False).values([MATH]) is None


class _Client:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.urls = []

    async def get(self, url, timeout=None):
        self.urls.append(url)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _response(status, body=None):
    return SimpleNamespace(status_code=status, json=lambda: body)


def _refresh(lanes, client):
    async def run():
        lanes.maybe_refresh(client, "http://validator/")
        if lanes._task is not None:
            await lanes._task
    asyncio.run(run())


def test_refresh_adopts_a_published_price():
    lanes = _LaneValues({MATH: 2.66, LOGIC: 1.0})
    client = _Client(_response(200, _tasks({
        "applied": True, "by_environment": {MATH: {"value": 0.1}, LOGIC: {"value": 0.3}},
    })))
    _refresh(lanes, client)
    assert client.urls == ["http://validator/tasks"]
    assert lanes.values([MATH, LOGIC]) == {MATH: 0.1, LOGIC: 0.3}


def test_an_unpublished_price_keeps_configured_values_and_backs_off():
    lanes = _LaneValues({MATH: 2.66, LOGIC: 1.0})
    client = _Client(_response(404))
    _refresh(lanes, client)
    assert lanes.values([MATH, LOGIC]) == {MATH: 2.66, LOGIC: 1.0}
    _refresh(lanes, client)
    assert len(client.urls) == 1
    assert lanes._retry_at - time.monotonic() > _LaneValues.RETRY_SECONDS


def test_a_failed_fetch_keeps_the_last_price():
    lanes = _LaneValues({})
    lanes._live = {MATH: 0.5}
    _refresh(lanes, _Client(OSError("down")))
    assert lanes.values([MATH]) == {MATH: 0.5}
    lanes._retry_at = 0.0
    _refresh(lanes, _Client(_response(200, {"tasks": []})))
    assert lanes.values([MATH]) == {MATH: 0.5}


def test_live_prices_can_be_switched_off():
    lanes = _LaneValues({MATH: 2.0}, live=False)
    client = _Client()
    _refresh(lanes, client)
    assert client.urls == []


class _SlowEnv:
    def __init__(self, gate=None, fail=0):
        self.gate = gate
        self.fail = fail
        self.calls = 0

    def __len__(self):
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            self.fail -= 1
            raise RuntimeError("hub down")
        return 10


def _wait(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_warmup_holds_a_lane_back_until_its_prompt_source_is_built():
    gate = threading.Event()
    warmup = _EnvironmentWarmup({MATH: _SlowEnv(gate), LOGIC: _SlowEnv()})
    _wait(lambda: warmup.ready(LOGIC))
    assert warmup.pending([MATH, LOGIC]) == [MATH]
    gate.set()
    _wait(lambda: warmup.ready(MATH))
    assert warmup.pending([MATH, LOGIC]) == []


def test_a_failed_warmup_is_retried(monkeypatch):
    monkeypatch.setattr(_EnvironmentWarmup, "RETRY_SECONDS", 0.0)
    env = _SlowEnv(fail=1)
    warmup = _EnvironmentWarmup({MATH: env})
    _wait(lambda: warmup._futures[MATH].done())
    assert not warmup.ready(MATH)  # failure noticed, retry scheduled
    assert not warmup.ready(MATH)  # retry started
    _wait(lambda: warmup.ready(MATH))
    assert env.calls == 2


def test_unknown_lanes_count_as_ready():
    assert _EnvironmentWarmup({}).ready(MATH)


def test_prompt_range_prefetch_runs_in_the_background():
    done = threading.Event()
    calls = []

    class _Dataset:
        def prefetch(self, lo, hi):
            calls.append((lo, hi, threading.current_thread().name))
            done.set()
            return 5

    assert _prefetch_prompt_range(MATH, SimpleNamespace(_dataset=_Dataset()), (10, 5010))
    assert done.wait(2)
    assert calls[0][:2] == (10, 5010)
    assert calls[0][2] != threading.current_thread().name
    assert not _prefetch_prompt_range(LOGIC, SimpleNamespace(), (0, 10))
