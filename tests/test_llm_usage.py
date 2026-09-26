"""요청 한도 관리: 가짜 시계와 가짜 클라이언트로 시험한다 (실제 요청 없음)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel

from excursion_tracer.llm.base import CallContext, SchemaError
from excursion_tracer.llm.fake import FakeClient, FakeDailyQuota, FakeRateLimit
from excursion_tracer.llm.usage import KST, QuotaExhausted, UsageTracker


class Out(BaseModel):
    x: int


class Clock:
    def __init__(self, start: datetime):
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += timedelta(seconds=s)


# 2026-09-26 10:00 KST = 2026-09-26 01:00 UTC = 2026-09-25 18:00 PDT
START = datetime(2026, 9, 26, 1, 0, tzinfo=timezone.utc)


def make(tmp_path, rpm=15, rpd=500, start=START):
    clock = Clock(start)
    tr = UsageTracker(rpm, rpd, 0.9, "America/Los_Angeles", log_path=tmp_path / "usage.jsonl",
                      clock=clock, sleep=clock.sleep)
    return tr, clock


def ok_client(tr):
    return FakeClient(tr, lambda s, p, schema: '{"x": 1}')


def call(client, scn="scn_1"):
    return client.structured("sys", "prompt", Out, 100, 0.0, CallContext(run_id="r", scenario_id=scn))


def test_window_and_next_reset(tmp_path):
    tr, _ = make(tmp_path)
    assert tr.window() == "2026-09-25"
    nxt = tr.next_reset()
    assert nxt.tzinfo == KST and (nxt.hour, nxt.day) == (16, 26)
    # 겨울(PST)에는 한국 시간 17시에 초기화된다
    tr2, _ = make(tmp_path, start=datetime(2026, 12, 1, 1, 0, tzinfo=timezone.utc))
    assert tr2.next_reset().hour == 17


def test_stops_before_call_at_90_percent(tmp_path):
    tr, clock = make(tmp_path, rpm=1000, rpd=10)
    client = ok_client(tr)
    for i in range(9):
        call(client)
    assert tr.used_in_window() == 9 == tr.cap
    n_calls = len(client.calls)
    with pytest.raises(QuotaExhausted) as ei:
        call(client)
    assert len(client.calls) == n_calls  # 요청을 보내지 않았다
    assert ei.value.next_reset_kst.hour == 16
    assert "16:00" in str(ei.value)


def test_new_window_resets_count(tmp_path):
    tr, clock = make(tmp_path, rpm=1000, rpd=10)
    client = ok_client(tr)
    for _ in range(9):
        call(client)
    clock.now += timedelta(hours=16)  # 태평양 자정을 넘긴다
    assert tr.used_in_window() == 0
    call(client)


def test_usage_persists_across_trackers(tmp_path):
    tr, clock = make(tmp_path, rpm=1000, rpd=10)
    for _ in range(9):
        call(ok_client(tr))
    tr2 = UsageTracker(1000, 10, 0.9, "America/Los_Angeles", log_path=tr.log_path,
                       clock=clock, sleep=clock.sleep)
    with pytest.raises(QuotaExhausted):
        call(ok_client(tr2))


def test_min_interval_between_calls(tmp_path):
    tr, clock = make(tmp_path, rpm=15)
    client = ok_client(tr)
    call(client)
    call(client)
    call(client)
    assert clock.slept == [pytest.approx(4.0), pytest.approx(4.0)]


def test_log_record_fields(tmp_path):
    tr, _ = make(tmp_path)
    call(ok_client(tr), scn="scn_1007")
    rec = json.loads(tr.log_path.read_text().splitlines()[0])
    for key in ("ts", "window", "run_id", "scenario_id", "model", "input_tokens",
                "output_tokens", "status", "cost_usd"):
        assert key in rec
    assert rec["scenario_id"] == "scn_1007" and rec["cost_usd"] == 0.0 and rec["status"] == "ok"


def test_rate_limit_retries_with_backoff(tmp_path):
    tr, clock = make(tmp_path, rpm=1000)
    state = {"n": 0}

    def responder(s, p, schema):
        state["n"] += 1
        if state["n"] <= 2:
            raise FakeRateLimit("429")
        return '{"x": 2}'

    res = FakeClient(tr, responder).structured("s", "p", Out, 10, 0.0)
    assert res.data == {"x": 2}
    statuses = [json.loads(l)["status"] for l in tr.log_path.read_text().splitlines()]
    assert statuses == ["rate_limited", "rate_limited", "ok"]
    assert 2.0 in clock.slept and 4.0 in clock.slept


def test_rate_limit_gives_up_after_5_retries(tmp_path):
    tr, clock = make(tmp_path, rpm=1000)

    def responder(s, p, schema):
        raise FakeRateLimit("429")

    with pytest.raises(FakeRateLimit):
        FakeClient(tr, responder).structured("s", "p", Out, 10, 0.0)
    assert len(tr.log_path.read_text().splitlines()) == 6


def test_daily_quota_error_stops_without_retry(tmp_path):
    tr, clock = make(tmp_path, rpm=1000)
    calls = []

    def responder(s, p, schema):
        calls.append(1)
        raise FakeDailyQuota("429 PerDay")

    with pytest.raises(QuotaExhausted):
        FakeClient(tr, responder).structured("s", "p", Out, 10, 0.0)
    assert len(calls) == 1


def test_schema_error_is_raised_after_recording(tmp_path):
    tr, _ = make(tmp_path)
    with pytest.raises(SchemaError) as ei:
        FakeClient(tr, lambda s, p, schema: '{"x": "not int"}').structured("s", "p", Out, 10, 0.0)
    assert ei.value.raw == '{"x": "not int"}'
    assert tr.used_in_window() == 1


def test_tracker_requires_rate_limits(cfg):
    llm = cfg.llm.model_copy(update={"rpm_limit": None})
    with pytest.raises(ValueError):
        UsageTracker.from_config(llm)


def test_config_limits_are_used_as_written(cfg, tmp_path):
    tr = UsageTracker.from_config(cfg.llm, log_path=tmp_path / "u.jsonl")
    assert (tr.rpm_limit, tr.rpd_limit) == (cfg.llm.rpm_limit, cfg.llm.rpd_limit)
    assert tr.cap == int(cfg.llm.rpd_limit * cfg.llm.stop_at_ratio)
