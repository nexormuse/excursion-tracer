"""에이전트: 근거 묶음, 추가 확인, 실행 루프(가짜 클라이언트), 숫자 대조, 프롬프트."""

from __future__ import annotations

import builtins
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from excursion_tracer.agent.checks import run_check, run_checks
from excursion_tracer.agent.evidence import build_evidence
from excursion_tracer.agent.loop import find_run_id, make_run_id, run_scenario, run_set
from excursion_tracer.agent.numcheck import check_report, extract_numbers
from excursion_tracer.agent.prompts import SYSTEM_PROMPT, prompt_hash, system_prompt
from excursion_tracer.agent.schema import CheckRequest, Report
from excursion_tracer.llm.fake import FakeClient
from excursion_tracer.llm.usage import UsageTracker
from excursion_tracer.stats.data import load_scenario

from .conftest import generated_dev_dirs

EV = [{"evidence_id": "E7", "summary": "median diff -0.041"},
      {"evidence_id": "E18", "summary": "change point"}]


def hyp(**kw):
    base = {"rank": 1, "entity_type": "tool", "step_id": "S40", "tool_id": "S40-T1",
            "chamber_id": "", "recipe_id": "", "onset": "", "confidence": 0.7, "evidence": EV,
            "falsification_test": "x", "recommended_action": "y"}
    base.update(kw)
    return base


def report(**kw):
    base = {"scenario_id": "wrong", "method": "whatever", "verdict": "cause_found",
            "hypotheses": [hyp()], "no_cause_explanation": "", "confounding_notes": "",
            "limitations": ""}
    base.update(kw)
    return base


def tracker(tmp_path):
    return UsageTracker(1000, 10000, 0.9, "America/Los_Angeles", log_path=tmp_path / "u.jsonl",
                        clock=lambda: datetime(2026, 9, 26, 1, tzinfo=timezone.utc),
                        sleep=lambda s: None)


# ---------------------------------------------------------------- 근거 묶음

def test_evidence_within_limit_and_numbered(cfg, fixture_set):
    for d in fixture_set.dirs() + generated_dev_dirs():
        pack = build_evidence(load_scenario(d), cfg)
        assert len(pack) <= cfg.agent.evidence_max_chars, d.name
        ids = re.findall(r"^E(\d+) ", pack, re.M)
        assert ids and [int(i) for i in ids] == list(range(1, len(ids) + 1))


def test_evidence_does_not_touch_answer_files(cfg, fixture_set, monkeypatch):
    real_open, real_read = builtins.open, Path.read_text

    def check(p):
        if Path(str(p)).name in ("ground_truth.json", "meta.json"):
            raise AssertionError(f"읽으면 안 되는 파일: {p}")

    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (check(f), real_open(f, *a, **k))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (check(self), real_read(self, *a, **k))[1])
    for d in fixture_set.dirs()[:3]:
        build_evidence(load_scenario(d), cfg)


def test_evidence_has_no_answer_words(cfg, fixture_set):
    for d in fixture_set.dirs():
        pack = build_evidence(load_scenario(d), cfg).lower()
        for w in ("fault", "inject", "ground truth", "stickiness", "f0a", "f0b", "drift"):
            assert w not in pack


# ---------------------------------------------------------------- 추가 확인

def test_all_checks_run(cfg, fixture_set):
    d = load_scenario(fixture_set.dirs()[4])
    tool = d.history["tool_id"].iloc[0]
    step = tool.split("-")[0]
    chamber = d.history["chamber_id"].iloc[0]
    reqs = [
        CheckRequest(name="commonality_scan", level="chamber", reason="r"),
        CheckRequest(name="commonality_scan", level="tool", product="P2", reason="r"),
        CheckRequest(name="time_trend", entity_id=tool, reason="r"),
        CheckRequest(name="get_events", tool_id=tool, reason="r"),
        CheckRequest(name="get_events", step_id=step, reason="r"),
        CheckRequest(name="before_after", entity_id=chamber, event_ts="2026-01-10T00:00:00", reason="r"),
        CheckRequest(name="interaction_test", step_id=step, reason="r"),
        CheckRequest(name="confounding_check", entity_ids=[tool, chamber], reason="r"),
        CheckRequest(name="drilldown", entity_id=tool, by="chamber", reason="r"),
        CheckRequest(name="drilldown", entity_id=tool, by="product", reason="r"),
        CheckRequest(name="product_mix", reason="r"),
    ]
    for r in reqs:
        out = run_check(d, cfg, r)
        assert not out.startswith("error"), (r.name, out)
        assert len(out) <= cfg.agent.check_output_max_chars


@pytest.mark.parametrize("req", [
    CheckRequest(name="time_trend", entity_id="S99-T9", reason="r"),
    CheckRequest(name="time_trend", entity_id="nonsense", reason="r"),
    CheckRequest(name="before_after", entity_id="S01-T1", event_ts="", reason="r"),
    CheckRequest(name="confounding_check", entity_ids=["S01-T1"], reason="r"),
    CheckRequest(name="drilldown", entity_id="S01-T1", by="", reason="r"),
    CheckRequest(name="get_events", reason="r"),
])
def test_bad_check_returns_error_text(cfg, fixture_set, req):
    d = load_scenario(fixture_set.dirs()[0])
    assert run_check(d, cfg, req).startswith("error")


def test_checks_are_numbered_and_capped(cfg, fixture_set):
    d = load_scenario(fixture_set.dirs()[0])
    reqs = [CheckRequest(name="product_mix", reason="r")] * 5
    out = run_checks(d, cfg, reqs)
    assert [r["check_id"] for r in out] == ["C1", "C2", "C3"]


# ---------------------------------------------------------------- 실행 루프

def _responder(round1: dict, round2: dict | None = None, bad_first=0):
    state = {"n": 0}

    def respond(system, prompt, schema):
        state["n"] += 1
        if state["n"] <= bad_first:
            return '{"broken": true}'
        if schema.__name__ == "Round1":
            return json.dumps(round1)
        return json.dumps(round2)

    return respond


def test_round1_only(cfg, fixture_set, tmp_path):
    client = FakeClient(tracker(tmp_path), _responder({"report": report(), "needs_checks": False, "checks": []}))
    d = fixture_set.dirs()[4]
    rec = run_scenario(client, cfg, d, "run", tmp_path / "run")
    assert rec["requests"] == 1 and not rec["invalid"]
    assert rec["final_report"]["scenario_id"] == d.name
    assert rec["final_report"]["method"] == "agent:fake-model"
    assert (tmp_path / "run" / f"{d.name}.json").is_file()
    assert "evidence_pack" in rec and rec["round2"] is None


def test_round1_with_checks_then_round2(cfg, fixture_set, tmp_path):
    r1 = {"report": report(), "needs_checks": True, "checks": [
        {"name": "product_mix", "reason": "mix"},
        {"name": "time_trend", "entity_id": "S99-T9", "reason": "bad id"}]}
    r2 = report(verdict="no_equipment_cause", hypotheses=[], no_cause_explanation="product mix")
    client = FakeClient(tracker(tmp_path), _responder(r1, r2))
    rec = run_scenario(client, cfg, fixture_set.dirs()[4], "run", tmp_path / "run")
    assert rec["requests"] == 2 and rec["needs_checks"]
    assert [c["check_id"] for c in rec["checks"]] == ["C1", "C2"]
    assert rec["checks"][1]["result"].startswith("error")
    assert rec["final_report"]["verdict"] == "no_equipment_cause"
    assert rec["preliminary_report"]["verdict"] == "cause_found"
    prompt2 = client.calls[1]["prompt"]
    assert "Your preliminary report" in prompt2 and "C1 product_mix" in prompt2


def test_schema_retry_once_then_ok(cfg, fixture_set, tmp_path):
    client = FakeClient(tracker(tmp_path),
                        _responder({"report": report(), "needs_checks": False, "checks": []}, bad_first=1))
    rec = run_scenario(client, cfg, fixture_set.dirs()[4], "run", tmp_path / "run")
    assert rec["requests"] == 2 and not rec["invalid"]
    assert "did not match the required JSON schema" in client.calls[1]["prompt"]


def test_invalid_after_two_failures(cfg, fixture_set, tmp_path):
    client = FakeClient(tracker(tmp_path), _responder({}, bad_first=10))
    rec = run_scenario(client, cfg, fixture_set.dirs()[4], "run", tmp_path / "run")
    assert rec["invalid"] and rec["final_report"] is None and rec["requests"] == 2


def test_round2_failure_keeps_preliminary(cfg, fixture_set, tmp_path):
    r1 = {"report": report(), "needs_checks": True, "checks": [{"name": "product_mix", "reason": "r"}]}
    client = FakeClient(tracker(tmp_path), _responder(r1, {"bad": 1}))
    rec = run_scenario(client, cfg, fixture_set.dirs()[4], "run", tmp_path / "run")
    assert rec["round2_failed"] and not rec["invalid"]
    assert rec["final_report"] == rec["preliminary_report"] and rec["requests"] == 3


def test_run_set_skips_done(cfg, fixture_set, tmp_path):
    client = FakeClient(tracker(tmp_path), _responder({"report": report(), "needs_checks": False, "checks": []}))
    dirs = fixture_set.dirs()[:3]
    first = run_set(client, cfg, dirs, "run", tmp_path, limit=2, log=lambda s: None)
    second = run_set(client, cfg, dirs, "run", tmp_path, log=lambda s: None)
    assert len(first) == 2 and [r["scenario_id"] for r in second] == [dirs[2].name]
    assert len(client.calls) == 3


def test_usage_is_logged_per_call(cfg, fixture_set, tmp_path):
    tr = tracker(tmp_path)
    client = FakeClient(tr, _responder({"report": report(), "needs_checks": False, "checks": []}))
    run_scenario(client, cfg, fixture_set.dirs()[4], "run-x", tmp_path / "run")
    rec = json.loads(tr.log_path.read_text().splitlines()[-1])
    assert rec["run_id"] == "run-x" and rec["purpose"] == "round1"


def test_run_id_format_and_resume(tmp_path):
    rid = make_run_id("m")
    assert rid.endswith(f"-m-{prompt_hash()[:8]}")
    (tmp_path / f"2026-09-26-m-{prompt_hash()[:8]}").mkdir()
    assert find_run_id("m", tmp_path) == f"2026-09-26-m-{prompt_hash()[:8]}"


# ---------------------------------------------------------------- 숫자 대조

def test_extract_numbers_skips_ids_and_dates():
    text = "S40-T1 in scn_1001 (E7, C2) at 2026-01-13T13:30 had median diff -0.041 and OR 4.470, 21%"
    assert extract_numbers(text) == ["-0.041", "4.470", "21%"]


def test_extract_numbers_skips_written_dates():
    assert extract_numbers("around the onset time of January 14, then Jan 9th and 3 Feb 2026") == []
    assert extract_numbers("1월 14일 이후 0.041 하락") == ["0.041"]


def test_numcheck():
    rep = Report.model_validate(report(hypotheses=[hyp(evidence=[
        {"evidence_id": "E7", "summary": "median diff -0.041, rate 0.39 vs 11%, n 2850"},
        {"evidence_id": "E18", "summary": "p 2.57e-201, 3 lots"}])]))
    src = ["E7 median diff -0.041, low 0.386 vs 0.108, n 2850, p 2.57e-201"]
    assert check_report(rep, src) == {"unsupported": [], "bad": False}
    rep2 = Report.model_validate(report(hypotheses=[hyp(evidence=[
        {"evidence_id": "E7", "summary": "median diff -0.055"},
        {"evidence_id": "E18", "summary": "ok"}])]))
    assert check_report(rep2, src) == {"unsupported": ["-0.055"], "bad": True}


# ---------------------------------------------------------------- 프롬프트

def test_system_prompt_is_appendix_c():
    s = system_prompt("en")
    assert s.startswith("You are an investigation assistant for a semiconductor yield engineer.")
    assert "Write it in English." in s and "{report_language}" not in s
    assert "Korean" in system_prompt("ko")


def test_prompt_has_no_fault_specific_instruction():
    assert not re.search(r"\bF[0-5][ab]?\b", SYSTEM_PROMPT)
    for w in ("drift", "recipe version", "inject"):
        assert w not in SYSTEM_PROMPT.lower()
