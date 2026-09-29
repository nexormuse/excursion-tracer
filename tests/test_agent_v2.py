"""에이전트 v2: ID 검증·정규화, 근거 묶음 v2, 반증 확인 강제 흐름(가짜 클라이언트), 채점까지."""

from __future__ import annotations

import builtins
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from excursion_tracer.agent.evidence_v2 import build_evidence_v2
from excursion_tracer.agent.loop_v2 import find_run_id_v2, run_scenario_v2, run_set_v2
from excursion_tracer.agent.prompts_v2 import SYSTEM_PROMPT_V2, prompt_hash_v2, system_prompt_v2
from excursion_tracer.agent.prompts import prompt_hash
from excursion_tracer.agent.validate_ids import FabIndex, normalize_hypothesis, normalize_report
from excursion_tracer.eval.ground_truth import load_ground_truth
from excursion_tracer.eval.run_eval import evaluate
from excursion_tracer.llm.fake import FakeClient
from excursion_tracer.llm.usage import UsageTracker
from excursion_tracer.stats.data import load_scenario

EV = [{"evidence_id": "E12", "summary": "a"}, {"evidence_id": "E30", "summary": "b"}]


def tracker(tmp_path):
    return UsageTracker(1000, 100000, 0.9, "America/Los_Angeles", log_path=tmp_path / "u.jsonl",
                        clock=lambda: datetime(2026, 9, 26, 1, tzinfo=timezone.utc), sleep=lambda s: None)


def hyp(**kw):
    base = {"rank": 1, "entity_type": "tool", "step_id": "", "tool_id": "", "chamber_id": "",
            "recipe_id": "", "onset": "", "confidence": 0.7, "evidence": EV,
            "falsification_test": "x", "recommended_action": "y"}
    base.update(kw)
    return base


def report(hyps=None, verdict="cause_found", **kw):
    base = {"scenario_id": "x", "method": "m", "verdict": verdict, "hypotheses": hyps or [],
            "no_cause_explanation": "", "confounding_notes": "", "limitations": ""}
    base.update(kw)
    return base


def _fab(d):
    return json.loads((d / "fab.json").read_text())


def _multi_chamber(fab):
    for s in fab["steps"]:
        for t in s["tools"]:
            if len(t["chambers"]) > 1:
                return s["step_id"], t["tool_id"], t["chambers"]
    raise AssertionError


# ---------------------------------------------------------------- ID 검증

def test_normalize_short_ids(fixture_set):
    fab = _fab(fixture_set.dirs()[0])
    idx = FabIndex.from_fab(fab)
    step, tool, chambers = _multi_chamber(fab)
    short_t, short_c = tool.split("-")[1], chambers[1].split("-")[2]
    r = normalize_hypothesis(hyp(entity_type="chamber", step_id=step, tool_id=short_t, chamber_id=short_c), idx)
    assert r.ok and r.hypothesis["tool_id"] == tool and r.hypothesis["chamber_id"] == chambers[1]
    assert len(r.normalized) == 2
    r = normalize_hypothesis(hyp(entity_type="chamber", step_id="", chamber_id=chambers[0]), idx)
    assert r.ok and r.hypothesis["step_id"] == step and r.hypothesis["tool_id"] == tool
    spec = next(s for s in fab["steps"] if s["product_specific_recipe"])
    r = normalize_hypothesis(hyp(entity_type="tool_recipe", step_id=spec["step_id"],
                                 tool_id=spec["tools"][0]["tool_id"], recipe_id="R2"), idx)
    assert r.ok and r.hypothesis["recipe_id"] == f"{spec['step_id']}-R2"


def test_full_ids_are_untouched(fixture_set):
    fab = _fab(fixture_set.dirs()[0])
    step, tool, chambers = _multi_chamber(fab)
    r = normalize_hypothesis(hyp(entity_type="chamber", step_id=step, tool_id=tool, chamber_id=chambers[0]),
                             FabIndex.from_fab(fab))
    assert r.ok and r.normalized == []


def test_invalid_and_ambiguous_ids(fixture_set):
    fab = _fab(fixture_set.dirs()[0])
    idx = FabIndex.from_fab(fab)
    step, tool, chambers = _multi_chamber(fab)
    assert not normalize_hypothesis(hyp(step_id=step, tool_id="T99"), idx).ok
    assert not normalize_hypothesis(hyp(step_id="S99", tool_id=tool), idx).ok
    other = next(t["tool_id"] for s in fab["steps"] for t in s["tools"] if s["step_id"] != step)
    assert not normalize_hypothesis(hyp(step_id=step, tool_id=other), idx).ok  # 소속이 다름
    assert not normalize_hypothesis(hyp(entity_type="chamber", step_id=step, tool_id=tool), idx).ok  # 필수 비어 있음
    multi = [t for t in next(s for s in fab["steps"] if s["step_id"] == step)["tools"] if len(t["chambers"]) > 1]
    if len(multi) > 1:  # 설비 없이 C1만 쓰면 여러 설비로 해석된다
        r = normalize_hypothesis(hyp(entity_type="chamber", step_id=step, chamber_id="C1"), idx)
        assert not r.ok and "ambiguous" in r.problems[0]


# ---------------------------------------------------------------- 근거 묶음 v2

def test_evidence_v2_limit_sections_and_full_ids(cfg, fixture_set):
    for d in fixture_set.dirs():
        pack = build_evidence_v2(load_scenario(d), cfg)
        assert len(pack) <= cfg.v2.evidence_max_chars, d.name
        for sec in ("Product-adjusted change", "Normal spread before the alert", "Event scan",
                    "normal spread"):
            assert sec in pack
        ids = [int(i) for i in re.findall(r"^E(\d+) ", pack, re.M)]
        assert ids == list(range(1, len(ids) + 1))
        assert not re.search(r"(?<![\w-])(T\d+|C\d+)(?![\w-])", pack)  # 줄인 ID가 없다
        low = pack.lower()
        for w in ("fault", "inject", "ground truth", "stickiness", "drift", "intermittent"):
            assert w not in low


def test_evidence_v2_reads_no_answer_files(cfg, fixture_set, monkeypatch):
    real_open, real_read = builtins.open, Path.read_text

    def check(p):
        if Path(str(p)).name in ("ground_truth.json", "meta.json"):
            raise AssertionError(f"읽으면 안 되는 파일: {p}")

    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (check(f), real_open(f, *a, **k))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (check(self), real_read(self, *a, **k))[1])
    build_evidence_v2(load_scenario(fixture_set.dirs()[5]), cfg)


# ---------------------------------------------------------------- 흐름 (가짜 클라이언트)

class Script:
    """호출 순서대로 응답을 돌려준다. 문자열이면 그대로, dict면 JSON으로."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, system, prompt, schema):
        self.prompts.append((schema.__name__, prompt, system))
        r = self.responses.pop(0)
        return r if isinstance(r, str) else json.dumps(r)


def _answer_hyp(d, short=False):
    gt = load_ground_truth(d)
    f = gt["faults"][0]
    kind = {"F1": "chamber", "F2": "tool_recipe", "F3": "tool", "F5": "recipe", "F6": "chamber"}[f["type"]]
    h = hyp(entity_type=kind, step_id=f["step_id"], tool_id=f["tool_id"] or "",
            chamber_id=f["chamber_id"] or "", recipe_id=f["recipe_id"] or "", onset=f["onset_ts"])
    if short and h["chamber_id"]:
        h["tool_id"] = h["tool_id"].split("-")[1]
        h["chamber_id"] = h["chamber_id"].split("-")[2]
    return h


def _f1_dir(fixture_set):
    return next(d for d in fixture_set.dirs() if load_ground_truth(d)["fault_code"] == "F1")


def test_v2_flow_round1_checks_round2(cfg, fixture_set, tmp_path):
    d = _f1_dir(fixture_set)
    wrong = hyp(entity_type="tool", step_id=load_ground_truth(d)["faults"][0]["step_id"],
                tool_id=load_ground_truth(d)["faults"][0]["tool_id"])
    r1 = {"report": report([wrong]), "checks": [
        {"name": "drilldown", "entity_id": wrong["tool_id"], "by": "chamber", "reason": "which chamber"},
        {"name": "product_mix", "reason": "mix"}]}
    r2 = report([_answer_hyp(d)])
    script = Script(r1, r2)
    client = FakeClient(tracker(tmp_path), script)
    rec = run_scenario_v2(client, cfg, d, "run", tmp_path / "run")
    assert rec["requests"] == 2 and not rec["invalid"] and rec["needs_checks"]
    assert [c["check_id"] for c in rec["checks"]] == ["C1", "C2"]
    assert not rec["checks"][0]["result"].startswith("error")
    assert [s for s, _, _ in script.prompts] == ["Round1V2", "Report"]
    assert "Your preliminary report" in script.prompts[1][1] and "C1 drilldown" in script.prompts[1][1]
    assert script.prompts[0][2] == system_prompt_v2("en")
    assert "Event scan" in script.prompts[0][1]
    assert rec["final_report"]["method"] == "agent_v2:fake-model" and rec["prompt_hash"] == prompt_hash_v2()
    assert rec["preliminary_report"]["hypotheses"][0]["entity_type"] == "tool"
    assert rec["final_report"]["hypotheses"][0]["entity_type"] == "chamber"


def test_v2_round1_without_checks_is_rejected_then_invalid(cfg, fixture_set, tmp_path):
    d = _f1_dir(fixture_set)
    no_checks = {"report": report([_answer_hyp(d)]), "checks": []}
    client = FakeClient(tracker(tmp_path), Script(no_checks, no_checks))
    rec = run_scenario_v2(client, cfg, d, "run", tmp_path / "run")
    assert rec["invalid"] and rec["final_report"] is None and rec["requests"] == 2
    assert "did not match the required JSON schema" in rec["round1"]["attempts"][1]["prompt"]


def test_v2_short_ids_are_normalized_without_retry(cfg, fixture_set, tmp_path):
    d = _f1_dir(fixture_set)
    r1 = {"report": report([_answer_hyp(d, short=True)]), "checks": [{"name": "product_mix", "reason": "r"}]}
    client = FakeClient(tracker(tmp_path), Script(r1, report([_answer_hyp(d, short=True)])))
    rec = run_scenario_v2(client, cfg, d, "run", tmp_path / "run")
    f = load_ground_truth(d)["faults"][0]
    h = rec["final_report"]["hypotheses"][0]
    assert (h["tool_id"], h["chamber_id"]) == (f["tool_id"], f["chamber_id"])
    assert len(rec["id_normalized"]) == 2 and rec["id_problems"] == [] and rec["id_retry"] is None
    assert rec["requests"] == 2


def test_v2_unresolved_ids_trigger_one_retry(cfg, fixture_set, tmp_path):
    d = _f1_dir(fixture_set)
    bad = hyp(entity_type="tool", step_id="S01", tool_id="T99")
    r1 = {"report": report([bad]), "checks": [{"name": "product_mix", "reason": "r"}]}
    script = Script(r1, report([bad]), report([_answer_hyp(d)]))
    client = FakeClient(tracker(tmp_path), script)
    rec = run_scenario_v2(client, cfg, d, "run", tmp_path / "run")
    assert rec["requests"] == 3 and rec["id_retry"] is not None
    assert "T99" in script.prompts[2][1] and "Copy every ID exactly" in script.prompts[2][1]
    assert rec["id_problems"] == [] and rec["final_report"]["hypotheses"][0]["chamber_id"]
    assert rec["prelim_id_problems"]


def test_v2_round2_failure_keeps_preliminary(cfg, fixture_set, tmp_path):
    d = _f1_dir(fixture_set)
    r1 = {"report": report([_answer_hyp(d)]), "checks": [{"name": "product_mix", "reason": "r"}]}
    client = FakeClient(tracker(tmp_path), Script(r1, '{"bad": 1}', '{"bad": 2}'))
    rec = run_scenario_v2(client, cfg, d, "run", tmp_path / "run")
    assert rec["round2_failed"] and rec["final_report"] == rec["preliminary_report"] and rec["requests"] == 3


def test_v2_run_set_skips_done_and_run_id(cfg, fixture_set, tmp_path):
    dirs = fixture_set.dirs()[:2]
    r1 = {"report": report(verdict="no_equipment_cause", no_cause_explanation="product mix"),
          "checks": [{"name": "product_mix", "reason": "r"}]}
    r2 = report(verdict="no_equipment_cause", no_cause_explanation="product mix")
    client = FakeClient(tracker(tmp_path), Script(r1, r2, r1, r2))
    assert len(run_set_v2(client, cfg, dirs, "rv2", tmp_path, log=lambda s: None)) == 2
    assert run_set_v2(client, cfg, dirs, "rv2", tmp_path, log=lambda s: None) == []
    assert find_run_id_v2("m", tmp_path).endswith(f"-m-v2-{prompt_hash_v2()[:8]}")
    assert prompt_hash_v2() != prompt_hash()


def test_v2_records_are_scored(cfg, fixture_set, tmp_path):
    """1회차 오답 → 확인 → 2회차 정답(줄인 ID) 기록이 채점까지 이어지고, 추가 확인 효과와 ID 지표가 나온다."""
    data_root = tmp_path / "data"
    (data_root / "dev").mkdir(parents=True)
    run_dir = tmp_path / "runs" / "v2run"
    for d in fixture_set.dirs():
        gt = load_ground_truth(d)
        if gt["fault_code"] not in ("F1", "F0a"):
            continue
        (data_root / "dev" / d.name).symlink_to(d)
        if gt["fault_code"] == "F1":
            f = gt["faults"][0]
            first = report([hyp(entity_type="tool", step_id=f["step_id"], tool_id="S01-T1")])
            final = report([_answer_hyp(d, short=True)])
        else:
            first = report([hyp(entity_type="tool", step_id="S01", tool_id="S01-T1", confidence=0.9)])
            final = report(verdict="no_equipment_cause", no_cause_explanation="within normal spread")
        client = FakeClient(tracker(tmp_path), Script(
            {"report": first, "checks": [{"name": "product_mix", "reason": "r"}]}, final))
        run_scenario_v2(client, cfg, d, "v2run", run_dir)
    out = evaluate(cfg, "dev", data_root, {"agent_v2": run_dir}, tmp_path / "results", chance_draws=10)["dev"]
    a = out["agent_v2"]
    assert a["hit3_strict"]["value"] == 1.0 and a["false_alarm"]["value"] == 0.0
    ce = a["check_effect"]
    assert ce["requested"]["value"] == 1.0 and ce["top_changed_given_requested"]["value"] == 1.0
    assert ce["changed_hit3_strict_prelim"]["value"] == 0.0 and ce["changed_hit3_strict_final"]["value"] == 1.0
    assert ce["changed_false_alarm_prelim"]["value"] == 1.0 and ce["changed_false_alarm_final"]["value"] == 0.0
    assert a["id_check"]["normalized_scenarios"]["k"] >= 1


# ---------------------------------------------------------------- 프롬프트

def test_system_prompt_v2_is_spec_and_general():
    s = system_prompt_v2("en")
    assert s.startswith("You are an investigation assistant for a semiconductor yield engineer.")
    assert "Copy IDs exactly and in full" in s and "request 1–3 checks" in s
    assert "{report_language}" not in s and "Write it in English." in s
    assert not re.search(r"\bF[0-9][ab]?\b", SYSTEM_PROMPT_V2)
    for w in ("intermittent", "drift", "inject", "recipe version"):
        assert w not in SYSTEM_PROMPT_V2.lower()
