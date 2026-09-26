"""평가기: 대조 규칙, 지표, 신뢰구간, 쌍대 비교, 반복성, 우연 수준."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from excursion_tracer.agent.schema import Evidence, Hypothesis, Report
from excursion_tracer.eval.compare import mcnemar_paired
from excursion_tracer.eval.ground_truth import load_ground_truth
from excursion_tracer.eval.match import match_fault, score
from excursion_tracer.eval.metrics import (
    calibration, chance_hit3, core_metrics, rate, repeatability, wilson,
)
from excursion_tracer.eval.run_eval import evaluate, load_report

EV = [Evidence(evidence_id="E1", summary="a"), Evidence(evidence_id="E2", summary="b")]


def H(rank=1, kind="tool", step="S12", tool="", chamber="", recipe="", conf=0.8, onset=""):
    return Hypothesis(rank=rank, entity_type=kind, step_id=step, tool_id=tool, chamber_id=chamber,
                      recipe_id=recipe, confidence=conf, onset=onset, evidence=EV,
                      falsification_test="x", recommended_action="y")


def R(*hyps, verdict="cause_found", expl=""):
    return Report(scenario_id="scn_1", method="m", verdict=verdict, hypotheses=list(hyps),
                  no_cause_explanation=expl)


F1 = {"type": "F1", "step_id": "S12", "tool_id": "S12-T3", "chamber_id": "S12-T3-C2",
      "recipe_id": None, "onset_ts": "2026-01-10T06:00:00", "delta": 0.07}
F2 = {"type": "F2", "step_id": "S20", "tool_id": "S20-T1", "chamber_id": None,
      "recipe_id": "S20-R2", "onset_ts": "2026-01-11T00:00:00", "delta": 0.07}
F3 = {"type": "F3", "step_id": "S30", "tool_id": "S30-T2", "chamber_id": None,
      "recipe_id": None, "onset_ts": "2026-01-09T00:00:00", "delta": 0.07}
F5 = {"type": "F5", "step_id": "S07", "tool_id": None, "chamber_id": None,
      "recipe_id": "S07-R1", "onset_ts": "2026-01-12T00:00:00", "delta": 0.07}


def GT(code, *faults):
    return {"scenario_id": "scn_1", "fault_code": code, "faults": list(faults),
            "cell": {"effect": "medium", "stickiness": "low"}, "regen_count": 0}


# ---------------------------------------------------------------- 대조 규칙

@pytest.mark.parametrize("h,fault,expected", [
    (H(kind="chamber", tool="S12-T3", chamber="S12-T3-C2"), F1, (True, True)),
    (H(kind="chamber", chamber="S12-T3-C2"), F1, (True, True)),
    (H(kind="tool", tool="S12-T3"), F1, (False, True)),
    (H(kind="chamber", tool="S12-T3", chamber="S12-T3-C1"), F1, (False, True)),
    (H(kind="tool", tool="S12-T1"), F1, (False, False)),
    (H(kind="tool_recipe", step="S20", tool="S20-T1", recipe="S20-R2"), F2, (True, True)),
    (H(kind="tool", step="S20", tool="S20-T1"), F2, (False, True)),
    (H(kind="recipe", step="S20", recipe="S20-R2"), F2, (False, True)),
    (H(kind="tool_recipe", step="S20", tool="S20-T2", recipe="S20-R1"), F2, (False, False)),
    (H(kind="tool", step="S30", tool="S30-T2"), F3, (True, True)),
    (H(kind="tool", step="S30", tool="S30-T1"), F3, (False, True)),
    (H(kind="tool", step="S31", tool="S31-T2"), F3, (False, False)),
    (H(kind="recipe", step="S07", recipe="S07-R1"), F5, (True, True)),
    (H(kind="tool", step="S07", tool="S07-T1"), F5, (False, True)),
])
def test_match_rules(h, fault, expected):
    assert match_fault(h, fault) == expected


def test_single_fault_ranks():
    s = score(R(H(1, tool="S12-T1"), H(2, tool="S12-T3"),
                H(3, kind="chamber", chamber="S12-T3-C2")), GT("F1", F1), tau=0.5)
    assert (s["rank_strict"], s["rank_loose"]) == (3, 2)
    assert s["hit3_strict"] and not s["hit1_strict"] and s["hit3_loose"] and not s["hit1_loose"]


def test_f4_needs_both_for_strict():
    gt = GT("F4", F1, F3)
    both = score(R(H(1, step="S30", tool="S30-T2"), H(2, kind="chamber", chamber="S12-T3-C2")),
                 gt, tau=0.5)
    assert both["hit3_strict"] and both["hit1_strict"] and both["rank_strict"] == 2
    one = score(R(H(1, tool="S12-T3")), gt, tau=0.5)
    assert not one["hit3_strict"] and one["hit3_loose"] and one["hit1_loose"]
    assert one["rank_strict"] is None and one["rank_loose"] == 1


def test_no_cause_verdict_is_miss_for_cause_scenario():
    s = score(R(H(kind="chamber", chamber="S12-T3-C2"), verdict="no_equipment_cause"),
              GT("F1", F1), tau=0.5)
    assert s["missed"] and not s["hit3_strict"]


def test_f0_rules():
    gt = GT("F0a")
    assert score(R(verdict="no_equipment_cause"), gt, 0.5)["false_alarm"] is False
    assert score(R(H(conf=0.4)), gt, 0.5)["false_alarm"] is False
    assert score(R(H(conf=0.5)), gt, 0.5)["false_alarm"] is True
    assert score(None, gt, 0.5)["false_alarm"] is True


def test_f0b_explanation():
    gt = GT("F0b")
    assert score(R(verdict="no_equipment_cause", expl="Product mix shifted to P2"), gt, 0.5)["f0b_explained"]
    assert not score(R(verdict="no_equipment_cause", expl="random fluctuation"), gt, 0.5)["f0b_explained"]


def test_onset_accuracy():
    gt = GT("F1", F1)
    ok = score(R(H(kind="chamber", chamber="S12-T3-C2", onset="2026-01-10T20:00:00")), gt, 0.5)
    bad = score(R(H(kind="chamber", chamber="S12-T3-C2", onset="2026-01-12T00:00:00")), gt, 0.5)
    none = score(R(H(tool="S12-T3")), gt, 0.5)
    assert ok["onset_ok"] is True and bad["onset_ok"] is False and none["onset_ok"] is None


def test_invalid_report_is_wrong():
    s = score(None, GT("F3", F3), tau=0.5)
    assert s["invalid"] and s["missed"] and not s["hit3_strict"]


# ---------------------------------------------------------------- 지표

def test_wilson_known_values():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(0.1611, abs=1e-4)


def test_rate_ignores_none():
    r = rate([True, False, None, True])
    assert (r["k"], r["n"]) == (2, 3)


def test_mcnemar():
    a = pd.Series([True] * 10 + [False] * 2)
    b = pd.Series([False] * 10 + [True] * 2)
    res = mcnemar_paired(a, b)
    assert (res["only_a"], res["only_b"]) == (10, 2)
    assert res["p"] == pytest.approx(0.03857, abs=1e-4)
    assert mcnemar_paired(a, a)["p"] == 1.0


def test_repeatability():
    same = [R(H(tool="S12-T3")), R(H(tool="S12-T3", conf=0.3))]
    diff = [R(H(tool="S12-T3")), R(H(tool="S12-T1"))]
    r = repeatability({"a": same, "b": diff, "c": [R(verdict="no_equipment_cause")] * 3})
    assert (r["k"], r["n"]) == (2, 3)


def test_core_metrics_and_calibration():
    rows = []
    for code, rep, gt in [
        ("F1", R(H(kind="chamber", chamber="S12-T3-C2", conf=0.9)), GT("F1", F1)),
        ("F3", R(H(step="S30", tool="S30-T1", conf=0.3)), GT("F3", F3)),
        ("F0a", R(H(conf=0.9)), GT("F0a")),
        ("F0b", R(verdict="no_equipment_cause"), GT("F0b")),
    ]:
        rows.append({"fault_code": code, **score(rep, gt, 0.5)})
    df = pd.DataFrame(rows)
    m = core_metrics(df)
    assert (m["hit3_strict"]["k"], m["hit3_strict"]["n"]) == (1, 2)
    assert (m["false_alarm"]["k"], m["false_alarm"]["n"]) == (1, 2)
    assert m["mrr_strict"]["value"] == pytest.approx(0.5)
    cal = calibration(df)
    assert len(cal) == 5 and cal[4]["n"] == 1 and cal[1]["n"] == 1


def test_chance_level_is_small_and_deterministic(fixture_set):
    d = next(d for d in fixture_set.dirs() if load_ground_truth(d)["fault_code"] == "F3")
    fab = json.loads((d / "fab.json").read_text())
    gt = load_ground_truth(d)
    a = chance_hit3(fab, gt, n_draws=500)
    assert a == chance_hit3(fab, gt, n_draws=500)
    n_tools = sum(len(s["tools"]) for s in fab["steps"])
    assert a["pool_size"] == n_tools
    assert a["hit3_strict"] == pytest.approx(3 / n_tools, abs=0.02)


def test_load_report(tmp_path):
    good = R(H(tool="S12-T3"))
    (tmp_path / "a.json").write_text(good.model_dump_json())
    (tmp_path / "b.json").write_text('{"scenario_id": "x"}')
    (tmp_path / "c.json").write_text(json.dumps({"final_report": json.loads(good.model_dump_json()),
                                                "requests": 2}))
    (tmp_path / "d.json").write_text(json.dumps({"final_report": None, "invalid": True}))
    assert load_report(tmp_path / "a.json")[0] == good
    assert load_report(tmp_path / "b.json")[0] is None
    rep, extra = load_report(tmp_path / "c.json")
    assert rep == good and extra["requests"] == 2
    assert load_report(tmp_path / "d.json")[0] is None
    assert load_report(tmp_path / "missing.json")[0] is None


def test_evaluate_end_to_end(cfg, fixture_set, tmp_path):
    """정답을 그대로 옮긴 보고서는 원인 시나리오에서 모두 엄격 적중이고, F0에서 오경보가 없다."""
    data_root = tmp_path / "data"
    (data_root / "dev").mkdir(parents=True)
    rep_dir = tmp_path / "oracle"
    rep_dir.mkdir()
    for d in fixture_set.dirs():
        (data_root / "dev" / d.name).symlink_to(d)
        gt = load_ground_truth(d)
        hyps = []
        for i, f in enumerate(gt["faults"]):
            kind = {"F1": "chamber", "F2": "tool_recipe", "F3": "tool", "F5": "recipe"}[f["type"]]
            hyps.append(H(i + 1, kind=kind, step=f["step_id"], tool=f["tool_id"] or "",
                          chamber=f["chamber_id"] or "", recipe=f["recipe_id"] or "",
                          onset=f["onset_ts"]))
        rep = R(*hyps) if hyps else R(verdict="no_equipment_cause", expl="product mix change")
        (rep_dir / f"{d.name}.json").write_text(rep.model_copy(update={"scenario_id": d.name})
                                                .model_dump_json())
    out = evaluate(cfg, "dev", data_root, {"oracle": rep_dir, "missing": tmp_path / "none"},
                   tmp_path / "results", chance_draws=200)["dev"]
    assert out["oracle"]["hit3_strict"]["value"] == 1.0
    assert out["oracle"]["hit1_strict"]["value"] == 1.0
    assert out["oracle"]["false_alarm"]["value"] == 0.0
    assert out["oracle"]["onset_accuracy"]["value"] == 1.0
    assert out["missing"]["invalid_rate"]["value"] == 1.0
    assert out["compare"]["oracle_vs_missing"]["hit3_strict_mcnemar"]["only_a"] > 0
    per = pd.read_csv(tmp_path / "results" / "per_scenario.csv")
    assert len(per) == 2 * len(fixture_set.dirs())
    assert 0 < out["chance"]["hit3_strict"] < 0.2
