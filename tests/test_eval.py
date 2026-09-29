"""평가기: 대조 규칙, 지표, 신뢰구간, 쌍대 비교, 반복성, 우연 수준."""

from __future__ import annotations

import json
from pathlib import Path

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
    (H(kind="chamber", tool="S12-T3", chamber="S12-T3-C2"), {**F1, "type": "F6"}, (True, True)),
    (H(kind="tool", tool="S12-T3"), {**F1, "type": "F6"}, (False, True)),
    (H(kind="tool", step="S07", tool="S07-T1"), F5, (False, True)),
])
def test_match_rules(h, fault, expected):
    assert match_fault(h, fault) == expected


def test_single_fault_ranks():
    s = score(R(H(1, tool="S12-T1"), H(2, tool="S12-T3"),
                H(3, kind="chamber", chamber="S12-T3-C2")), GT("F1", F1), tau=0.5)
    assert (s["rank_strict"], s["rank_loose"]) == (3, 2)
    assert s["hit3_strict"] and not s["hit1_strict"] and s["hit3_loose"] and not s["hit1_loose"]


def test_f4_hit_is_any_fault_and_both_is_separate():
    gt = GT("F4", F1, F3)
    both = score(R(H(1, step="S30", tool="S30-T2"), H(2, kind="chamber", chamber="S12-T3-C2")),
                 gt, tau=0.5)
    assert both["hit3_strict"] and both["hit1_strict"] and both["rank_strict"] == 1
    assert both["both_hit3_strict"] is True
    one = score(R(H(1, tool="S12-T1"), H(2, kind="chamber", chamber="S12-T3-C2")), gt, tau=0.5)
    assert one["hit3_strict"] and not one["hit1_strict"] and one["rank_strict"] == 2
    assert one["both_hit3_strict"] is False
    loose = score(R(H(1, tool="S12-T3")), gt, tau=0.5)
    assert not loose["hit3_strict"] and loose["hit3_loose"] and loose["hit1_loose"]
    assert loose["rank_strict"] is None and loose["rank_loose"] == 1


def test_hit1_never_exceeds_hit3():
    import itertools

    gt = GT("F4", F1, F3)
    pool = [H(tool="S12-T1"), H(kind="chamber", chamber="S12-T3-C2"), H(step="S30", tool="S30-T2"),
            H(tool="S12-T3")]
    for combo in itertools.permutations(pool, 3):
        hyps = [h.model_copy(update={"rank": i + 1}) for i, h in enumerate(combo)]
        s = score(R(*hyps), gt, tau=0.5)
        assert s["hit1_strict"] <= s["hit3_strict"] and s["hit1_loose"] <= s["hit3_loose"]


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
            kind = {"F1": "chamber", "F2": "tool_recipe", "F3": "tool", "F5": "recipe",
                    "F6": "chamber"}[f["type"]]
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


def test_evaluate_repeatability(tmp_path):
    from excursion_tracer.eval.run_eval import evaluate_repeatability

    dirs = [tmp_path / f"r{i}" for i in range(3)]
    for d in dirs:
        d.mkdir()
    same = R(H(tool="S12-T3"))
    for d in dirs:
        (d / "scn_a.json").write_text(same.model_dump_json())
    (dirs[0] / "scn_b.json").write_text(R(H(tool="S12-T1")).model_dump_json())
    (dirs[1] / "scn_b.json").write_text(R(H(tool="S12-T2")).model_dump_json())
    (dirs[2] / "scn_b.json").write_text(R(H(tool="S12-T1")).model_dump_json())
    res = evaluate_repeatability("test", "agent", dirs, ["scn_a", "scn_b", "scn_c"], tmp_path)
    assert (res["k"], res["n"]) == (1, 2) and res["missing"] == ["scn_c"]
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["test"]["agent"]["repeatability"]["k"] == 1


def test_stratified_sample_is_proportional_and_fixed():
    import sys
    from collections import Counter

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_extra import stratified_sample

    kinds = ["F0a"] * 12 + ["F0b"] * 12 + ["F1"] * 24 + ["F2"] * 18 + ["F3"] * 18 + ["F4"] * 18 + ["F5"] * 18
    codes = {f"scn_{5001 + i}": c for i, c in enumerate(kinds)}
    s = stratified_sample(codes, 20, 5000)
    assert s == stratified_sample(codes, 20, 5000)
    assert Counter(codes[x] for x in s) == {"F1": 4, "F2": 3, "F3": 3, "F4": 3, "F5": 3, "F0a": 2, "F0b": 2}


def test_check_effect_metric(cfg):
    from excursion_tracer.eval.metrics import check_effect
    from excursion_tracer.eval.run_eval import _check_effect

    gt = GT("F1", F1)
    prelim = R(H(tool="S12-T1"))
    final = R(H(kind="chamber", chamber="S12-T3-C2"))
    e = _check_effect({"needs_checks": True, "preliminary_report": json.loads(prelim.model_dump_json())},
                      final, gt, cfg)
    assert e == {"checks_requested": True, "top_changed": True,
                 "prelim_hit3_strict": False, "prelim_false_alarm": None}
    none = _check_effect({"needs_checks": False, "preliminary_report": None}, final, gt, cfg)
    assert none["checks_requested"] is False and none["top_changed"] is None
    rows = [{"fault_code": "F1", **score(final, gt, 0.5), **e},
            {"fault_code": "F1", **score(final, gt, 0.5), **none}]
    ce = check_effect(pd.DataFrame(rows))
    assert (ce["requested"]["k"], ce["requested"]["n"]) == (1, 2)
    assert ce["changed_hit3_strict_prelim"]["k"] == 0 and ce["changed_hit3_strict_final"]["k"] == 1


def test_evaluate_replaces_set_and_keeps_repeatability(cfg, fixture_set, tmp_path):
    from excursion_tracer.eval.run_eval import evaluate_repeatability

    data_root = tmp_path / "data"
    (data_root / "dev").mkdir(parents=True)
    rep_dir = tmp_path / "reps"
    rep_dir.mkdir()
    for d in fixture_set.dirs()[:4]:
        (data_root / "dev" / d.name).symlink_to(d)
        (rep_dir / f"{d.name}.json").write_text(
            R(verdict="no_equipment_cause").model_copy(update={"scenario_id": d.name}).model_dump_json())
    res = tmp_path / "results"
    evaluate(cfg, "dev", data_root, {"old": rep_dir, "agent": rep_dir}, res, chance_draws=10)
    evaluate_repeatability("dev", "agent", [rep_dir, rep_dir], [d.name for d in fixture_set.dirs()[:4]], res)
    out = evaluate(cfg, "dev", data_root, {"agent": rep_dir, "baseline": rep_dir}, res, chance_draws=10)["dev"]
    assert "old" not in out and "repeatability" in out["agent"]
    assert "mcnemar_p" in out["compare"]
    per = pd.read_csv(res / "per_scenario.csv")
    assert set(per["method"]) == {"agent", "baseline"}
