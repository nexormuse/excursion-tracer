"""통계 엔진 단위 테스트: 알려진 효과를 심은 작은 합성 데이터로 확인한다."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu, rankdata
from statsmodels.stats.multitest import multipletests

from excursion_tracer.agent.schema import Report
from excursion_tracer.stats.baseline import run_baseline
from excursion_tracer.stats.commonality import _tie_term, commonality_scan, mwu_less, scan_all
from excursion_tracer.stats.confounding import confounding_check, phi
from excursion_tracer.stats.data import ScenarioData, load_scenario, stouffer
from excursion_tracer.stats.interaction import interaction_test
from excursion_tracer.stats.timewin import before_after, change_point, time_trend

T0 = pd.Timestamp("2026-01-01")
BASE = {"P1": 0.92, "P2": 0.88}
STEPS = {
    # 단계: (설비 수, 설비당 챔버 수, 제품별 레시피 여부)
    "S01": (3, 1, False),
    "S02": (2, 2, False),
    "S03": (2, 1, True),
}


def _fab() -> dict:
    steps = []
    for sid, (nt, nc, spec) in STEPS.items():
        steps.append({
            "step_id": sid, "area": "etch",
            "recipes": [f"{sid}-R1", f"{sid}-R2"] if spec else [f"{sid}-R1"],
            "product_specific_recipe": spec,
            "tools": [{"tool_id": f"{sid}-T{t + 1}",
                       "chambers": [f"{sid}-T{t + 1}-C{c + 1}" for c in range(nc)]}
                      for t in range(nt)],
        })
    return {"n_steps": len(steps), "products": ["P1", "P2"],
            "product_recipe": {"P1": "R1", "P2": "R2"}, "steps": steps}


def make_data(effect=None, n_lots=240, wafers=10, seed=0, tool_of=None, p2_prob=None,
              lot_sd=0.01, wafer_sd=0.01, p2_by_lot=None, events=None, window_days=None) -> ScenarioData:
    """effect(lot, slot, route, t_hours, product) -> 수율 감소량.

    tool_of(step, lot, route, rng) 로 설비 배정을 바꿀 수 있다 (기본은 무작위).
    p2_prob(route) 로 설비 경로에 따라, p2_by_lot(i) 로 로트 순번에 따라 제품 비중을 바꿀 수 있다.
    events 는 events.parquet 형식의 행 목록, window_days=(시작일, 끝일) 은 알림 구간이다.
    """
    rng = np.random.default_rng(seed)
    hist, waf = [], []
    for i in range(n_lots):
        lot = f"L{i + 1:04d}"
        t = i * 2.0
        route = {}
        for sid, (nt, _, _) in STEPS.items():
            route[sid] = tool_of(sid, i, route, rng) if tool_of else int(rng.integers(nt))
        prob = p2_prob(route) if p2_prob else (p2_by_lot(i) if p2_by_lot else 0.4)
        product = "P2" if rng.random() < prob else "P1"
        lot_eff = rng.normal(0, lot_sd)
        for s in range(wafers):
            wid = f"{lot}-W{s + 1:02d}"
            y = BASE[product] + lot_eff + rng.normal(0, wafer_sd)
            for k, (sid, (_, nc, spec)) in enumerate(STEPS.items()):
                tool = f"{sid}-T{route[sid] + 1}"
                hist.append({
                    "lot_id": lot, "wafer_id": wid, "slot": s + 1, "step_id": sid,
                    "tool_id": tool, "chamber_id": f"{tool}-C{s % nc + 1}",
                    "recipe_id": f"{sid}-R{2 if spec and product == 'P2' else 1}",
                    "track_in_ts": T0 + pd.Timedelta(hours=t + k),
                    "track_out_ts": T0 + pd.Timedelta(hours=t + k + 0.5),
                })
            if effect is not None:
                y -= effect(i, s, route, t, product)
            waf.append({"wafer_id": wid, "lot_id": lot, "product": product,
                        "yield": float(np.clip(y, 0, 1)),
                        "test_ts": T0 + pd.Timedelta(hours=t + 24)})
    wafers_df = pd.DataFrame(waf)
    thr = float(np.quantile(wafers_df["yield"], 0.1))
    ws, we = window_days or (0, 60)
    alert = {
        "level": "alarm",
        "window_start": (T0 + pd.Timedelta(days=ws)).isoformat(),
        "window_end": (T0 + pd.Timedelta(days=we)).isoformat(),
        "baseline": {"low_yield_threshold": thr, "low_rate": 0.1},
        "by_product": {},
    }
    ev_cols = ["event_id", "event_type", "step_id", "tool_id", "chamber_id", "recipe_id", "ts"]
    return ScenarioData("scn_test", _fab(), alert, pd.DataFrame(hist), wafers_df,
                        pd.DataFrame(events or [], columns=ev_cols), pd.DataFrame())


# ---------------------------------------------------------------- 기본 함수

def test_mwu_matches_scipy():
    rng = np.random.default_rng(1)
    x = np.round(rng.normal(size=300), 1)  # 동점 포함
    m = rng.random(300) < 0.3
    expected = mannwhitneyu(x[m], x[~m], alternative="less", method="asymptotic").pvalue
    assert mwu_less(rankdata(x), m, _tie_term(x)) == pytest.approx(expected, rel=1e-9)


def test_stouffer():
    assert stouffer([0.03], [1.0]) == pytest.approx(0.03)
    assert stouffer([0.05, 0.05], [1, 1]) < 0.05
    assert stouffer([0.05, 0.95], [1, 1]) == pytest.approx(0.5)


def test_phi():
    a = np.array([1, 1, 0, 0], bool)
    assert phi(a, a) == pytest.approx(1.0)
    assert phi(a, ~a) == pytest.approx(-1.0)


# ---------------------------------------------------------------- 공통성 스캔

def test_scan_finds_planted_tool():
    d = make_data(lambda i, s, r, t, p: 0.03 if r["S01"] == 1 else 0.0)
    top = scan_all(d).iloc[0]
    assert top["entity_id"] == "S01-T2" and top["q"] < 1e-6
    assert top["median_diff"] < -0.02


def test_scan_finds_planted_chamber():
    d = make_data(lambda i, s, r, t, p: 0.04 if r["S02"] == 0 and s % 2 == 1 else 0.0)
    top = commonality_scan(d, "chamber", top_k=3).iloc[0]
    assert top["entity_id"] == "S02-T1-C2" and top["q"] < 1e-6


def test_scan_null_has_no_significant_candidate():
    # 웨이퍼가 서로 독립인 경우 (로트 효과 없음)
    d = make_data(None, seed=3, lot_sd=0.0)
    assert scan_all(d)["q"].min() > 0.01


def test_q_is_bh_over_all_levels():
    d = make_data(lambda i, s, r, t, p: 0.02 if r["S01"] == 0 else 0.0)
    df = scan_all(d)
    assert set(df["level"]) >= {"tool", "chamber"}
    q = multipletests(df["p"].to_numpy(), method="fdr_bh")[1]
    np.testing.assert_allclose(df["q"].to_numpy(), q)
    assert (df["q"].diff().dropna() >= -1e-15).all()


def test_stratification_removes_product_mix_effect():
    # S01-T3로 간 로트는 수율이 낮은 P2가 많지만 설비 효과는 없다.
    d = make_data(None, seed=5, p2_prob=lambda r: 0.9 if r["S01"] == 2 else 0.2)
    strat = scan_all(d)
    raw = scan_all(d, stratify=False)
    assert raw.set_index("entity_id").loc["S01-T3", "q"] < 1e-6
    assert strat.set_index("entity_id").loc["S01-T3", "q"] > 0.01


def test_product_filter():
    d = make_data(lambda i, s, r, t, p: 0.03 if r["S01"] == 1 and p == "P2" else 0.0)
    assert commonality_scan(d, "tool", product="P2").iloc[0]["entity_id"] == "S01-T2"


# ---------------------------------------------------------------- 시간 분석

ONSET_H = 300.0


def _late_effect(i, s, r, t, p):
    return 0.03 if r["S01"] == 1 and t >= ONSET_H else 0.0


def test_change_point_finds_onset():
    d = make_data(_late_effect)
    cp = change_point(d, "S01-T2", n_perm=200)
    found = pd.Timestamp(cp["ts"])
    assert abs((found - (T0 + pd.Timedelta(hours=ONSET_H))) / pd.Timedelta(hours=1)) <= 12
    assert cp["p"] < 0.01 and cp["mean_after"] < cp["mean_before"]


def test_change_point_null():
    d = make_data(None, seed=7)
    assert change_point(d, "S01-T2", n_perm=200)["p"] > 0.05


def test_time_trend_shape():
    d = make_data(_late_effect)
    tt = time_trend(d, "S01-T2")
    assert list(tt.columns) == ["day", "mean_in", "mean_out", "n_in", "n_out", "diff"]
    late = tt[tt["day"] >= T0 + pd.Timedelta(hours=ONSET_H + 24)]
    assert (late["diff"] < -0.015).mean() > 0.8


def test_before_after():
    d = make_data(_late_effect)
    ev = T0 + pd.Timedelta(hours=ONSET_H)
    assert before_after(d, "S01-T2", ev, days=3)["p"] < 1e-3
    assert before_after(d, "S01-T1", ev, days=3)["p"] > 0.01


# ---------------------------------------------------------------- 상호작용·교란

def test_interaction_finds_tool_recipe_cell():
    d = make_data(lambda i, s, r, t, p: 0.03 if r["S03"] == 0 and p == "P2" else 0.0)
    res = interaction_test(d, "S03")
    assert res["p_interaction"] < 1e-3
    assert (res["worst_cell"]["tool_id"], res["worst_cell"]["recipe_id"]) == ("S03-T1", "S03-R2")


def test_interaction_single_recipe_step():
    assert np.isnan(interaction_test(make_data(None), "S01")["p_interaction"])


def test_confounding_points_to_true_entity():
    # S02 설비는 대개 S01 설비를 따라간다 (S01-T1 → S02-T1). 원인은 S01-T1.
    def tool_of(step, i, route, rng):
        if step == "S02":
            if route["S01"] == 0:
                return 0 if rng.random() < 0.8 else 1
            return 1 if rng.random() < 0.8 else 0
        return int(rng.integers(STEPS[step][0]))

    d = make_data(lambda i, s, r, t, p: 0.03 if r["S01"] == 0 else 0.0, tool_of=tool_of)
    res = confounding_check(d, ["S01-T1", "S02-T1"])
    pair = res["pairs"][0]
    assert pair["phi"] > 0.4
    assert pair["a_only"]["mean_yield"] < pair["b_only"]["mean_yield"] - 0.02


# ---------------------------------------------------------------- 기준선

def test_baseline_reports_planted_cause(cfg):
    d = make_data(_late_effect)
    rep = run_baseline(d, cfg)
    assert rep.verdict == "cause_found" and rep.method == "baseline"
    h = rep.hypotheses[0]
    assert (h.entity_type, h.tool_id, h.step_id) == ("tool", "S01-T2", "S01")
    assert h.onset and 0 <= h.confidence <= 1 and len(h.evidence) >= 2
    Report.model_validate_json(rep.model_dump_json())


def test_baseline_null_is_no_cause(cfg):
    rep = run_baseline(make_data(None, seed=3, lot_sd=0.0), cfg)
    assert rep.verdict == "no_equipment_cause" and rep.hypotheses == []
    assert rep.no_cause_explanation


def test_baseline_reads_no_ground_truth_or_meta(cfg, fixture_set, monkeypatch):
    real_open = builtins.open
    real_read = Path.read_text

    def check(path):
        name = Path(str(path)).name
        if name in ("ground_truth.json", "meta.json"):
            raise AssertionError(f"읽으면 안 되는 파일: {path}")

    def guarded_open(file, *a, **k):
        check(file)
        return real_open(file, *a, **k)

    def guarded_read(self, *a, **k):
        check(self)
        return real_read(self, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read)
    d = fixture_set.dirs()[4]
    rep = run_baseline(load_scenario(d), cfg)
    assert rep.scenario_id == d.name
    json.loads(rep.model_dump_json())
