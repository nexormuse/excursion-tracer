"""통계 v2 단위 테스트: 평소 차이·제품 보정, 이벤트 전후 스캔, 설비×레시피 판정, 기준선 v2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from excursion_tracer.agent.schema import Report
from excursion_tracer.stats.baseline_v2 import confidence, run_baseline_v2
from excursion_tracer.stats.commonality import scan_all
from excursion_tracer.stats.events_scan import events_scan
from excursion_tracer.stats.nullref import add_ratio, null_reference, product_adjusted

from .test_stats import T0, make_data

ONSET_H = 10 * 24.0  # 10일차 처리분부터
WINDOW = (11, 25)    # 알림 구간 (측정일)


def _after(t):
    return t >= ONSET_H


def ev(event_id, etype, step, ts_h, tool=None, chamber=None, recipe=None):
    return {"event_id": event_id, "event_type": etype, "step_id": step, "tool_id": tool,
            "chamber_id": chamber, "recipe_id": recipe, "ts": T0 + pd.Timedelta(hours=ts_h)}


# ---------------------------------------------------------------- 평소 차이

def test_null_reference_and_ratio(cfg):
    d = make_data(lambda i, s, r, t, p: 0.04 if r["S01"] == 1 and _after(t) else 0.0, window_days=WINDOW)
    ref = null_reference(d, cfg)
    assert ref.n_by_level["tool"] > 0 and 0 < ref.ref("tool") < 0.02
    assert ref.by_level["recipe"] is None or ref.ref("recipe") > 0
    scan = add_ratio(scan_all(d), ref).set_index("entity_id")
    assert scan.loc["S01-T2", "ratio"] > 3
    others = scan.drop(index=[e for e in scan.index if e.startswith("S01-T2")])
    assert others["ratio"].max() < scan.loc["S01-T2", "ratio"]


def test_null_window_uses_only_pre_alert_wafers(cfg):
    d = make_data(None, window_days=WINDOW)
    lo, hi = cfg.v2.null_window_test_days
    w = d.window_wafers(lo, hi)
    days = (w["test_ts"] - T0) / pd.Timedelta(days=1)
    assert days.min() >= lo and days.max() < hi + 1


def test_product_adjusted_separates_mix_change(cfg):
    # 10일차 이후 투입분은 P2(기준 수율이 낮음) 비중이 높다. 설비 효과는 없다.
    d = make_data(None, p2_by_lot=lambda i: 0.8 if i * 2.0 >= ONSET_H else 0.3, window_days=WINDOW,
                  lot_sd=0.005)
    pa = product_adjusted(d, cfg)
    assert pa["by_product"]["P2"]["share_alert"] > pa["by_product"]["P2"]["share_before"] + 0.3
    assert pa["overall_change"] < -0.01
    assert abs(pa["mix_fixed_change"]) < 0.005
    for v in pa["by_product"].values():
        assert abs(v["change"]) < 0.006


# ---------------------------------------------------------------- 이벤트 전후 스캔

def test_event_scan_finds_recipe_change(cfg):
    change_h = ONSET_H
    d = make_data(lambda i, s, r, t, p: 0.04 if t >= change_h else 0.0, window_days=WINDOW,
                  events=[ev("EV0001", "RECIPE_CHANGE", "S01", change_h, recipe="S01-R1"),
                          ev("EV0002", "PM", "S02", 5 * 24.0, tool="S02-T1"),
                          ev("EV0003", "CHAMBER_PM", "S02", 15 * 24.0, tool="S02-T2", chamber="S02-T2-C1")])
    df = events_scan(d, cfg).set_index("event_id")
    assert set(df.index) == {"EV0001", "EV0002", "EV0003"}
    top = df.sort_values("ratio", ascending=False).iloc[0]
    assert top.name == "EV0001" and top["target"] == "S01-R1" and top["level"] == "recipe"
    assert top["median_diff"] < -0.03 and top["q"] < 1e-6 and top["ratio"] > 3
    assert df.loc["EV0002", "ratio"] < 1.5 and df.loc["EV0003", "ratio"] < 1.5
    assert (df["null_ref"] > 0).all()


# ---------------------------------------------------------------- 기준선 v2

def test_baseline_v2_tool_recipe(cfg):
    d = make_data(lambda i, s, r, t, p: 0.04 if r["S03"] == 0 and p == "P2" and _after(t) else 0.0,
                  window_days=WINDOW)
    rep = run_baseline_v2(d, cfg)
    h = rep.hypotheses[0]
    assert rep.verdict == "cause_found" and rep.method == "baseline_v2"
    assert (h.entity_type, h.tool_id, h.recipe_id) == ("tool_recipe", "S03-T1", "S03-R2")
    Report.model_validate_json(rep.model_dump_json())


def test_baseline_v2_recipe_from_event(cfg):
    d = make_data(lambda i, s, r, t, p: 0.05 if t >= ONSET_H else 0.0, window_days=WINDOW,
                  events=[ev("EV0001", "RECIPE_CHANGE", "S01", ONSET_H, recipe="S01-R1")])
    rep = run_baseline_v2(d, cfg)
    assert rep.verdict == "cause_found"
    kinds = [(h.entity_type, h.recipe_id) for h in rep.hypotheses]
    assert ("recipe", "S01-R1") in kinds
    h = next(h for h in rep.hypotheses if h.recipe_id == "S01-R1")
    assert pd.Timestamp(h.onset) == T0 + pd.Timedelta(hours=ONSET_H)


def test_baseline_v2_planted_tool(cfg):
    d = make_data(lambda i, s, r, t, p: 0.04 if r["S01"] == 1 and _after(t) else 0.0, window_days=WINDOW)
    rep = run_baseline_v2(d, cfg)
    h = rep.hypotheses[0]
    assert rep.verdict == "cause_found" and (h.entity_type, h.tool_id) == ("tool", "S01-T2")
    assert h.confidence == cfg.v2.baseline_conf_by_ratio.above_3
    assert len(h.evidence) >= 2


def test_baseline_v2_null_is_no_cause(cfg):
    rep = run_baseline_v2(make_data(None, seed=3, lot_sd=0.0, window_days=WINDOW), cfg)
    assert rep.verdict == "no_equipment_cause" and rep.hypotheses == []
    assert "normal spread" in rep.no_cause_explanation


def test_confidence_bands(cfg):
    c = cfg.v2.baseline_conf_by_ratio
    assert confidence(cfg, 1.0) == c.below_1_5
    assert confidence(cfg, 2.0) == c.from_1_5_to_3
    assert confidence(cfg, 3.5) == c.above_3
    assert confidence(cfg, float("nan")) == c.below_1_5
