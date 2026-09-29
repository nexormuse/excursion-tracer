"""원인 없는 시나리오(F0a·F0b)에서 설비별 수율 차이가 미끼 수준인지 (§5.1.8).

판정: 설비별 |중앙값 차이|의 중앙값 < 0.01. 최대값은 물량이 적은 설비의 잡음과
경로 몰림(stickiness)에 따라 커지므로 판정에 쓰지 않는다.
판정 대상은 실제로 생성한 dev·test 세트다. 테스트용 임시 시나리오는 값을 기록만 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from excursion_tracer.eval.ground_truth import load_ground_truth

from .conftest import generated_dev_dirs, generated_test2_dirs, generated_test_dirs

DECOY_LIMIT = 0.01


def tool_median_diffs(scenario_dir: Path) -> pd.Series:
    """알림 구간에 측정된 웨이퍼로, 설비마다 (그 설비 − 같은 단계 다른 설비) 수율 중앙값 차이.

    제품 구성 차이를 빼기 위해 제품별 중앙값을 뺀 잔차로 비교한다.
    """
    alert = json.loads((scenario_dir / "alert.json").read_text())
    w = pd.read_parquet(scenario_dir / "wafers.parquet")
    w = w[(w["test_ts"] >= pd.Timestamp(alert["window_start"]))
          & (w["test_ts"] < pd.Timestamp(alert["window_end"]))]
    w = w.assign(r=w["yield"] - w.groupby("product")["yield"].transform("median"))
    h = pd.read_parquet(scenario_dir / "history.parquet", columns=["wafer_id", "step_id", "tool_id"])
    m = h.merge(w[["wafer_id", "r"]], on="wafer_id")
    out = {}
    for step, g in m.groupby("step_id"):
        for tool, gt in g.groupby("tool_id"):
            rest = g.loc[g["tool_id"] != tool, "r"]
            if len(rest):
                out[tool] = float(np.median(gt["r"]) - np.median(rest))
    return pd.Series(out)


def _no_cause_dirs(dirs):
    return [d for d in dirs if load_ground_truth(d)["fault_code"] in ("F0a", "F0b")]


def _assert_decoy_level(dirs):
    assert dirs
    bad = {}
    for d in dirs:
        diffs = tool_median_diffs(d).abs()
        if diffs.median() >= DECOY_LIMIT:
            bad[d.name] = round(float(diffs.median()), 4)
    assert not bad, bad


def test_no_cause_tool_differences_recorded_for_fixture(fixture_set, record_property):
    """테스트용 임시 시나리오: 판정하지 않고 값을 기록한다."""
    dirs = _no_cause_dirs(fixture_set.dirs())
    assert dirs
    for d in dirs:
        diffs = tool_median_diffs(d).abs()
        assert len(diffs) > 0
        record_property(d.name, {"median": round(float(diffs.median()), 4),
                                 "max": round(float(diffs.max()), 4)})


@pytest.mark.skipif(not generated_dev_dirs(), reason="data/dev 가 아직 없다")
def test_no_cause_tool_differences_are_decoy_level_dev_set():
    _assert_decoy_level(_no_cause_dirs(generated_dev_dirs()))


@pytest.mark.skipif(not generated_test_dirs(), reason="data/test 가 아직 없다")
def test_no_cause_tool_differences_are_decoy_level_test_set():
    _assert_decoy_level(_no_cause_dirs(generated_test_dirs()))


@pytest.mark.skipif(not generated_test2_dirs(), reason="data/test2 가 아직 없다")
def test_no_cause_tool_differences_are_decoy_level_test2_set():
    _assert_decoy_level(_no_cause_dirs(generated_test2_dirs()))
