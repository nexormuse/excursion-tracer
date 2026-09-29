"""평소 차이 기준과 제품 보정 하락.

- 평소 차이: 알림 전 구간(측정일 null_window_test_days, 원인 시작 전)에서 같은 공통성 스캔
  (tool·chamber·recipe, 제품 층화)을 돌려, 수준별 |중앙값 차이|의 null_quantile 백분위수를 둔다.
  그 구간에 비교가 없는 수준은 설비 수준 값을 쓴다.
- 평소 대비 배수: 알림 구간 후보의 |중앙값 차이| ÷ 그 수준의 평소 차이.
- 제품 보정 하락: 제품별 (알림 구간 평균 − 알림 전 평균)과 제품 비중 변화.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from excursion_tracer.config import Config
from excursion_tracer.stats.commonality import LEVELS, scan_all
from excursion_tracer.stats.data import ScenarioData

FALLBACK_LEVEL = "tool"


@dataclass
class NullReference:
    by_level: dict[str, float]  # 수준 → 평소 차이
    n_by_level: dict[str, int]  # 그 값을 만든 비교 수
    window_days: tuple[int, int]
    quantile: float

    def ref(self, level: str) -> float:
        v = self.by_level.get(level)
        if v is None or not np.isfinite(v) or v <= 0:
            v = self.by_level.get(FALLBACK_LEVEL, np.nan)
        return float(v)

    def ratio(self, level: str, median_diff: float) -> float:
        r = self.ref(level)
        return float(abs(median_diff) / r) if r and np.isfinite(r) and r > 0 else float("nan")


def null_reference(data: ScenarioData, cfg: Config) -> NullReference:
    lo, hi = cfg.v2.null_window_test_days
    hist = data.window_history(lo, hi)
    scan = scan_all(data, fdr_method=cfg.stats.fdr_method, stratify=cfg.stats.stratify_by_product,
                    history=hist)
    by, n = {}, {}
    for level in LEVELS:
        d = scan.loc[scan["level"] == level, "median_diff"].abs()
        n[level] = int(len(d))
        by[level] = float(np.quantile(d, cfg.v2.null_quantile)) if len(d) else None
    return NullReference(by, n, (lo, hi), cfg.v2.null_quantile)


def add_ratio(scan: pd.DataFrame, ref: NullReference) -> pd.DataFrame:
    """스캔 결과에 평소 차이(null_ref)와 평소 대비 배수(ratio)를 붙인다."""
    out = scan.copy()
    out["null_ref"] = [ref.ref(lv) for lv in out["level"]]
    out["ratio"] = out["median_diff"].abs() / out["null_ref"]
    return out


def product_adjusted(data: ScenarioData, cfg: Config) -> dict:
    """제품별 알림 구간 평균 − 알림 전 평균, 전체 평균의 변화, 제품 비중 변화."""
    lo, hi = cfg.v2.null_window_test_days
    before = data.window_wafers(lo, hi)
    during = data.analysis_wafers
    out = {"window_days_before": [lo, hi], "by_product": {}}
    for p in sorted(set(before["product"]) | set(during["product"])):
        b = before.loc[before["product"] == p, "yield"]
        a = during.loc[during["product"] == p, "yield"]
        out["by_product"][p] = {
            "mean_before": float(b.mean()) if len(b) else None,
            "mean_alert": float(a.mean()) if len(a) else None,
            "change": float(a.mean() - b.mean()) if len(a) and len(b) else None,
            "share_before": float(len(b) / len(before)) if len(before) else None,
            "share_alert": float(len(a) / len(during)) if len(during) else None,
            "n_before": int(len(b)), "n_alert": int(len(a)),
        }
    out["overall_change"] = float(during["yield"].mean() - before["yield"].mean())
    # 제품 비중이 알림 전과 같았다면의 전체 변화 (제품별 변화를 알림 전 비중으로 가중)
    parts = [(v["change"], v["share_before"]) for v in out["by_product"].values()
             if v["change"] is not None and v["share_before"] is not None]
    out["mix_fixed_change"] = float(sum(c * w for c, w in parts)) if parts else None
    return out
