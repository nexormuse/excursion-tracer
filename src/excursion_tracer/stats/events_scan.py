"""이벤트 전후 스캔: 관측 기간의 모든 이벤트에 대해 대상 웨이퍼의 전후 수율을 비교한다.

- 대상: 레시피 변경 → 그 단계에서 그 레시피를 쓴 모든 웨이퍼, PM → 그 설비, 챔버 PM → 그 챔버.
- 비교: 이벤트 전후 event_scan_days일 안에 그 단계에서 처리된 웨이퍼 (처리 시각 기준).
  제품별로 Mann-Whitney 단측(뒤가 더 낮은지)을 구해 가중 Stouffer로 합치고, 전체를 BH 보정한다.
- 효과: 뒤 − 앞의 수율 잔차 중앙값 차이.
- 평소 차이: 알림 전 구간이 끝나기 전에 전후 창이 모두 들어가는 가짜 이벤트 시각을 두고,
  같은 종류의 모든 대상(레시피 전부, 설비 전부, 다챔버 챔버 전부)에서 같은 비교를 한 |중앙값 차이|의
  null_quantile 백분위수를 이벤트 종류별로 쓴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from excursion_tracer.config import Config
from excursion_tracer.stats.data import ScenarioData, stouffer

TARGET_COLUMN = {"RECIPE_CHANGE": "recipe_id", "PM": "tool_id", "CHAMBER_PM": "chamber_id"}
TARGET_LEVEL = {"RECIPE_CHANGE": "recipe", "PM": "tool", "CHAMBER_PM": "chamber"}
COLUMNS = ["event_id", "event_type", "target", "level", "step_id", "ts", "n_before", "n_after",
           "median_diff", "p", "q", "null_ref", "ratio"]


def _target(ev) -> str:
    return ev[TARGET_COLUMN[ev["event_type"]]]


def _compare(hs: pd.DataFrame, mask: np.ndarray, ts: pd.Timestamp, days: float) -> dict | None:
    t = hs["track_in_ts"]
    span = pd.Timedelta(days=days)
    before = mask & ((t >= ts - span) & (t < ts)).to_numpy()
    after = mask & ((t >= ts) & (t < ts + span)).to_numpy()
    if not before.any() or not after.any():
        return None
    ps, ws = [], []
    prod = hs["product"].to_numpy()
    resid = hs["resid"].to_numpy()
    for p in np.unique(prod):
        b = resid[before & (prod == p)]
        a = resid[after & (prod == p)]
        if len(a) and len(b):
            ps.append(mannwhitneyu(a, b, alternative="less").pvalue)
            ws.append(np.sqrt(len(a) * len(b) / (len(a) + len(b))))
    if not ps:
        return None
    return {
        "n_before": int(before.sum()), "n_after": int(after.sum()),
        "median_diff": float(np.median(resid[after]) - np.median(resid[before])),
        "p": stouffer(ps, ws),
    }


def _by_step(data: ScenarioData) -> dict[str, pd.DataFrame]:
    h = data.full_history
    return {s: g.reset_index(drop=True) for s, g in h.groupby("step_id")}


def event_null_reference(data: ScenarioData, cfg: Config, steps: dict | None = None) -> dict[str, float]:
    steps = steps or _by_step(data)
    days = cfg.v2.event_scan_days
    _, hi = cfg.v2.null_window_test_days
    ts = data.day0 + pd.Timedelta(days=hi + 1 - days)
    diffs: dict[str, list[float]] = {k: [] for k in TARGET_COLUMN}
    multi = data.multi_chamber_tools
    for step_id, hs in steps.items():
        for etype, col in TARGET_COLUMN.items():
            values = hs[col].unique()
            if etype == "CHAMBER_PM":
                values = [c for c in values if c.rsplit("-", 1)[0] in multi]
            for v in values:
                r = _compare(hs, (hs[col] == v).to_numpy(), ts, days)
                if r is not None:
                    diffs[etype].append(abs(r["median_diff"]))
    q = cfg.v2.null_quantile
    return {k: (float(np.quantile(v, q)) if v else float("nan")) for k, v in diffs.items()}


def events_scan(data: ScenarioData, cfg: Config) -> pd.DataFrame:
    """모든 이벤트의 전후 비교 (q 오름차순, 동률은 |중앙값 차이| 내림차순)."""
    steps = _by_step(data)
    days = cfg.v2.event_scan_days
    rows = []
    for _, ev in data.events.iterrows():
        hs = steps.get(ev["step_id"])
        target = _target(ev)
        if hs is None or target is None or (isinstance(target, float) and np.isnan(target)):
            continue
        col = TARGET_COLUMN[ev["event_type"]]
        r = _compare(hs, (hs[col] == target).to_numpy(), pd.Timestamp(ev["ts"]), days)
        if r is None:
            continue
        rows.append({"event_id": ev["event_id"], "event_type": ev["event_type"], "target": target,
                     "level": TARGET_LEVEL[ev["event_type"]], "step_id": ev["step_id"],
                     "ts": pd.Timestamp(ev["ts"]), **r})
    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c not in ("q", "null_ref", "ratio")])
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["q"] = multipletests(df["p"].to_numpy(), method=cfg.stats.fdr_method)[1]
    ref = event_null_reference(data, cfg, steps)
    df["null_ref"] = df["event_type"].map(ref)
    df["ratio"] = df["median_diff"].abs() / df["null_ref"]
    return (df.assign(_a=df["median_diff"].abs())
              .sort_values(["q", "_a", "event_id"], ascending=[True, False, True], kind="mergesort")
              .drop(columns="_a").reset_index(drop=True))
