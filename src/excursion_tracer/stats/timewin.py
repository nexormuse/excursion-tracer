"""시간 분석: 일별 추이, 변화점, 이벤트 전후 비교.

시간 분석은 시작 전 상태가 필요하므로 알림 구간이 아니라 측정된 웨이퍼 전체를 쓰고,
시각은 해당 단계의 처리 시각(track_in_ts)을 기준으로 한다.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from excursion_tracer.stats.data import ScenarioData


def _step_rows(data: ScenarioData, entity_id: str) -> tuple[pd.DataFrame, np.ndarray]:
    step = entity_id.split("-")[0]
    h = data.full_history
    hs = h[h["step_id"] == step]
    return hs, (hs[data.column_for(entity_id)] == entity_id).to_numpy()


def time_trend(data: ScenarioData, entity_id: str) -> pd.DataFrame:
    """처리일별 수율 잔차 평균: entity를 거친 웨이퍼 vs 같은 단계 나머지."""
    hs, m = _step_rows(data, entity_id)
    day = hs["track_in_ts"].dt.floor("D").to_numpy()
    df = pd.DataFrame({"day": day, "in": m, "resid": hs["resid"].to_numpy()})
    g = df.groupby(["day", "in"])["resid"].agg(["mean", "size"]).unstack("in")
    out = pd.DataFrame(index=g.index)
    out["mean_in"] = g[("mean", True)] if ("mean", True) in g else np.nan
    out["mean_out"] = g[("mean", False)] if ("mean", False) in g else np.nan
    out["n_in"] = g[("size", True)].fillna(0).astype(int) if ("size", True) in g else 0
    out["n_out"] = g[("size", False)].fillna(0).astype(int) if ("size", False) in g else 0
    out["diff"] = out["mean_in"] - out["mean_out"]
    return out.reset_index()


def _best_split(x: np.ndarray, min_seg: int) -> tuple[int, float]:
    """전후 평균 차이(앞 − 뒤)를 표본 크기로 가중한 값이 최대인 분할 위치."""
    n = len(x)
    cs = np.cumsum(x)
    k = np.arange(min_seg, n - min_seg + 1)
    before = cs[k - 1] / k
    after = (cs[-1] - cs[k - 1]) / (n - k)
    stat = (before - after) * np.sqrt(k * (n - k) / n)
    i = int(np.argmax(stat))
    return int(k[i]), float(stat[i])


def change_point(data: ScenarioData, entity_id: str, n_perm: int = 500) -> dict:
    """entity를 거친 웨이퍼의 로트별 평균 잔차를 처리 시각 순으로 놓고 하락 변화점을 찾는다.

    변화점 시각은 분할 뒤 첫 로트의 처리 시각. p값은 순서를 섞는 순열검정.
    잔차는 같은 날 같은 단계 나머지 웨이퍼의 평균을 빼서 전체 추이를 걷어 낸다.
    """
    hs, m = _step_rows(data, entity_id)
    day = hs["track_in_ts"].dt.floor("D")
    rest_mean = hs.loc[~m].groupby(day[~m])["resid"].mean()
    rel = hs.loc[m, "resid"] - day[m].map(rest_mean).fillna(0.0)
    lots = (
        pd.DataFrame({"lot": hs.loc[m, "lot_id"], "t": hs.loc[m, "track_in_ts"], "r": rel})
        .groupby("lot").agg(t=("t", "min"), r=("r", "mean"))
        .sort_values("t")
    )
    n = len(lots)
    min_seg = max(3, int(0.05 * n))
    if n < 2 * min_seg:
        return {"entity_id": entity_id, "ts": "", "mean_before": float("nan"),
                "mean_after": float("nan"), "p": float("nan"), "n_lots": n}
    x = lots["r"].to_numpy()
    k, stat = _best_split(x, min_seg)
    rng = np.random.default_rng(zlib.crc32(entity_id.encode()))
    exceed = 0
    for _ in range(n_perm):
        if _best_split(rng.permutation(x), min_seg)[1] >= stat:
            exceed += 1
    return {
        "entity_id": entity_id,
        "ts": lots["t"].iloc[k].isoformat(),
        "mean_before": float(x[:k].mean()),
        "mean_after": float(x[k:].mean()),
        "p": (exceed + 1) / (n_perm + 1),
        "n_lots": n,
    }


def before_after(data: ScenarioData, entity_id: str, event_ts, days: float = 3) -> dict:
    """이벤트 전후 ±days일에 entity에서 처리된 웨이퍼 수율 비교 (단측: 뒤가 더 낮은지)."""
    hs, m = _step_rows(data, entity_id)
    ts = pd.Timestamp(event_ts)
    t = hs["track_in_ts"]
    span = pd.Timedelta(days=days)
    before = hs.loc[m & ((t >= ts - span) & (t < ts)).to_numpy(), "resid"].to_numpy()
    after = hs.loc[m & ((t >= ts) & (t < ts + span)).to_numpy(), "resid"].to_numpy()
    out = {"entity_id": entity_id, "event_ts": ts.isoformat(), "days": days,
           "n_before": len(before), "n_after": len(after),
           "mean_before": float(before.mean()) if len(before) else float("nan"),
           "mean_after": float(after.mean()) if len(after) else float("nan"),
           "p": float("nan")}
    if len(before) and len(after):
        out["p"] = float(mannwhitneyu(after, before, alternative="less").pvalue)
    return out
