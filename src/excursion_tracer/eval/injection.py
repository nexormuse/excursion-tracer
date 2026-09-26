"""생성기 검증용: 정답 요인을 거친 웨이퍼와 나머지의 수율 차이를 관측 파일에서 다시 잰다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from excursion_tracer.eval.ground_truth import load_ground_truth


def fault_weights(history: pd.DataFrame, fault: dict, observe_end: pd.Timestamp) -> pd.Series:
    """웨이퍼별로 이 원인이 δ의 몇 배만큼 수율을 깎았는지 (드리프트는 0~1, 나머지는 0 또는 1)."""
    h = history[history["step_id"] == fault["step_id"]]
    onset = pd.Timestamp(fault["onset_ts"])
    hit = h["track_in_ts"] >= onset
    kind = fault["type"]
    if kind == "F1":
        hit &= h["chamber_id"] == fault["chamber_id"]
    elif kind == "F2":
        hit &= (h["tool_id"] == fault["tool_id"]) & (h["recipe_id"] == fault["recipe_id"])
    elif kind == "F3":
        hit &= h["tool_id"] == fault["tool_id"]
    elif kind != "F5":
        raise ValueError(f"모르는 원인 유형: {kind}")
    w = hit.astype(float)
    if kind == "F3":
        w = w * ((h["track_in_ts"] - onset) / (observe_end - onset)).clip(0, 1)
    return pd.Series(w.to_numpy(), index=h["wafer_id"].to_numpy())


def injection_check(scenario_dir: str | Path, observe_end: pd.Timestamp) -> list[dict]:
    """원인마다 관측 파일에서 잰 효과 크기와 설정한 δ를 돌려준다.

    다른 원인에 걸린 웨이퍼를 뺀 뒤, 제품별 평균을 뺀 수율 잔차를 이 원인의 가중치에
    회귀한 기울기(부호 반대)를 효과 크기로 본다. 가중치가 0/1이면 이는
    "나머지 평균 − 걸린 웨이퍼 평균"과 같고, 드리프트는 0 → δ 기울기를 잰다.
    """
    d = Path(scenario_dir)
    gt = load_ground_truth(d)
    wafers = pd.read_parquet(d / "wafers.parquet")
    history = pd.read_parquet(d / "history.parquet")
    prod_mean = wafers.groupby("product")["yield"].transform("mean")
    resid = (wafers["yield"] - prod_mean).to_numpy()
    index = wafers["wafer_id"].to_numpy()

    weights = [
        fault_weights(history, f, observe_end).reindex(index).fillna(0.0).to_numpy()
        for f in gt["faults"]
    ]
    out = []
    for i, f in enumerate(gt["faults"]):
        other = np.zeros(len(resid), dtype=bool)
        for j, w in enumerate(weights):
            if j != i:
                other |= w > 0
        keep = ~other
        x, y = weights[i][keep], resid[keep]
        slope = float(np.polyfit(x, y, 1)[0])
        out.append(
            {
                "type": f["type"],
                "n_hit": int((x > 0).sum()),
                "delta": float(f["delta"]),
                "measured": -slope,
            }
        )
    return out
