"""시나리오 관측 파일 읽기와 분석용 표 준비.

읽는 파일: fab.json, history.parquet, wafers.parquet, events.parquet, lots.parquet, alert.json.
분석 대상은 알림 구간에 측정된 웨이퍼다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

OBSERVED_INPUTS = (
    "fab.json", "history.parquet", "wafers.parquet", "events.parquet", "lots.parquet", "alert.json",
)


@dataclass
class ScenarioData:
    scenario_id: str
    fab: dict
    alert: dict
    history: pd.DataFrame  # 웨이퍼 × 단계
    wafers: pd.DataFrame  # 측정된 웨이퍼 전체
    events: pd.DataFrame
    lots: pd.DataFrame

    @property
    def window_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.alert["window_start"])

    @property
    def window_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.alert["window_end"])

    @property
    def low_threshold(self) -> float:
        return float(self.alert["baseline"]["low_yield_threshold"])

    @cached_property
    def analysis_wafers(self) -> pd.DataFrame:
        """알림 구간에 측정된 웨이퍼. resid = 수율 − 같은 제품의 구간 내 중앙값."""
        w = self.wafers
        w = w[(w["test_ts"] >= self.window_start) & (w["test_ts"] < self.window_end)].copy()
        w["low"] = w["yield"] < self.low_threshold
        w["resid"] = w["yield"] - w.groupby("product")["yield"].transform("median")
        return w.reset_index(drop=True)

    @cached_property
    def all_wafers(self) -> pd.DataFrame:
        """측정된 웨이퍼 전체 (시간 분석용). resid 기준은 기준 구간이 아니라 전체 중앙값."""
        w = self.wafers.copy()
        w["low"] = w["yield"] < self.low_threshold
        w["resid"] = w["yield"] - w.groupby("product")["yield"].transform("median")
        return w.reset_index(drop=True)

    @cached_property
    def analysis_history(self) -> pd.DataFrame:
        """분석 대상 웨이퍼의 이력에 수율 정보를 붙인 표."""
        return self._join(self.analysis_wafers)

    @cached_property
    def full_history(self) -> pd.DataFrame:
        return self._join(self.all_wafers)

    def _join(self, w: pd.DataFrame) -> pd.DataFrame:
        cols = ["wafer_id", "product", "yield", "low", "resid"]
        return self.history.merge(w[cols], on="wafer_id", how="inner")

    @cached_property
    def steps(self) -> dict[str, dict]:
        return {s["step_id"]: s for s in self.fab["steps"]}

    @cached_property
    def multi_chamber_tools(self) -> set[str]:
        return {t["tool_id"] for s in self.fab["steps"] for t in s["tools"] if len(t["chambers"]) > 1}

    def entity_level(self, entity_id: str) -> str:
        """ID 모양으로 종류를 안다: S12-T3 설비, S12-T3-C2 챔버, S12-R1 레시피."""
        parts = entity_id.split("-")
        if len(parts) == 3 and parts[2].startswith("C"):
            return "chamber"
        if len(parts) == 2 and parts[1].startswith("T"):
            return "tool"
        if len(parts) == 2 and parts[1].startswith("R"):
            return "recipe"
        raise ValueError(f"알 수 없는 ID: {entity_id}")

    def column_for(self, entity_id: str) -> str:
        return {"tool": "tool_id", "chamber": "chamber_id", "recipe": "recipe_id"}[
            self.entity_level(entity_id)
        ]

    def membership(self, entity_id: str, history: pd.DataFrame | None = None) -> pd.Series:
        """분석 웨이퍼마다 이 entity를 거쳤는지 (wafer_id 인덱스, bool)."""
        h = self.analysis_history if history is None else history
        step = entity_id.split("-")[0]
        hs = h[h["step_id"] == step]
        hit = set(hs.loc[hs[self.column_for(entity_id)] == entity_id, "wafer_id"])
        ids = self.analysis_wafers["wafer_id"]
        return pd.Series(ids.isin(hit).to_numpy(), index=ids.to_numpy())


def load_scenario(scenario_dir: str | Path) -> ScenarioData:
    d = Path(scenario_dir)
    return ScenarioData(
        scenario_id=d.name,
        fab=json.loads((d / "fab.json").read_text(encoding="utf-8")),
        alert=json.loads((d / "alert.json").read_text(encoding="utf-8")),
        history=pd.read_parquet(d / "history.parquet"),
        wafers=pd.read_parquet(d / "wafers.parquet"),
        events=pd.read_parquet(d / "events.parquet"),
        lots=pd.read_parquet(d / "lots.parquet"),
    )


def stouffer(pvalues, weights) -> float:
    """단측 p값들을 가중 Stouffer로 합친다."""
    from scipy.stats import norm

    p = np.clip(np.asarray(pvalues, dtype=float), 1e-300, 1 - 1e-16)
    w = np.asarray(weights, dtype=float)
    if len(p) == 0 or not np.any(w > 0):
        return float("nan")
    z = norm.isf(p)
    return float(norm.sf(np.sum(w * z) / np.sqrt(np.sum(w**2))))
