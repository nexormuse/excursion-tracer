"""후보 간 교란 확인: 같은 웨이퍼를 공유하는 정도(파이 계수)와 층화 비교."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from excursion_tracer.stats.data import ScenarioData


def phi(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    n11 = np.sum(a & b)
    n10 = np.sum(a & ~b)
    n01 = np.sum(~a & b)
    n00 = np.sum(~a & ~b)
    den = np.sqrt(float(n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return float((n11 * n00 - n10 * n01) / den) if den > 0 else float("nan")


def _group(y: np.ndarray, low: np.ndarray, m: np.ndarray) -> dict:
    n = int(m.sum())
    return {
        "n": n,
        "mean_yield": float(y[m].mean()) if n else float("nan"),
        "low_rate": float(low[m].mean()) if n else float("nan"),
    }


def confounding_check(data: ScenarioData, entity_ids: list[str]) -> dict:
    """후보 쌍마다 파이 계수와 A만·B만·둘 다·둘 다 아님 웨이퍼의 평균 수율과 저수율 비율."""
    w = data.analysis_wafers
    y = w["resid"].to_numpy()
    low = w["low"].to_numpy()
    member = {e: data.membership(e).to_numpy() for e in entity_ids}
    pairs = []
    for a, b in combinations(entity_ids, 2):
        ma, mb = member[a], member[b]
        pairs.append({
            "a": a,
            "b": b,
            "phi": phi(ma, mb),
            "a_only": _group(y, low, ma & ~mb),
            "b_only": _group(y, low, ~ma & mb),
            "both": _group(y, low, ma & mb),
            "neither": _group(y, low, ~ma & ~mb),
        })
    return {"entity_ids": list(entity_ids), "pairs": pairs}
