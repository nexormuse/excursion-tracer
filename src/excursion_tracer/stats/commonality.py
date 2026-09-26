"""공통성 스캔: 단계마다 설비·챔버·레시피를 거친 웨이퍼와 같은 단계 나머지 웨이퍼를 비교한다.

- 수율 비교: Mann-Whitney U (단측, 더 낮은지), 제품별로 계산해 가중 Stouffer로 합친다.
- 저수율 비율 비교: Fisher 정확검정 (단측), 제품별 p를 같은 방식으로 합치고 오즈비는 Mantel-Haenszel.
- 효과: 수율 잔차(수율 − 제품별 중앙값)의 중앙값 차이, 저수율 비율 차이.
  층화하지 않으면(stratify=False) 원래 수율로 비교한다.
- 모든 수준의 Mann-Whitney p값을 한 번에 Benjamini-Hochberg로 보정해 q값을 붙인다.

챔버 수준은 다챔버 설비의 챔버만 본다 (단일 챔버는 설비와 같은 비교다).
레시피 수준은 레시피가 2개 이상인 단계만 보고, 제품 안에서 비교할 수 없는 층은 건너뛴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm, rankdata
from statsmodels.stats.multitest import multipletests

from excursion_tracer.stats.data import ScenarioData, stouffer

LEVELS = ("tool", "chamber", "recipe")
LEVEL_COLUMN = {"tool": "tool_id", "chamber": "chamber_id", "recipe": "recipe_id"}
COLUMNS = [
    "entity_id", "level", "step_id", "n_wafers", "low_rate_in", "low_rate_out",
    "median_diff", "odds_ratio", "p", "p_fisher", "q",
]


def mwu_less(ranks: np.ndarray, in_mask: np.ndarray, tie_term: float) -> float:
    """미리 구한 순위로 Mann-Whitney U 단측(in < out) p값 (정규근사, 연속성 보정)."""
    n1 = int(in_mask.sum())
    n = len(ranks)
    n2 = n - n1
    if n1 == 0 or n2 == 0:
        return float("nan")
    u1 = ranks[in_mask].sum() - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = (u1 - mu + 0.5) / np.sqrt(var)
    return float(norm.cdf(z))


def _tie_term(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    return float(np.sum(counts**3 - counts))


def _entities(data: ScenarioData, step_id: str, g: pd.DataFrame, level: str) -> list[str]:
    col = LEVEL_COLUMN[level]
    ids = sorted(g[col].unique())
    if level == "chamber":
        ids = [c for c in ids if c.rsplit("-", 1)[0] in data.multi_chamber_tools]
    elif level == "recipe":
        if len(data.steps[step_id]["recipes"]) < 2:
            return []
    elif level == "tool" and len(ids) < 2:
        return []
    return ids


def scan_all(
    data: ScenarioData,
    levels=LEVELS,
    product: str | None = None,
    fdr_method: str = "fdr_bh",
    stratify: bool = True,
) -> pd.DataFrame:
    """모든 단계·수준의 비교 결과 (BH 보정 q 포함). q 오름차순, 동률은 |median_diff| 내림차순."""
    h = data.analysis_history
    if product:
        h = h[h["product"] == product]
    value = "resid" if stratify else "yield"
    rows = []
    for step_id, g in h.groupby("step_id", sort=True):
        strata = list(g.groupby("product")) if stratify else [("all", g)]
        prepared = []
        for _, s in strata:
            x = s[value].to_numpy()
            prepared.append((s, rankdata(x), _tie_term(x)))
        for level in levels:
            col = LEVEL_COLUMN[level]
            for ent in _entities(data, step_id, g, level):
                ps, pf, ws = [], [], []
                mh_num = mh_den = 0.0
                n_in_tot = n_out_tot = low_in_tot = low_out_tot = 0
                r_in, r_out = [], []
                for s, ranks, tie in prepared:
                    m = (s[col] == ent).to_numpy()
                    n1, n2 = int(m.sum()), int((~m).sum())
                    if n1 == 0 or n2 == 0:
                        continue
                    low = s["low"].to_numpy()
                    a, b = int(low[m].sum()), n1 - int(low[m].sum())
                    c, d = int(low[~m].sum()), n2 - int(low[~m].sum())
                    ps.append(mwu_less(ranks, m, tie))
                    pf.append(fisher_exact([[a, b], [c, d]], alternative="greater").pvalue)
                    ws.append(np.sqrt(n1 * n2 / (n1 + n2)))
                    n = n1 + n2
                    mh_num += a * d / n
                    mh_den += b * c / n
                    n_in_tot += n1
                    n_out_tot += n2
                    low_in_tot += a
                    low_out_tot += c
                    x = s[value].to_numpy()
                    r_in.append(x[m])
                    r_out.append(x[~m])
                if not ps:
                    continue
                med = float(np.median(np.concatenate(r_in)) - np.median(np.concatenate(r_out)))
                rows.append({
                    "entity_id": ent,
                    "level": level,
                    "step_id": step_id,
                    "n_wafers": n_in_tot,
                    "low_rate_in": low_in_tot / n_in_tot,
                    "low_rate_out": low_out_tot / n_out_tot,
                    "median_diff": med,
                    "odds_ratio": mh_num / mh_den if mh_den > 0 else float("inf"),
                    "p": stouffer(ps, ws),
                    "p_fisher": stouffer(pf, ws),
                })
    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "q"])
    if len(df):
        df["q"] = multipletests(df["p"].to_numpy(), method=fdr_method)[1]
    else:
        df["q"] = pd.Series(dtype=float)
    return sort_candidates(df)


def sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(_abs=df["median_diff"].abs())
        .sort_values(["q", "_abs", "entity_id"], ascending=[True, False, True], kind="mergesort")
        .drop(columns="_abs")
        .reset_index(drop=True)
    )


def commonality_scan(
    data: ScenarioData,
    level: str,
    product: str | None = None,
    top_k: int = 10,
    fdr_method: str = "fdr_bh",
) -> pd.DataFrame:
    """한 수준의 상위 k개. q값은 모든 수준을 함께 보정한 값이다."""
    if level not in LEVELS:
        raise ValueError(f"level은 {LEVELS} 중 하나: {level}")
    df = scan_all(data, product=product, fdr_method=fdr_method)
    return df[df["level"] == level].head(top_k).reset_index(drop=True)
