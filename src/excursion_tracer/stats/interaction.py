"""설비×레시피 2원 분산분석."""

from __future__ import annotations

import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from excursion_tracer.stats.data import ScenarioData


def interaction_test(data: ScenarioData, step_id: str) -> dict:
    """알림 구간 웨이퍼로 yield ~ tool * recipe. 상호작용 p값과 가법 모형에서 가장 벗어난 셀."""
    h = data.analysis_history
    g = h[h["step_id"] == step_id][["tool_id", "recipe_id", "yield"]]
    out = {"step_id": step_id, "p_interaction": float("nan"), "worst_cell": {}}
    if g["tool_id"].nunique() < 2 or g["recipe_id"].nunique() < 2:
        out["note"] = "설비나 레시피가 1개뿐이라 검정할 수 없다"
        return out
    model = smf.ols("Q('yield') ~ C(tool_id) * C(recipe_id)", data=g).fit()
    table = anova_lm(model, typ=2)
    key = "C(tool_id):C(recipe_id)"
    out["p_interaction"] = float(table.loc[key, "PR(>F)"]) if key in table.index else float("nan")

    add = smf.ols("Q('yield') ~ C(tool_id) + C(recipe_id)", data=g).fit()
    g = g.assign(pred=add.fittedvalues)
    cells = g.groupby(["tool_id", "recipe_id"]).agg(
        mean=("yield", "mean"), pred=("pred", "mean"), n=("yield", "size")
    )
    cells["dev"] = cells["mean"] - cells["pred"]
    se = g["yield"].std(ddof=1) / np.sqrt(cells["n"])
    cells["z"] = cells["dev"] / se
    worst = cells["z"].idxmin()
    row = cells.loc[worst]
    out["worst_cell"] = {
        "tool_id": worst[0], "recipe_id": worst[1], "n": int(row["n"]),
        "mean": float(row["mean"]), "additive_pred": float(row["pred"]), "deviation": float(row["dev"]),
    }
    return out

