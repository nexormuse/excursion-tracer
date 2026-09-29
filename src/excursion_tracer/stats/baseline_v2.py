"""통계 기준선 v2: 평소 차이 기준, 이벤트 전후 스캔, 설비×레시피 가설을 쓴다 (method="baseline_v2").

1. 후보: 공통성 스캔(tool·chamber·recipe)과 이벤트 전후 스캔의 결과 중 q ≤ stats.no_cause_q이고
   수율이 낮아지는 쪽인 것. 각 후보에 평소 대비 배수(|중앙값 차이| ÷ 평소 차이)를 붙인다.
2. 설비×레시피: 제품별 레시피 단계의 설비 후보는 제품별 효과를 따로 본다. 한 제품에서만 평소 차이를
   넘고(배수 > 1) 다른 제품에서는 평소 차이 안(배수 ≤ 1)이면 그 설비 + 그 제품의 레시피 가설로 바꾼다.
3. 배수 내림차순으로 같은 대상을 한 번만 남겨 상위 3개를 가설로 낸다. 시작 시각은 이벤트 후보면
   이벤트 시각, 아니면 변화점.
4. 원인 없음: 배수가 baseline_no_cause_ratio 이상인 후보가 없으면 no_equipment_cause.
5. 신뢰도: 배수 1.5 미만 / 1.5~3 / 3 이상에 따라 단계로 준다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from excursion_tracer.agent.schema import Evidence, Hypothesis, Report
from excursion_tracer.config import Config
from excursion_tracer.stats.commonality import scan_all
from excursion_tracer.stats.data import ScenarioData
from excursion_tracer.stats.events_scan import events_scan
from excursion_tracer.stats.nullref import NullReference, add_ratio, null_reference, product_adjusted
from excursion_tracer.stats.timewin import change_point

METHOD = "baseline_v2"
N_HYPOTHESES = 3


def _f(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3g}"


def confidence(cfg: Config, ratio: float) -> float:
    c = cfg.v2.baseline_conf_by_ratio
    if not np.isfinite(ratio) or ratio < cfg.v2.baseline_no_cause_ratio:
        return c.below_1_5
    return c.from_1_5_to_3 if ratio < 3 else c.above_3


def tool_recipe_candidates(data: ScenarioData, cfg: Config, tools: pd.DataFrame,
                           ref: NullReference) -> dict[str, dict]:
    """설비 후보마다 제품별 효과를 보고, 한 제품에서만 평소 차이를 넘으면 설비×레시피로 바꿀 정보를 낸다."""
    recipe_of = data.fab.get("product_recipe", {})
    spec_steps = {s["step_id"] for s in data.fab["steps"] if s["product_specific_recipe"]}
    tools = tools[tools["step_id"].isin(spec_steps)]
    if tools.empty:
        return {}
    per = {}
    for p in sorted(data.analysis_wafers["product"].unique()):
        sp = scan_all(data, levels=("tool",), product=p, fdr_method=cfg.stats.fdr_method)
        per[p] = sp.set_index("entity_id")
    out = {}
    tool_ref = ref.ref("tool")
    for _, row in tools.iterrows():
        ent = row["entity_id"]
        eff = {p: (float(sp.loc[ent, "median_diff"]) if ent in sp.index else np.nan) for p, sp in per.items()}
        ratios = {p: (abs(v) / tool_ref if np.isfinite(v) and v < 0 else 0.0) for p, v in eff.items()}
        over = [p for p, r in ratios.items() if r > 1]
        within = [p for p, r in ratios.items() if r <= 1]
        if len(over) == 1 and len(within) == len(ratios) - 1 and over[0] in recipe_of:
            p = over[0]
            out[ent] = {"product": p, "recipe_id": f"{row['step_id']}-{recipe_of[p]}",
                        "ratio": ratios[p], "median_diff": eff[p], "by_product": eff,
                        "q": float(per[p].loc[ent, "q"])}
    return out


def candidates(data: ScenarioData, cfg: Config) -> tuple[pd.DataFrame, NullReference, dict]:
    ref = null_reference(data, cfg)
    scan = add_ratio(scan_all(data, fdr_method=cfg.stats.fdr_method,
                              stratify=cfg.stats.stratify_by_product), ref)
    q = cfg.stats.no_cause_q
    scan = scan[(scan["q"] <= q) & (scan["median_diff"] < 0)]
    rows = []
    tr = tool_recipe_candidates(data, cfg, scan[scan["level"] == "tool"], ref)
    for _, r in scan.iterrows():
        base = {"source": "commonality", "step_id": r["step_id"], "median_diff": r["median_diff"],
                "q": r["q"], "ratio": r["ratio"], "null_ref": r["null_ref"], "n": int(r["n_wafers"]),
                "low_in": r["low_rate_in"], "low_out": r["low_rate_out"], "ts": None,
                "tool_id": "", "chamber_id": "", "recipe_id": "", "note": ""}
        ent = r["entity_id"]
        if r["level"] == "tool" and ent in tr:
            t = tr[ent]
            rows.append({**base, "entity_type": "tool_recipe", "tool_id": ent, "recipe_id": t["recipe_id"],
                         "ratio": t["ratio"], "median_diff": t["median_diff"], "q": t["q"],
                         "note": "effect by product " + ", ".join(f"{p} {_f(v)}" for p, v in t["by_product"].items())})
        elif r["level"] == "tool":
            rows.append({**base, "entity_type": "tool", "tool_id": ent})
        elif r["level"] == "chamber":
            rows.append({**base, "entity_type": "chamber", "chamber_id": ent, "tool_id": ent.rsplit("-", 1)[0]})
        else:
            rows.append({**base, "entity_type": "recipe", "recipe_id": ent})
    ev = events_scan(data, cfg)
    ev = ev[(ev["q"] <= q) & (ev["median_diff"] < 0)]
    for _, r in ev.iterrows():
        base = {"source": "event", "step_id": r["step_id"], "median_diff": r["median_diff"], "q": r["q"],
                "ratio": r["ratio"], "null_ref": r["null_ref"], "n": int(r["n_after"]), "low_in": None,
                "low_out": None, "ts": r["ts"], "tool_id": "", "chamber_id": "", "recipe_id": "",
                "note": f"{r['event_type']} {r['event_id']} at {r['ts']:%Y-%m-%dT%H:%M}, before n {r['n_before']}"}
        if r["event_type"] == "RECIPE_CHANGE":
            rows.append({**base, "entity_type": "recipe", "recipe_id": r["target"]})
        elif r["event_type"] == "PM":
            rows.append({**base, "entity_type": "tool", "tool_id": r["target"]})
        else:
            rows.append({**base, "entity_type": "chamber", "chamber_id": r["target"],
                         "tool_id": r["target"].rsplit("-", 1)[0]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, ref, product_adjusted(data, cfg)
    df = df.sort_values(["ratio", "q"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    df["key"] = list(zip(df["entity_type"], df["tool_id"], df["chamber_id"], df["recipe_id"]))
    df = df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)
    return df, ref, product_adjusted(data, cfg)


def _entity(r) -> str:
    if r["entity_type"] == "tool_recipe":
        return f"{r['tool_id']} x {r['recipe_id']}"
    return r["chamber_id"] or r["recipe_id"] or r["tool_id"]


def _hypothesis(data: ScenarioData, cfg: Config, r, rank: int) -> Hypothesis:
    ent = _entity(r)
    ev = [Evidence(evidence_id="null_reference_ratio",
                   summary=f"{ent}: |median diff| {abs(r['median_diff']):.3f} vs normal spread {_f(r['null_ref'])} "
                           f"= {r['ratio']:.2f}x, q {_f(r['q'])}")]
    if r["source"] == "event":
        onset = pd.Timestamp(r["ts"]).isoformat()
        ev.append(Evidence(evidence_id="event_before_after",
                           summary=f"{r['note']}: after-before median diff {r['median_diff']:.3f}, after n {r['n']}"))
    else:
        target = r["tool_id"] if r["entity_type"] == "tool_recipe" else ent
        cp = change_point(data, target, n_perm=cfg.stats.permutation_n)
        onset = cp["ts"]
        ev.append(Evidence(evidence_id="mann_whitney_stouffer_bh",
                           summary=f"{ent}: median diff {r['median_diff']:.3f}, n {r['n']}, "
                                   f"low-yield rate {_f(r['low_in'])} vs {_f(r['low_out'])}"))
        ev.append(Evidence(evidence_id="change_point_permutation",
                           summary=f"change point {cp['ts'] or 'n/a'}, p {_f(cp['p'])}"))
        if r["note"]:
            ev.append(Evidence(evidence_id="product_split", summary=r["note"]))
    return Hypothesis(
        rank=rank, entity_type=r["entity_type"], step_id=r["step_id"], tool_id=r["tool_id"],
        chamber_id=r["chamber_id"], recipe_id=r["recipe_id"], onset=onset or "",
        confidence=confidence(cfg, float(r["ratio"])), evidence=ev,
        falsification_test=f"If {ent} is not the cause, its difference should stay within the normal spread "
                           f"measured before the alert.",
        recommended_action=f"Inspect {ent} and compare wafers processed before and after the onset.",
    )


def run_baseline_v2(data: ScenarioData, cfg: Config) -> Report:
    cands, ref, pa = candidates(data, cfg)
    thr = cfg.v2.baseline_no_cause_ratio
    top_ratio = float(cands["ratio"].max()) if len(cands) else 0.0
    mix = (f"product-adjusted change: overall {pa['overall_change']:.3f}, with fixed product mix "
           f"{_f(pa['mix_fixed_change'])}; " + ", ".join(
               f"{p} {_f(v['change'])} (share {_f(v['share_before'])} -> {_f(v['share_alert'])})"
               for p, v in pa["by_product"].items()))
    if len(cands) == 0 or top_ratio < thr:
        return Report(
            scenario_id=data.scenario_id, method=METHOD, verdict="no_equipment_cause",
            no_cause_explanation=(f"no candidate beyond {thr}x the normal spread (max {top_ratio:.2f}x); {mix}"),
            limitations="Statistics only.",
        )
    hyps = [_hypothesis(data, cfg, r, i + 1) for i, (_, r) in enumerate(cands.head(N_HYPOTHESES).iterrows())]
    return Report(scenario_id=data.scenario_id, method=METHOD, verdict="cause_found", hypotheses=hyps,
                  limitations=f"Statistics only; {mix}")
