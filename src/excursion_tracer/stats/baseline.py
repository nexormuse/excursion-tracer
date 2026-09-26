"""통계만 쓰는 기준선 판정.

1. 설비·챔버·레시피 스캔 결과를 합쳐 q값 오름차순, 동률은 |중앙값 차이| 내림차순으로 정렬한다.
2. 상위 3개를 가설로 내고, 시작 시각은 변화점으로 채운다.
3. 원인 없음: 최소 q > no_cause_q 이거나, q ≤ no_cause_q인 후보의 최대 |중앙값 차이| < no_cause_min_effect.
4. 신뢰도 = 1 − q.
"""

from __future__ import annotations

from excursion_tracer.agent.schema import Evidence, Hypothesis, Report
from excursion_tracer.config import Config
from excursion_tracer.stats.commonality import scan_all
from excursion_tracer.stats.data import ScenarioData
from excursion_tracer.stats.timewin import change_point

METHOD = "baseline"
N_HYPOTHESES = 3


def _fmt(x: float) -> str:
    return f"{x:.3g}"


def _hypothesis(data: ScenarioData, row, rank: int, n_perm: int) -> Hypothesis:
    ent, level, step = row["entity_id"], row["level"], row["step_id"]
    ids = {"tool_id": "", "chamber_id": "", "recipe_id": ""}
    if level == "tool":
        ids["tool_id"] = ent
    elif level == "chamber":
        ids["chamber_id"] = ent
        ids["tool_id"] = ent.rsplit("-", 1)[0]
    else:
        ids["recipe_id"] = ent
    cp = change_point(data, ent, n_perm=n_perm)
    evidence = [
        Evidence(
            evidence_id="mann_whitney_stouffer_bh",
            summary=(f"{ent}: median yield diff {row['median_diff']:.3f} vs rest of {step}, "
                     f"n {row['n_wafers']}, p {_fmt(row['p'])}, q {_fmt(row['q'])}"),
        ),
        Evidence(
            evidence_id="fisher_mantel_haenszel",
            summary=(f"low-yield rate {row['low_rate_in']:.3f} vs {row['low_rate_out']:.3f}, "
                     f"odds ratio {row['odds_ratio']:.3f}, p {_fmt(row['p_fisher'])}"),
        ),
        Evidence(
            evidence_id="change_point_permutation",
            summary=(f"change point {cp['ts'] or 'n/a'}, lot mean before {cp['mean_before']:.3f}, "
                     f"after {cp['mean_after']:.3f}, p {_fmt(cp['p'])}, lots {cp['n_lots']}"),
        ),
    ]
    return Hypothesis(
        rank=rank,
        entity_type=level,
        step_id=step,
        onset=cp["ts"],
        confidence=float(min(1.0, max(0.0, 1.0 - row["q"]))),
        evidence=evidence,
        falsification_test=(f"If {ent} is not the cause, its wafers should not show lower yield than "
                            f"other {step} wafers within each product, and no drop at the change point."),
        recommended_action=f"Inspect {ent} and compare wafers processed before and after the change point.",
        **ids,
    )


def run_baseline(data: ScenarioData, cfg: Config) -> Report:
    sc = cfg.stats
    scan = scan_all(data, fdr_method=sc.fdr_method, stratify=sc.stratify_by_product)
    sig = scan[scan["q"] <= sc.no_cause_q]
    min_q = float(scan["q"].min()) if len(scan) else float("nan")
    max_eff = float(sig["median_diff"].abs().max()) if len(sig) else 0.0
    no_cause = len(sig) == 0 or max_eff < sc.no_cause_min_effect
    if no_cause:
        return Report(
            scenario_id=data.scenario_id,
            method=METHOD,
            verdict="no_equipment_cause",
            no_cause_explanation=(
                f"min q {_fmt(min_q)} (threshold {sc.no_cause_q}); max |median diff| among "
                f"q <= {sc.no_cause_q}: {max_eff:.3f} (threshold {sc.no_cause_min_effect})"
            ),
            limitations="Commonality statistics only; no product-mix or recipe-change analysis.",
        )
    hyps = [
        _hypothesis(data, row, i + 1, sc.permutation_n)
        for i, (_, row) in enumerate(scan.head(N_HYPOTHESES).iterrows())
    ]
    return Report(
        scenario_id=data.scenario_id,
        method=METHOD,
        verdict="cause_found",
        hypotheses=hyps,
        limitations="Commonality statistics only; confounding between candidates is not resolved.",
    )
