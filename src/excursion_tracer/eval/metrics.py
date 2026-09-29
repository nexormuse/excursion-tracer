"""지표 계산: 비율에는 Wilson 95% 신뢰구간을 붙인다."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from excursion_tracer.agent.schema import Hypothesis
from excursion_tracer.eval.match import NO_CAUSE_CODES, TOP_K, score

Z95 = 1.959963984540054
CALIBRATION_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001)


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def rate(values) -> dict:
    """True/False 목록 (None은 제외) → 비율과 Wilson 신뢰구간."""
    v = [bool(x) for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))]
    k, n = sum(v), len(v)
    lo, hi = wilson(k, n)
    return {"value": k / n if n else None, "k": k, "n": n,
            "ci95": [lo, hi] if n else [None, None]}


def mrr(ranks) -> dict:
    r = [0.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else 1.0 / x for x in ranks]
    return {"value": float(np.mean(r)) if r else None, "n": len(r)}


def _is_cause(df: pd.DataFrame) -> pd.Series:
    return ~df["fault_code"].isin(NO_CAUSE_CODES)


def core_metrics(df: pd.DataFrame) -> dict:
    """한 방법의 시나리오별 채점 표로 주요 지표를 만든다."""
    c = df[_is_cause(df)]
    z = df[~_is_cause(df)]
    out = {
        "n_scenarios": int(len(df)),
        "n_cause": int(len(c)),
        "n_no_cause": int(len(z)),
        "hit1_strict": rate(c["hit1_strict"]),
        "hit3_strict": rate(c["hit3_strict"]),
        "hit1_loose": rate(c["hit1_loose"]),
        "hit3_loose": rate(c["hit3_loose"]),
        "mrr_strict": mrr(c["rank_strict"]),
        "mrr_loose": mrr(c["rank_loose"]),
        "false_alarm": rate(z["false_alarm"]),
        "miss_rate": rate(c["missed"]),
        "onset_accuracy": rate(c["onset_ok"]),
        "f0b_explained": rate(df.loc[df["fault_code"] == "F0b", "f0b_explained"]),
        "invalid_rate": rate(df["invalid"]),
        "f4_both_hit3_strict": rate(df.loc[df["fault_code"] == "F4", "both_hit3_strict"])
        if "both_hit3_strict" in df else rate([]),
    }
    if "numcheck_bad" in df and df["numcheck_bad"].notna().any():
        out["numcheck_bad_ratio"] = rate(df["numcheck_bad"])
    if "seconds" in df and df["seconds"].notna().any():
        s = df["seconds"].dropna()
        out["seconds_per_scenario"] = {"mean": float(s.mean()), "max": float(s.max()), "n": int(len(s))}
    if "checks_requested" in df and df["checks_requested"].notna().any():
        out["check_effect"] = check_effect(df)
    for col in ("requests", "input_tokens", "output_tokens"):
        if col in df and df[col].notna().any():
            out[f"{col}_per_scenario"] = {"mean": float(df[col].dropna().mean())}
    return out


def check_effect(df: pd.DataFrame) -> dict:
    """추가 확인 효과: 요청 비율, 그중 1순위가 바뀐 비율, 바뀐 경우 잠정 → 최종 적중(원인 시나리오)과
    오경보(원인 없는 시나리오)."""
    req = df[df["checks_requested"].astype("boolean").fillna(False)]
    changed = req[req["top_changed"].astype("boolean").fillna(False)]
    cause = ~changed["fault_code"].isin(NO_CAUSE_CODES)
    return {
        "requested": rate(df["checks_requested"]),
        "top_changed_given_requested": rate(req["top_changed"]),
        "changed_hit3_strict_prelim": rate(changed.loc[cause, "prelim_hit3_strict"]),
        "changed_hit3_strict_final": rate(changed.loc[cause, "hit3_strict"]),
        "changed_false_alarm_prelim": rate(changed.loc[~cause, "prelim_false_alarm"]),
        "changed_false_alarm_final": rate(changed.loc[~cause, "false_alarm"]),
    }


def calibration(df: pd.DataFrame) -> list[dict]:
    """원인 시나리오에서 1순위 신뢰도 구간 5개별 실제 Hit@1 (엄격)."""
    c = df[_is_cause(df) & df["top_conf"].notna() & ~df["missed"].astype(bool)]
    out = []
    for lo, hi in zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:]):
        m = (c["top_conf"] >= lo) & (c["top_conf"] < hi)
        r = rate(c.loc[m, "hit1_strict"])
        r.update({"bin": [lo, min(hi, 1.0)],
                  "mean_conf": float(c.loc[m, "top_conf"].mean()) if m.any() else None})
        out.append(r)
    return out


def breakdown(df: pd.DataFrame, by: str) -> dict:
    out = {}
    for key, g in df.groupby(by, dropna=False):
        key = "none" if key is None or (isinstance(key, float) and np.isnan(key)) else str(key)
        cause = _is_cause(g)
        entry = {"n": int(len(g))}
        if cause.any():
            gc = g[cause]
            entry.update({
                "hit1_strict": rate(gc["hit1_strict"]), "hit3_strict": rate(gc["hit3_strict"]),
                "hit3_loose": rate(gc["hit3_loose"]), "miss_rate": rate(gc["missed"]),
            })
            if "both_hit3_strict" in gc and gc["both_hit3_strict"].notna().any():
                entry["both_hit3_strict"] = rate(gc["both_hit3_strict"])
        if (~cause).any():
            entry["false_alarm"] = rate(g.loc[~cause, "false_alarm"])
        out[key] = entry
    return out


# ---------------------------------------------------------------- 우연 수준

def _pool(fab: dict, kind: str) -> list[Hypothesis]:
    """정답 유형과 같은 수준의 모든 후보 (무작위 추측용)."""
    ev = [{"evidence_id": "random", "summary": ""}] * 2
    base = {"confidence": 0.5, "evidence": ev, "falsification_test": "", "recommended_action": ""}
    pool = []
    for s in fab["steps"]:
        sid = s["step_id"]
        if kind == "F1":
            for t in s["tools"]:
                if len(t["chambers"]) > 1:
                    pool += [Hypothesis(rank=1, entity_type="chamber", step_id=sid, tool_id=t["tool_id"],
                                        chamber_id=c, **base) for c in t["chambers"]]
        elif kind == "F2":
            if s["product_specific_recipe"]:
                pool += [Hypothesis(rank=1, entity_type="tool_recipe", step_id=sid, tool_id=t["tool_id"],
                                    recipe_id=r, **base) for t in s["tools"] for r in s["recipes"]]
        elif kind == "F3":
            pool += [Hypothesis(rank=1, entity_type="tool", step_id=sid, tool_id=t["tool_id"], **base)
                     for t in s["tools"]]
        elif kind == "F5":
            pool += [Hypothesis(rank=1, entity_type="recipe", step_id=sid, recipe_id=r, **base)
                     for r in s["recipes"]]
    return pool


def chance_hit3(fab: dict, gt: dict, n_draws: int = 2000, seed: int = 0) -> dict:
    """정답과 같은 수준의 후보에서 무작위로 3개를 골랐을 때의 Hit@3 기대값 (엄격·느슨)."""
    from excursion_tracer.agent.schema import Report

    kinds = sorted({f["type"] for f in gt["faults"]})
    pool = [h for k in kinds for h in _pool(fab, k)]
    rng = np.random.default_rng(seed)
    strict = loose = 0
    for _ in range(n_draws):
        idx = rng.choice(len(pool), size=min(TOP_K, len(pool)), replace=False)
        hyps = [pool[i].model_copy(update={"rank": r + 1}) for r, i in enumerate(idx)]
        rep = Report(scenario_id=gt["scenario_id"], method="chance", verdict="cause_found",
                     hypotheses=hyps)
        s = score(rep, gt, tau=0.5)
        strict += bool(s["hit3_strict"])
        loose += bool(s["hit3_loose"])
    return {"hit3_strict": strict / n_draws, "hit3_loose": loose / n_draws, "pool_size": len(pool)}


def chance_level(scenario_dirs: list[Path], gts: dict[str, dict], n_draws: int = 2000) -> dict:
    per = []
    for d in scenario_dirs:
        gt = gts[d.name]
        if gt["fault_code"] in NO_CAUSE_CODES:
            continue
        fab = json.loads((d / "fab.json").read_text(encoding="utf-8"))
        per.append(chance_hit3(fab, gt, n_draws=n_draws))
    return {
        "hit3_strict": float(np.mean([p["hit3_strict"] for p in per])) if per else None,
        "hit3_loose": float(np.mean([p["hit3_loose"] for p in per])) if per else None,
        "n_cause": len(per),
        "n_draws": n_draws,
    }


# ---------------------------------------------------------------- 반복성

def repeatability(runs_by_scenario: dict[str, list]) -> dict:
    """같은 시나리오를 여러 번 실행했을 때 판정과 1순위 가설이 모두 같은 비율."""
    def key(rep):
        if rep is None:
            return ("invalid",)
        if rep.verdict == "no_equipment_cause" or not rep.hypotheses:
            return (rep.verdict,)
        h = min(rep.hypotheses, key=lambda x: x.rank)
        return (rep.verdict, h.entity_type, h.step_id, h.tool_id, h.chamber_id, h.recipe_id)

    same = [len({key(r) for r in reps}) == 1 for reps in runs_by_scenario.values() if len(reps) > 1]
    return rate(same)
