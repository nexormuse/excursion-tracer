"""보고서와 정답 대조 규칙.

| 원인 | 엄격 적중 | 느슨한 적중 |
| F1 | 챔버 일치 | 같은 설비 |
| F2 | 설비와 레시피 모두 일치 | 설비 일치, 또는 같은 단계에서 레시피 일치 |
| F3 | 설비 일치 | 같은 단계 |
| F4 | 두 원인 중 하나 이상이 상위 k개 안에서 엄격 적중 | 하나 이상이 상위 k개 안에서 느슨한 적중 |
| F5 | 레시피 일치 | 같은 단계 |
| F0a·F0b | no_equipment_cause, 또는 1순위 신뢰도 < τ | 동일 |

F4는 보조 지표로 두 원인 모두 상위 3개 안에서 엄격 적중했는지(both_hit3_strict)를 따로 본다.
순위(MRR용)는 첫 적중 순위다. 이 정의로 모든 유형에서 Hit@1 ≤ Hit@3이다.
가설의 설비는 tool_id, 비어 있으면 chamber_id에서 얻는다.
판정이 no_equipment_cause인 보고서는 원인 시나리오에서 가설이 없는 것으로 본다.
"""

from __future__ import annotations

import re

import pandas as pd

from excursion_tracer.agent.schema import Hypothesis, Report

NO_CAUSE_CODES = ("F0a", "F0b")
TOP_K = 3
PRODUCT_MIX_RE = re.compile(r"product[\s_-]*(mix|share|composition|ratio)|제품\s*(구성|비중|비율)", re.I)


def hyp_tool(h: Hypothesis) -> str:
    if h.tool_id:
        return h.tool_id
    if h.chamber_id:
        return h.chamber_id.rsplit("-", 1)[0]
    return ""


def match_fault(h: Hypothesis, f: dict) -> tuple[bool, bool]:
    """(엄격 적중, 느슨한 적중)."""
    kind = f["type"]
    tool = hyp_tool(h)
    same_step = h.step_id == f["step_id"]
    if kind == "F1":
        strict = bool(h.chamber_id) and h.chamber_id == f["chamber_id"]
        return strict, strict or tool == f["tool_id"]
    if kind == "F2":
        tool_ok = tool == f["tool_id"]
        recipe_ok = bool(h.recipe_id) and h.recipe_id == f["recipe_id"]
        strict = tool_ok and recipe_ok
        return strict, strict or tool_ok or (same_step and recipe_ok)
    if kind == "F3":
        strict = tool == f["tool_id"]
        return strict, strict or same_step
    if kind == "F5":
        strict = bool(h.recipe_id) and h.recipe_id == f["recipe_id"]
        return strict, strict or same_step
    raise ValueError(f"대조 규칙이 없는 원인 유형: {kind}")


def _first_rank(hyps: list[Hypothesis], f: dict, strict: bool) -> int | None:
    for i, h in enumerate(hyps):
        s, l = match_fault(h, f)
        if (s if strict else l):
            return i + 1
    return None


def _onset_ok(h: Hypothesis, f: dict, tol_days: float) -> bool:
    if not h.onset:
        return False
    try:
        pred = pd.Timestamp(h.onset).tz_localize(None)
    except (ValueError, TypeError):
        return False
    return abs((pred - pd.Timestamp(f["onset_ts"])) / pd.Timedelta(days=1)) <= tol_days


def score(report: Report | None, gt: dict, tau: float, onset_tol_days: float = 1) -> dict:
    """시나리오 하나의 채점 결과. report가 None이면 형식 실패(invalid)로 오답 처리한다."""
    code = gt["fault_code"]
    cause = code not in NO_CAUSE_CODES
    out = {
        "invalid": report is None,
        "verdict": "" if report is None else report.verdict,
        "top_conf": None,
        "rank_strict": None, "rank_loose": None,
        "hit1_strict": None, "hit3_strict": None, "hit1_loose": None, "hit3_loose": None,
        "false_alarm": None, "missed": None, "onset_ok": None, "f0b_explained": None,
        "no_cause_correct": None, "both_hit3_strict": None,
    }
    hyps = [] if report is None else sorted(report.hypotheses, key=lambda h: h.rank)
    if hyps:
        out["top_conf"] = hyps[0].confidence

    if not cause:
        found = report is not None and report.verdict == "cause_found" and bool(hyps)
        fa = found and hyps[0].confidence >= tau
        out["false_alarm"] = bool(fa) if report is not None else True
        out["no_cause_correct"] = not out["false_alarm"]
        if code == "F0b":
            text = "" if report is None else f"{report.no_cause_explanation} {report.limitations}"
            out["f0b_explained"] = bool(PRODUCT_MIX_RE.search(text))
        return out

    out["missed"] = report is None or report.verdict == "no_equipment_cause"
    if out["missed"]:
        hyps = []
    top = hyps[:TOP_K]
    faults = gt["faults"]
    s_ranks = [_first_rank(top, f, True) for f in faults]
    l_ranks = [_first_rank(top, f, False) for f in faults]

    found_s = [r for r in s_ranks if r is not None]
    found_l = [r for r in l_ranks if r is not None]
    out["rank_strict"] = min(found_s) if found_s else None
    out["rank_loose"] = min(found_l) if found_l else None
    out["hit3_strict"] = bool(found_s)
    out["hit3_loose"] = bool(found_l)
    out["hit1_strict"] = out["rank_strict"] == 1
    out["hit1_loose"] = out["rank_loose"] == 1
    if len(faults) > 1:
        out["both_hit3_strict"] = all(r is not None for r in s_ranks)

    onset = []
    for f, r in zip(faults, s_ranks):
        if f["type"] in ("F1", "F5") and r is not None:
            onset.append(_onset_ok(top[r - 1], f, onset_tol_days))
    out["onset_ok"] = all(onset) if onset else None
    return out
