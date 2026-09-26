"""숫자 대조 검사: 보고서 문장 속 숫자가 근거 묶음·확인 결과에 있는 숫자인지 확인한다.

- 대상 필드: 근거 summary, falsification_test, recommended_action, confounding_notes, no_cause_explanation
- 허용: 상대 오차 1% 이내, 또는 근거 숫자를 보고서 숫자의 자릿수로 반올림한 값과 같음,
  또는 백분율 표기(21% ↔ 0.210)
- 제외: 1자리 정수, ID 속 숫자(S12-T3, scn_1001, E7, C2 등), 날짜·시각
"""

from __future__ import annotations

import re

from excursion_tracer.agent.schema import Report

ID_RE = re.compile(r"\b(?:[A-Za-z]+_\d+|[A-Za-z]{1,3}\d+(?:-[A-Za-z]\d+)*|R\d)\b")
MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
DATE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?|\b\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\b"
    rf"|\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}(?:\s+\d{{4}})?\b"
    r"|\d{1,2}월\s*\d{1,2}일"
)
NUM_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+|\.\d+|\d+)(?:[eE][-+]?\d+)?%?")


def _strip(text: str) -> str:
    return ID_RE.sub(" ", DATE_RE.sub(" ", text))


def extract_numbers(text: str) -> list[str]:
    return NUM_RE.findall(_strip(text))


def _value(tok: str) -> tuple[float, bool]:
    pct = tok.endswith("%")
    return float(tok.rstrip("%")), pct


def _decimals(tok: str) -> int | None:
    t = tok.rstrip("%").lstrip("+-")
    if "e" in t.lower():
        return None
    return len(t.split(".")[1]) if "." in t else 0


def source_numbers(texts: list[str]) -> list[float]:
    out = []
    for t in texts:
        for tok in extract_numbers(t):
            out.append(abs(_value(tok)[0]))
    return out


def supported(tok: str, sources: list[float]) -> bool:
    v, pct = _value(tok)
    v = abs(v)
    cands = [v, v / 100.0] if pct else [v]
    dec = _decimals(tok)
    for c in cands:
        for s in sources:
            if s == c or (s != 0 and abs(c - s) <= 0.01 * abs(s)):
                return True
            if dec is not None:
                d = dec + (2 if pct and c == v / 100.0 else 0)
                if round(s, d) == round(c, d):
                    return True
    return False


def report_texts(report: Report) -> list[str]:
    texts = [report.confounding_notes, report.no_cause_explanation]
    for h in report.hypotheses:
        texts += [e.summary for e in h.evidence]
        texts += [h.falsification_test, h.recommended_action]
    return [t for t in texts if t]


def check_report(report: Report, source_texts: list[str]) -> dict:
    """근거 없는 숫자 목록. 1자리 정수는 세지 않는다."""
    sources = source_numbers(source_texts)
    bad = []
    for text in report_texts(report):
        for tok in extract_numbers(text):
            core = tok.rstrip("%").lstrip("+-")
            if re.fullmatch(r"\d", core):
                continue
            if not supported(tok, sources):
                bad.append(tok)
    return {"unsupported": bad, "bad": bool(bad)}
