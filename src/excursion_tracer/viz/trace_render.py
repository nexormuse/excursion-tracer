"""대표 시나리오 추적 기록 (마크다운).

순서: 알림 → 도구 호출 순서와 각 결과의 핵심 숫자(근거 묶음·추가 확인 원문) → LLM 호출 →
최종 보고서 → (있으면) 한국어 재실행 보고서 → 정답 공개와 채점.
숫자는 실행 기록(runs/)의 원문과 results/per_scenario.csv에서만 옮긴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from excursion_tracer.config import PROJECT_ROOT
from excursion_tracer.eval.ground_truth import load_ground_truth

# 근거 묶음의 절 제목 → 그 절을 만든 통계 도구
SECTION_TOOL = [
    ("Alert", "alert.json 읽기"),
    ("Fab overview", "fab.json 읽기"),
    ("Commonality candidates", "commonality_scan (tool·chamber·recipe, 제품 층화 + 전체 BH)"),
    ("Change points", "change_point (상위 후보, 순열검정)"),
    ("Events within", "get_events (변화점 ±2일)"),
    ("Confounding", "confounding_check (파이 계수, 층화 비교)"),
    ("Tool x recipe", "interaction_test (설비×레시피 분산분석)"),
    ("Product mix", "product_mix (일별 제품 비중)"),
]
MAX_LINES = 6


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _cell(v) -> str:
    return "-" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _sections(pack: str) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for line in pack.splitlines():
        if line.startswith("## "):
            out.append((line[3:], []))
        elif out:
            out[-1][1].append(line)
    return out


def _entity(h: dict) -> str:
    if h["entity_type"] == "tool_recipe":
        return f"{h['tool_id']} × {h['recipe_id']}"
    return h["chamber_id"] or h["tool_id"] or h["recipe_id"]


def _report_md(fr: dict) -> list[str]:
    out = [f"- 판정: `{fr['verdict']}`"]
    for h in fr["hypotheses"]:
        out.append(f"- **{h['rank']}순위** {h['entity_type']} `{_entity(h)}` (단계 {h['step_id']}), "
                   f"신뢰도 {h['confidence']}, 시작 시각 {h['onset'] or '-'}")
        for e in h["evidence"]:
            out.append(f"  - 근거 {e['evidence_id']}: {e['summary']}")
        out.append(f"  - 반증 방법: {h['falsification_test']}")
        out.append(f"  - 권고 조치: {h['recommended_action']}")
    for key, label in (("no_cause_explanation", "원인 없음 설명"), ("confounding_notes", "교란 메모"),
                       ("limitations", "한계")):
        if fr[key]:
            out.append(f"- {label}: {fr[key]}")
    return out


def render_trace(scenario_dir: Path, run_record: Path, per: pd.DataFrame, reason: str,
                 showcase_record: Path | None = None) -> str:
    rec = json.loads(run_record.read_text(encoding="utf-8"))
    sid = rec["scenario_id"]
    alert = json.loads((scenario_dir / "alert.json").read_text(encoding="utf-8"))
    L = [f"# 추적 기록: {sid}", "", f"- 선정 이유: {reason}",
         f"- 실행: `{rel(run_record)}` (모델 {rec['model']}, 프롬프트 해시 {rec['prompt_hash'][:8]})", ""]

    L += ["## 1. 알림", "",
          f"- 수준 {alert['level']}, 구간 {alert['window_start']} ~ {alert['window_end']}, "
          f"저수율 기준 {alert['baseline']['low_yield_threshold']}"]
    for p, v in alert["by_product"].items():
        L.append(f"- {p}: 구간 평균 수율 {v['mean_yield']}, 기준 구간 평균 {v['baseline_mean']}, "
                 f"웨이퍼 {v['n_wafers']}, 저수율 {v['n_low']}")
    L.append("")

    L += ["## 2. 도구 호출 순서와 핵심 결과", "",
          "통계 코드가 아래 순서로 도구를 실행해 근거 묶음을 만들었다 (줄 앞 E#은 근거 ID). "
          f"절마다 앞의 {MAX_LINES}줄까지 옮긴다.", ""]
    step = 0
    for title, lines in _sections(rec["evidence_pack"]):
        tool = next((t for k, t in SECTION_TOOL if title.startswith(k)), title)
        step += 1
        L.append(f"**{step}. {tool}** — {title}")
        L.append("")
        L += [f"    {l}" for l in lines[:MAX_LINES]]
        if len(lines) > MAX_LINES:
            L.append(f"    … (이 절의 나머지 {len(lines) - MAX_LINES}줄 생략)")
        L.append("")

    r1 = rec["round1"]
    step += 1
    tok_in = sum(a["input_tokens"] for a in r1["attempts"])
    tok_out = sum(a["output_tokens"] for a in r1["attempts"])
    L.append(f"**{step}. LLM 1회차** — 요청 {len(r1['attempts'])}회 (스키마 재요청 포함), "
             f"입력 토큰 {tok_in}, 출력 토큰 {tok_out}, 추가 확인 요청 {rec['needs_checks']}")
    for a in r1["attempts"]:
        if a["error"]:
            L.append(f"  - 스키마 검증 실패: {a['error'].splitlines()[1].strip() if len(a['error'].splitlines()) > 1 else a['error'][:120]}")
    L.append("")
    if rec["checks"]:
        for c in rec["checks"]:
            step += 1
            L.append(f"**{step}. 추가 확인 {c['check_id']} {c['request']['name']}** — 이유: {c['request']['reason']}")
            L.append(f"    {c['result']}")
            L.append("")
        step += 1
        r2 = rec["round2"]
        L.append(f"**{step}. LLM 2회차** — 요청 {len(r2['attempts'])}회")
        L.append("")
    else:
        L.append("추가 확인 요청이 없어 1회차 보고서가 최종 보고서다.")
        L.append("")

    L += ["## 3. 최종 보고서", ""] + _report_md(rec["final_report"])
    nc = rec.get("numcheck") or {}
    L.append(f"- 숫자 대조: 근거 없는 숫자 {nc.get('unsupported', [])}")
    L.append("")

    if showcase_record and showcase_record.is_file():
        sc = json.loads(showcase_record.read_text(encoding="utf-8"))
        L += [f"## 4. 한국어 재실행 보고서 (`{rel(showcase_record)}`)", ""]
        L += _report_md(sc["final_report"]) if sc["final_report"] else ["- 형식 실패"]
        L.append(f"- 숫자 대조: 근거 없는 숫자 {(sc.get('numcheck') or {}).get('unsupported', [])}")
        L.append("")

    gt = load_ground_truth(scenario_dir)
    row = per[(per["scenario_id"] == sid) & (per["method"] == "agent")].iloc[0]
    base = per[(per["scenario_id"] == sid) & (per["method"] == "baseline")]
    L += ["## 5. 정답 공개", "",
          f"- 원인 유형 {gt['fault_code']}, 효과 크기 {gt['cell']['effect'] or '-'}, "
          f"stickiness {gt['cell']['stickiness']}, 재생성 {gt['regen_count']}회"]
    for f in gt["faults"]:
        ids = ", ".join(x for x in (f.get("chamber_id"), f.get("tool_id"), f.get("recipe_id")) if x)
        L.append(f"- {f['type']}: 단계 {f['step_id']}, {ids}, 시작 {f['onset_ts']}, δ {f['delta']}")
    if not gt["faults"]:
        L.append("- 설비 원인 없음")
    L.append("")
    L.append("| 방법 | 판정 | Hit@1 엄격 | Hit@3 엄격 | Hit@3 느슨 | 엄격 순위 | 오경보 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, r in [("agent", row)] + ([("baseline", base.iloc[0])] if len(base) else []):
        L.append(f"| {name} | {r['verdict']} | {_cell(r['hit1_strict'])} | {_cell(r['hit3_strict'])} | "
                 f"{_cell(r['hit3_loose'])} | {'-' if pd.isna(r['rank_strict']) else int(r['rank_strict'])} | "
                 f"{_cell(r['false_alarm'])} |")
    L.append("")
    return "\n".join(L)
