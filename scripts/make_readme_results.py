"""README.md의 결과 구간을 results/summary.json에서 다시 만든다.

    python scripts/make_readme_results.py

README.md의 <!-- results:start --> 와 <!-- results:end --> 사이를 바꾼다.
결과 숫자는 이 스크립트가 summary.json에서 읽어 쓴 것만 README에 들어간다.
"""

from __future__ import annotations

import json
import re

from excursion_tracer.config import PROJECT_ROOT

README = PROJECT_ROOT / "README.md"
SUMMARY = PROJECT_ROOT / "results" / "summary.json"
START, END = "<!-- results:start -->", "<!-- results:end -->"
CAUSE = ["F1", "F2", "F3", "F4", "F5"]
FAULT_NAME = {"F0a": "원인 없음 (우연)", "F0b": "원인 없음 (제품 구성 변화)", "F1": "정비 후 챔버 열화",
              "F2": "설비×레시피 조합", "F3": "설비 드리프트", "F4": "두 원인 동시",
              "F5": "레시피 버전 변경 (test 전용)"}


def r(x: dict | None) -> str:
    if not x or x.get("value") is None:
        return "-"
    lo, hi = x["ci95"]
    return f"{x['k']}/{x['n']} = {x['value']:.3f} [{lo:.3f}, {hi:.3f}]"


def block(s: dict) -> str:
    t = s["test"]
    a, b, ch, info = t["agent"], t["baseline"], t["chance"], t["info"]
    key = "baseline_vs_agent" if "baseline_vs_agent" in t["compare"] else "agent_vs_baseline"
    mc = t["compare"][key]["hit3_strict_mcnemar"]
    first, second = ("기준선", "에이전트") if key == "baseline_vs_agent" else ("에이전트", "기준선")
    mix = ", ".join(f"{k} {v}" for k, v in info["fault_mix"].items())
    L = [f"test 세트 {info['n_scenarios']}개 ({mix}). 원인 시나리오 {a['n_cause']}개, "
         f"원인 없는 시나리오 {a['n_no_cause']}개. 비율 옆 괄호는 95% Wilson 신뢰구간이다.", "",
         "| 지표 | LLM 에이전트 | 통계 기준선 |", "|---|---|---|"]
    rows = [("Hit@1 (엄격)", "hit1_strict"), ("Hit@3 (엄격)", "hit3_strict"),
            ("Hit@1 (느슨)", "hit1_loose"), ("Hit@3 (느슨)", "hit3_loose"),
            ("오경보율 (원인 없는 시나리오)", "false_alarm"), ("미탐율", "miss_rate"),
            ("시작 시각 정확도 (±1일)", "onset_accuracy"), ("제품 구성 변화 설명 (F0b)", "f0b_explained"),
            ("형식 실패율", "invalid_rate")]
    for label, key in rows:
        L.append(f"| {label} | {r(a.get(key))} | {r(b.get(key))} |")
    L.append(f"| MRR (엄격) | {a['mrr_strict']['value']:.3f} | {b['mrr_strict']['value']:.3f} |")
    L.append(f"| 근거 없는 숫자가 있는 보고서 비율 | {r(a.get('numcheck_bad_ratio'))} | - |")
    rep = a.get("repeatability")
    if rep:
        L.append(f"| 반복성 ({rep['n']}개 × {rep['runs_per_scenario']}회, 판정·1순위 일치) | {r(rep)} | - |")
    L.append(f"| 시나리오당 LLM 요청 수 (평균) | {a['requests_per_scenario']['mean']:.2f} | 0 |")
    L.append(f"| 시나리오당 처리 시간 (평균, 초) | {a['seconds_per_scenario']['mean']:.2f} | "
             f"{b['seconds_per_scenario']['mean']:.2f} |")
    L += ["", f"- 우연 수준 (정답과 같은 수준의 후보에서 무작위 3개): Hit@3 엄격 {ch['hit3_strict']:.4f}, "
              f"느슨 {ch['hit3_loose']:.4f}",
          f"- McNemar (Hit@3 엄격, 같은 원인 시나리오 {mc['n']}쌍): 둘 다 적중 {mc['both']}, {first}만 {mc['only_a']}, "
          f"{second}만 {mc['only_b']}, 둘 다 실패 {mc['neither']}, p = {mc['p']:.3f}",
          f"- 알림이 없어 다시 만든 시나리오 수: {info['regen_total']}", "",
          "원인 유형별 Hit@3 (엄격) / 오경보율", "",
          "| 유형 | 내용 | LLM 에이전트 | 통계 기준선 |", "|---|---|---|---|"]
    for code, e in a["by_fault"].items():
        key = "hit3_strict" if code in CAUSE else "false_alarm"
        suffix = "" if code in CAUSE else " (오경보)"
        L.append(f"| {code} | {FAULT_NAME.get(code, code)}{suffix} | {r(e.get(key))} | "
                 f"{r(b['by_fault'][code].get(key))} |")
    d = s.get("dev", {})
    if "agent_r1" in d and "agent_r2" in d:
        L += ["", f"프롬프트는 dev 세트({d['info']['n_scenarios']}개)로만 두 라운드 다듬은 뒤 고정했고, "
                  "그 뒤에 test 세트를 만들었다.", "",
              "| dev 라운드 | Hit@3 (엄격) | 오경보율 |", "|---|---|---|",
              f"| 1 | {r(d['agent_r1']['hit3_strict'])} | {r(d['agent_r1']['false_alarm'])} |",
              f"| 2 (고정) | {r(d['agent_r2']['hit3_strict'])} | {r(d['agent_r2']['false_alarm'])} |"]
    return "\n".join(L)


def main() -> int:
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    text = README.read_text(encoding="utf-8")
    new = re.sub(rf"{re.escape(START)}.*?{re.escape(END)}", f"{START}\n{block(s)}\n{END}", text, flags=re.S)
    README.write_text(new, encoding="utf-8")
    print(f"갱신: {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
