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
CAUSE = ["F1", "F2", "F3", "F4", "F5", "F6"]
FAULT_NAME = {"F0a": "원인 없음 (우연)", "F0b": "원인 없음 (제품 구성 변화)", "F1": "정비 후 챔버 열화",
              "F2": "설비×레시피 조합", "F3": "설비 드리프트", "F4": "두 원인 동시",
              "F5": "레시피 버전 변경", "F6": "간헐적 챔버 이상"}
LABEL = {"agent": "에이전트 v1", "baseline": "기준선 v1", "agent_v2": "에이전트 v2", "baseline_v2": "기준선 v2"}


def r(x: dict | None) -> str:
    if not x or x.get("value") is None:
        return "-"
    lo, hi = x["ci95"]
    return f"{x['k']}/{x['n']} = {x['value']:.3f} [{lo:.3f}, {hi:.3f}]"


def metrics_table(s: dict, methods: list[str]) -> list[str]:
    ms = [m for m in methods if m in s]
    L = ["| 지표 | " + " | ".join(LABEL[m] for m in ms) + " |", "|---|" + "---|" * len(ms)]
    rows = [("Hit@1 (엄격)", "hit1_strict"), ("Hit@3 (엄격)", "hit3_strict"), ("Hit@3 (느슨)", "hit3_loose"),
            ("F4 두 원인 모두 상위 3개 (엄격)", "f4_both_hit3_strict"),
            ("오경보율 (원인 없는 시나리오)", "false_alarm"), ("미탐율", "miss_rate"),
            ("시작 시각 정확도 (±1일)", "onset_accuracy"), ("제품 구성 변화 설명 (F0b)", "f0b_explained"),
            ("형식 실패율", "invalid_rate"), ("근거 없는 숫자가 있는 보고서", "numcheck_bad_ratio")]
    for label, key in rows:
        L.append(f"| {label} | " + " | ".join(r(s[m].get(key)) for m in ms) + " |")
    L.append("| MRR (엄격) | " + " | ".join(f"{s[m]['mrr_strict']['value']:.3f}" for m in ms) + " |")
    L.append("| 반복성 (판정·1순위 일치) | " + " | ".join(
        (f"{r(s[m]['repeatability'])} ({s[m]['repeatability']['runs_per_scenario']}회 실행)"
         if s[m].get("repeatability") else "-") for m in ms) + " |")
    L.append("| 시나리오당 LLM 요청 수 | " + " | ".join(
        (f"{s[m]['requests_per_scenario']['mean']:.2f}" if s[m].get("requests_per_scenario") else "0") for m in ms) + " |")
    L.append("| 시나리오당 처리 시간 (초) | " + " | ".join(
        (f"{s[m]['seconds_per_scenario']['mean']:.2f}" if s[m].get("seconds_per_scenario") else "-") for m in ms) + " |")
    return L


def fault_table(s: dict, methods: list[str]) -> list[str]:
    ms = [m for m in methods if m in s]
    L = ["| 유형 | 내용 | " + " | ".join(LABEL[m] for m in ms) + " |", "|---|---|" + "---|" * len(ms)]
    for code in s[ms[0]]["by_fault"]:
        key = "hit3_strict" if code in CAUSE else "false_alarm"
        suffix = "" if code in CAUSE else " (오경보)"
        L.append(f"| {code} | {FAULT_NAME.get(code, code)}{suffix} | " +
                 " | ".join(r(s[m]["by_fault"].get(code, {}).get(key)) for m in ms) + " |")
    return L


def compare_lines(s: dict) -> list[str]:
    out = []
    for key, v in s["compare"].items():
        if not isinstance(v, dict):
            continue
        a, b = key.split("_vs_")
        if a not in LABEL or b not in LABEL:
            continue
        mc = v["hit3_strict_mcnemar"]
        out.append(f"- McNemar (Hit@3 엄격, 같은 원인 시나리오 {mc['n']}쌍) {LABEL[a]} 대 {LABEL[b]}: 둘 다 적중 "
                   f"{mc['both']}, {LABEL[a]}만 {mc['only_a']}, {LABEL[b]}만 {mc['only_b']}, 둘 다 실패 "
                   f"{mc['neither']}, p = {mc['p']:.3f}")
    return out


def block(s: dict) -> str:
    L = []
    if "test2" in s:
        t = s["test2"]
        info = t["info"]
        mix = ", ".join(f"{k} {v}" for k, v in info["fault_mix"].items())
        L += ["### 2차: 새 시험 세트 test2", "",
              f"test2 {info['n_scenarios']}개 ({mix}). 2차 프롬프트를 고정한 뒤 새 seed로 만들었고, "
              "개발 중 없던 원인 유형 F6를 넣었다. 비율 옆 괄호는 95% Wilson 신뢰구간이다.", ""]
        L += metrics_table(t, ["agent_v2", "baseline_v2", "baseline"])
        L += [""] + compare_lines(t)
        ch = t["chance"]
        L.append(f"- 우연 수준 (정답과 같은 수준의 후보에서 무작위 3개): Hit@3 엄격 {ch['hit3_strict']:.4f}, "
                 f"느슨 {ch['hit3_loose']:.4f}")
        a = t.get("agent_v2", {})
        ce = a.get("check_effect")
        if ce:
            L.append(f"- 반증 확인: 요청 {r(ce['requested'])}, 1순위가 바뀐 비율 {r(ce['top_changed_given_requested'])}; "
                     f"바뀐 원인 시나리오의 Hit@3 엄격 잠정 {r(ce['changed_hit3_strict_prelim'])} → 최종 "
                     f"{r(ce['changed_hit3_strict_final'])}")
        ic = a.get("id_check")
        if ic:
            L.append(f"- ID 검증: 정규화한 시나리오 {r(ic['normalized_scenarios'])}, ID 재요청 {r(ic['retry'])}, "
                     f"해석 안 된 ID가 남은 보고서 {r(ic['unresolved_after_retry'])}")
        if a.get("usage"):
            L.append(f"- 에이전트 v2 LLM 요청 {a['usage']['logged_requests']}회, 비용 {a['usage']['cost_usd_total']:.2f}달러")
        L += ["", "원인 유형별 Hit@3 (엄격) / 오경보율 (test2)", ""]
        L += fault_table(t, ["agent_v2", "baseline_v2", "baseline"])
    if "test" in s:
        t = s["test"]
        L += ["", "### 1차: 시험 세트 test1", "",
              f"test1 {t['info']['n_scenarios']}개. 1차 프롬프트를 고정한 뒤 만들었다. 1차 결과를 본 뒤 F4의 적중 정의를 "
              "통일해(두 원인 중 하나 이상 적중) 두 방법 모두 다시 채점했다.", ""]
        L += metrics_table(t, ["agent", "baseline"])
        L += [""] + compare_lines(t)
        L += ["", "원인 유형별 Hit@3 (엄격) / 오경보율 (test1)", ""]
        L += fault_table(t, ["agent", "baseline"])
    d = s.get("dev", {})
    rounds = [(m, lab) for m, lab in (("agent_r1", "에이전트 v1 1라운드"), ("agent_r2", "에이전트 v1 2라운드 (고정)"),
                                      ("agent_v2_r1", "에이전트 v2 1라운드"), ("agent_v2_r2", "에이전트 v2 2라운드 (고정)"))
              if m in d]
    if rounds:
        L += ["", f"### 프롬프트 개선 라운드 (dev {d['info']['n_scenarios']}개, 시험 세트와 별도)", "",
              "| 라운드 | Hit@3 (엄격) | 오경보율 |", "|---|---|---|"]
        L += [f"| {lab} | {r(d[m]['hit3_strict'])} | {r(d[m]['false_alarm'])} |" for m, lab in rounds]
    return "\n".join(L)


def main() -> int:
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    text = README.read_text(encoding="utf-8")
    new = re.sub(rf"{re.escape(START)}.*?{re.escape(END)}", lambda _: f"{START}\n{block(s)}\n{END}", text,
                 flags=re.S)
    README.write_text(new, encoding="utf-8")
    print(f"갱신: {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
