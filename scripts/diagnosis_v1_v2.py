"""1차 진단 D-1~D-6이 2차에서 어떻게 됐는지 숫자로 정리한다 (1차 = 에이전트 v1·기준선 v1, test1 /
2차 = 에이전트 v2·기준선 v2, test2).

    python scripts/diagnosis_v1_v2.py --run1 runs/<v1 run_id> --run2 runs/<v2 run_id>

원천: results/summary.json, results/per_scenario.csv, results/v1_id_check.json, 실행 기록(runs/).
결과: results/diagnosis_v1_v2.json, 그리고 results/failure_analysis_v2.md 안의
<!-- diagnosis:start --> ~ <!-- diagnosis:end --> 구간에 표로 쓴다.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from excursion_tracer.config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"


def rate(x: dict | None) -> dict | None:
    if not x or x.get("value") is None:
        return None
    return {"k": x["k"], "n": x["n"], "value": x["value"], "ci95": x["ci95"]}


def check_errors(run: Path, prefix: str) -> dict:
    """추가 확인 요청 중 필드가 비거나 잘못돼 오류 메시지로 돌아간 것."""
    from collections import Counter

    total = err = 0
    scen, kinds = set(), Counter()
    for p in sorted(run.glob(f"{prefix}*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        for c in rec.get("checks", []):
            total += 1
            if c["result"].startswith("error"):
                err += 1
                scen.add(rec["scenario_id"])
                kinds[c["request"]["name"]] += 1
    return {"requests": total, "errors": err, "scenarios_with_error": len(scen), "errors_by_check": dict(kinds)}


def run_stats(run: Path, prefix: str) -> dict:
    """실행 기록에서 스키마 재요청, 출력 한도에서 끊긴 응답(스키마 실패 원문이 JSON 끝을 못 맺은 것) 수."""
    n = retry = eof = 0
    for p in sorted(run.glob(f"{prefix}*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        n += 1
        attempts = (rec.get("round1") or {}).get("attempts", []) + ((rec.get("round2") or {}).get("attempts", []))
        errs = [a for a in attempts if a.get("error")]
        retry += bool(errs)
        eof += sum("EOF while parsing" in a["error"] for a in errs)
    return {"scenarios": n, "scenarios_with_schema_retry": retry, "responses_cut_at_output_limit": eof}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run1", type=Path, required=True)
    ap.add_argument("--run2", type=Path, required=True)
    args = ap.parse_args(argv)

    s = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    per = pd.read_csv(RESULTS / "per_scenario.csv")
    idc = json.loads((RESULTS / "v1_id_check.json").read_text(encoding="utf-8"))
    a1, b1 = s["test"]["agent"], s["test"]["baseline"]
    a2, b2 = s["test2"]["agent_v2"], s["test2"]["baseline_v2"]

    def conf_share(set_name, method, lo):
        p = per[(per["set"] == set_name) & (per["method"] == method)]
        z = p[p["fault_code"].isin(["F0a", "F0b"]) & p["top_conf"].notna()]
        return {"k": int((z["top_conf"] >= lo).sum()), "n": int(len(z))}

    out = {
        "D-1 ID 표기": {
            "v1_wrong_cause_with_id_issue": idc["with_id_issue"], "v1_wrong_cause": idc["wrong_cause_scenarios"],
            "v1_fixable_to_answer": idc["unique_fix_matches_answer"],
            "v2_id_check": a2.get("id_check"),
            "v1_f1_hit3_strict": rate(a1["by_fault"]["F1"]["hit3_strict"]),
            "v2_f1_hit3_strict": rate(a2["by_fault"]["F1"]["hit3_strict"]),
        },
        "D-2 지표 정의": {
            "v1_agent_hit1_le_hit3": a1["hit1_strict"]["value"] <= a1["hit3_strict"]["value"],
            "v2_agent_hit1_le_hit3": a2["hit1_strict"]["value"] <= a2["hit3_strict"]["value"],
            "v1_agent_hit1_strict": rate(a1["hit1_strict"]), "v1_agent_hit3_strict": rate(a1["hit3_strict"]),
            "v2_agent_hit1_strict": rate(a2["hit1_strict"]), "v2_agent_hit3_strict": rate(a2["hit3_strict"]),
        },
        "D-3 오경보와 신뢰도": {
            "v1_agent_false_alarm": rate(a1["false_alarm"]), "v1_baseline_false_alarm": rate(b1["false_alarm"]),
            "v2_agent_false_alarm": rate(a2["false_alarm"]), "v2_baseline_false_alarm": rate(b2["false_alarm"]),
            "v1_no_cause_top_conf_ge_0_8": conf_share("test", "agent", 0.8),
            "v2_no_cause_top_conf_ge_0_8": conf_share("test2", "agent_v2", 0.8),
        },
        "D-4 추가 확인": {
            "v1_requested": rate((a1.get("check_effect") or {}).get("requested")),
            "v2_requested": rate((a2.get("check_effect") or {}).get("requested")),
            "v2_top_changed": rate((a2.get("check_effect") or {}).get("top_changed_given_requested")),
        },
        "D-5 단계 전체 원인과 설비×레시피": {
            "v1_agent_f5_hit3_strict": rate(a1["by_fault"]["F5"]["hit3_strict"]),
            "v2_agent_f5_hit3_strict": rate(a2["by_fault"]["F5"]["hit3_strict"]),
            "v1_baseline_f5_hit3_strict": rate(b1["by_fault"]["F5"]["hit3_strict"]),
            "v2_baseline_f5_hit3_strict": rate(b2["by_fault"]["F5"]["hit3_strict"]),
            "v1_baseline_f2_hit3_strict": rate(b1["by_fault"]["F2"]["hit3_strict"]),
            "v2_baseline_f2_hit3_strict": rate(b2["by_fault"]["F2"]["hit3_strict"]),
        },
        "D-6 출력 길이": {
            "v1": run_stats(args.run1, "scn_5"),
            "v2": run_stats(args.run2, "scn_7"),
        },
        "추가: 확인 요청 오류": {
            "v1": check_errors(args.run1, "scn_5"),
            "v2": check_errors(args.run2, "scn_7"),
        },
    }
    (RESULTS / "diagnosis_v1_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    write_table(out, RESULTS / "failure_analysis_v2.md")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _r(x: dict | None) -> str:
    return "-" if not x else f"{x['k']}/{x['n']} = {x['value']:.3f}"


def _c(x: dict) -> str:
    return f"{x['k']}/{x['n']}"


def write_table(out: dict, path: Path) -> None:
    d1, d2, d3, d4, d5, d6, d7 = (out[k] for k in out)
    ic = d1["v2_id_check"] or {}
    rows = [
        ("D-1 줄여 쓴 ID가 오답 처리", f"틀린 원인 시나리오 {d1['v1_wrong_cause']}개 중 ID 표기 문제 "
         f"{d1['v1_wrong_cause_with_id_issue']}개 (정규화하면 정답 {d1['v1_fixable_to_answer']}개); F1 Hit@3 엄격 "
         f"{_r(d1['v1_f1_hit3_strict'])}",
         f"ID 정규화 {ic.get('normalized_ids_total', '-')}건, 재요청 {_r(ic.get('retry'))}, 남은 문제 "
         f"{_r(ic.get('unresolved_after_retry'))}; F1 Hit@3 엄격 {_r(d1['v2_f1_hit3_strict'])}"),
        ("D-2 Hit@1 > Hit@3", f"통일 후 Hit@1 {_r(d2['v1_agent_hit1_strict'])}, Hit@3 {_r(d2['v1_agent_hit3_strict'])}",
         f"Hit@1 {_r(d2['v2_agent_hit1_strict'])}, Hit@3 {_r(d2['v2_agent_hit3_strict'])}"),
        ("D-3 오경보·신뢰도", f"에이전트 {_r(d3['v1_agent_false_alarm'])}, 기준선 {_r(d3['v1_baseline_false_alarm'])}; "
         f"원인 없는 시나리오 1순위 신뢰도 0.8 이상 {_c(d3['v1_no_cause_top_conf_ge_0_8'])}",
         f"에이전트 {_r(d3['v2_agent_false_alarm'])}, 기준선 v2 {_r(d3['v2_baseline_false_alarm'])}; "
         f"1순위 신뢰도 0.8 이상 {_c(d3['v2_no_cause_top_conf_ge_0_8'])}"),
        ("D-4 추가 확인 요청", f"{_r(d4['v1_requested'])}",
         f"{_r(d4['v2_requested'])}, 1순위 변경 {_r(d4['v2_top_changed'])}"),
        ("D-5 단계 전체 원인(F5)·기준선 설비×레시피(F2)",
         f"F5 에이전트 {_r(d5['v1_agent_f5_hit3_strict'])}, 기준선 {_r(d5['v1_baseline_f5_hit3_strict'])}; "
         f"기준선 F2 {_r(d5['v1_baseline_f2_hit3_strict'])}",
         f"F5 에이전트 {_r(d5['v2_agent_f5_hit3_strict'])}, 기준선 v2 {_r(d5['v2_baseline_f5_hit3_strict'])}; "
         f"기준선 v2 F2 {_r(d5['v2_baseline_f2_hit3_strict'])}"),
        ("D-6 출력 길이에서 끊김", f"스키마 재요청 시나리오 {d6['v1']['scenarios_with_schema_retry']}/{d6['v1']['scenarios']}, "
         f"끊긴 응답 {d6['v1']['responses_cut_at_output_limit']}",
         f"스키마 재요청 시나리오 {d6['v2']['scenarios_with_schema_retry']}/{d6['v2']['scenarios']}, "
         f"끊긴 응답 {d6['v2']['responses_cut_at_output_limit']}"),
        ("추가: 확인 요청이 오류로 돌아감 (필드 누락 등)",
         f"확인 요청 {d7['v1']['requests']}건",
         f"확인 요청 {d7['v2']['requests']}건 중 {d7['v2']['errors']}건 (시나리오 {d7['v2']['scenarios_with_error']}개; "
         + ", ".join(f"{k} {v}" for k, v in sorted(d7['v2']['errors_by_check'].items())) + ")"),
    ]
    lines = ["<!-- diagnosis:start -->", "## 1차 진단 D-1~D-6의 2차 결과",
             "", "1차 = 에이전트 v1·기준선 v1, test1 / 2차 = 에이전트 v2·기준선 v2, test2 (서로 다른 시험 세트). "
             "원천: results/diagnosis_v1_v2.json", "",
             "| 진단 | 1차 | 2차 |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    lines += ["<!-- diagnosis:end -->"]
    block = "\n".join(lines)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    pat = re.compile(r"<!-- diagnosis:start -->.*?<!-- diagnosis:end -->", re.S)
    if pat.search(text):
        text = pat.sub(block, text)
    else:
        head, sep, rest = text.partition("\n### ")
        text = head.rstrip() + "\n\n" + block + "\n" + (sep + rest if sep else "")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
