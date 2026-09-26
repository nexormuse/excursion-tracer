"""실패 사례 정리: 원인 유형이 겹치지 않게 5건을 골라 실행 기록에서 사실을 뽑아 마크다운으로 쓴다.

    python scripts/failure_cases.py --set test --run runs/<run_id>

실패 = 원인 시나리오에서 Hit@3 엄격 실패, 원인 없는 시나리오에서 오경보.
선정: 유형별 실패 비율이 높은 순(동률은 실패 수, 유형 이름 순)으로 5개 유형, 유형마다 번호가 가장 작은 실패 시나리오.
손으로 쓴 요약은 <!-- manual:start --> 와 <!-- manual:end --> 사이에 두면 다시 실행해도 남는다.
각 사례에 넣는 사실: 정답, 에이전트의 가설과 인용 근거 줄 원문, 정답 요인이 근거 묶음에 나온 줄,
전체 공통성 스캔에서 정답 요인의 순위, 추가 확인 요청 여부.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from excursion_tracer.config import PROJECT_ROOT, load_config
from excursion_tracer.eval.ground_truth import load_ground_truth
from excursion_tracer.stats.commonality import scan_all
from excursion_tracer.stats.data import load_scenario

N_CASES = 5
MANUAL_START, MANUAL_END = "<!-- manual:start -->", "<!-- manual:end -->"


def _is_fail(r) -> bool:
    if r["fault_code"] in ("F0a", "F0b"):
        return bool(r["false_alarm"])
    return not bool(r["hit3_strict"])


def fault_entities(f: dict) -> list[str]:
    ids = [f.get("chamber_id"), f.get("tool_id"), f.get("recipe_id")]
    return [x for x in ids if x]


def pack_lines(pack: str, ids: list[str]) -> list[str]:
    out = []
    for line in pack.splitlines():
        if any(re.search(rf"(?<![\w-]){re.escape(i)}(?![\w-])", line) for i in ids):
            out.append(line)
    return out


def select(per: pd.DataFrame) -> list[str]:
    per = per.assign(fail=per.apply(_is_fail, axis=1))
    g = per.groupby("fault_code")["fail"].agg(["sum", "size"])
    g["rate"] = g["sum"] / g["size"]
    g = g[g["sum"] > 0].reset_index().sort_values(["rate", "sum", "fault_code"],
                                                   ascending=[False, False, True])
    out = []
    for code in g["fault_code"].head(N_CASES):
        out.append(sorted(per[(per["fault_code"] == code) & per["fail"]]["scenario_id"])[0])
    return out


def case_md(cfg, sid: str, data_dir: Path, run_dir: Path, per_row) -> str:
    gt = load_ground_truth(data_dir / sid)
    rec = json.loads((run_dir / f"{sid}.json").read_text(encoding="utf-8"))
    pack = rec["evidence_pack"]
    fr = rec["final_report"]
    lines = [f"### {sid} — {gt['fault_code']} (효과 {gt['cell']['effect'] or '-'}, "
             f"stickiness {gt['cell']['stickiness']}, 알림 {per_row['alert_level']})", ""]
    lines.append("**정답**")
    if not gt["faults"]:
        lines.append("- 설비 원인 없음")
    scan = scan_all(load_scenario(data_dir / sid)) if gt["faults"] else None
    for f in gt["faults"]:
        ents = fault_entities(f)
        lines.append(f"- {f['type']}: step {f['step_id']}, {', '.join(ents)}, 시작 {f['onset_ts']}, δ {f['delta']}")
        for e in ents:
            hit = scan.index[scan["entity_id"] == e].tolist()
            rank = f"{hit[0] + 1}위 / {len(scan)} (q {scan.loc[hit[0], 'q']:.3g}, 중앙값 차이 {scan.loc[hit[0], 'median_diff']:.3f})" if hit else "스캔 대상 아님"
            lines.append(f"  - {e}: 전체 공통성 스캔 {rank}")
        found = pack_lines(pack, ents)
        lines.append(f"  - 근거 묶음에 나온 줄: {len(found)}개")
        for l in found:
            lines.append(f"    - `{l[:220]}`")
    lines += ["", "**에이전트 최종 보고서**",
              f"- 판정 {fr['verdict'] if fr else 'invalid'}, 추가 확인 요청 {rec['needs_checks']}, 요청 {rec['requests']}회"]
    if fr:
        for h in fr["hypotheses"]:
            ent = h["chamber_id"] or h["tool_id"] or h["recipe_id"]
            if h["entity_type"] == "tool_recipe":
                ent = f"{h['tool_id']} × {h['recipe_id']}"
            lines.append(f"- #{h['rank']} {h['entity_type']} {ent} (step {h['step_id']}), 신뢰도 {h['confidence']}, 시작 {h['onset'] or '-'}")
            for e in h["evidence"]:
                src = next((l for l in pack.splitlines() if l.startswith(e["evidence_id"] + " ")), None)
                lines.append(f"  - 인용 {e['evidence_id']}: \"{e['summary'][:200]}\"")
                if src:
                    lines.append(f"    - 원문 `{src[:220]}`")
        for key, label in (("no_cause_explanation", "원인 없음 설명"),
                           ("confounding_notes", "교란 메모"), ("limitations", "한계")):
            if fr[key]:
                lines.append(f"- {label}: \"{fr[key][:300]}\"")
    for c in rec["checks"]:
        lines.append(f"- 추가 확인 {c['check_id']} {c['request']['name']}: {c['result'][:200]}")
    lines += ["", f"기록: `{run_dir / (sid + '.json')}`", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="test")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--method", default="agent")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "failure_analysis.md")
    args = ap.parse_args(argv)

    cfg = load_config()
    per = pd.read_csv(PROJECT_ROOT / "results" / "per_scenario.csv")
    per = per[(per["set"] == args.set) & (per["method"] == args.method)].reset_index(drop=True)
    chosen = select(per)
    rows = per.set_index("scenario_id")
    parts = [f"# 실패 사례 분석 ({args.set}, {args.method})", "",
             "선정 규칙: 원인 시나리오는 Hit@3 엄격 실패, 원인 없는 시나리오는 오경보를 실패로 보고, "
             "실패 비율이 높은 원인 유형부터 5개 유형을 골라 유형마다 번호가 가장 작은 실패 시나리오를 택했다.", ""]
    fail = per.assign(fail=per.apply(_is_fail, axis=1)).groupby("fault_code")["fail"].agg(["sum", "size"])
    parts.append("| 유형 | 실패 | 전체 |")
    parts.append("|---|---|---|")
    for code, r in fail.iterrows():
        parts.append(f"| {code} | {int(r['sum'])} | {int(r['size'])} |")
    parts.append("")
    if args.out.is_file():
        old = args.out.read_text(encoding="utf-8")
        m = re.search(rf"{re.escape(MANUAL_START)}.*?{re.escape(MANUAL_END)}", old, flags=re.S)
        if m:
            parts += [m.group(0), ""]
    for sid in chosen:
        parts.append(case_md(cfg, sid, args.data / args.set, args.run, rows.loc[sid]))
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"선정: {chosen}\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
