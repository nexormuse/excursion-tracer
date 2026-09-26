"""결과 그림 7장과 대표 시나리오 추적 기록을 만든다.

    python scripts/make_figures.py --set test

대표 시나리오는 results/showcase_selection.json의 선정(성공 1, 실패 1, F5 1)을 쓴다.
선정 규칙과 각 시나리오의 채점 결과는 results/traces/README.md에 기록한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from excursion_tracer.config import PROJECT_ROOT
from excursion_tracer.viz import figures as F
from excursion_tracer.viz.trace_render import rel, render_trace

RESULTS = PROJECT_ROOT / "results"
LABEL = {"success": "성공", "failure": "실패", "f5": "F5"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="test", choices=["dev", "test"])
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--runs", type=Path, default=PROJECT_ROOT / "runs")
    args = ap.parse_args(argv)

    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    per = pd.read_csv(RESULTS / "per_scenario.csv")
    per_set = per[per["set"] == args.set]
    sel = json.loads((RESULTS / "showcase_selection.json").read_text(encoding="utf-8"))
    run_dir = args.runs / sel["base_run"]
    showcase_dir = args.runs / f"{sel['base_run']}-showcase-ko"
    fig_dir = RESULTS / "figures"

    F.setup()
    outs = [
        F.fig1_overall(summary, args.set, fig_dir / "fig1_overall.png"),
        F.fig2_difficulty(per, args.set, fig_dir / "fig2_difficulty.png"),
        F.fig3_by_fault(summary, args.set, fig_dir / "fig3_by_fault.png"),
        F.fig4_calibration(summary, args.set, fig_dir / "fig4_calibration.png"),
        (F.fig5_repeatability(summary, args.set, fig_dir / "fig5_repeatability.png")
         if "repeatability" in summary[args.set].get("agent", {}) else "fig5: 반복성 결과 없음, 건너뜀"),
        F.fig6_case_timeline(args.data / args.set / sel["selection"]["success"],
                             run_dir / f"{sel['selection']['success']}.json", per_set,
                             fig_dir / "fig6_case_timeline.png"),
        F.fig7_architecture(fig_dir / "fig7_architecture.png"),
    ]
    for o in outs:
        print(o)

    trace_dir = RESULTS / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    readme = ["# 대표 시나리오 추적 기록", "",
              f"세트 {args.set}, 실행 `{rel(run_dir)}`. 선정 규칙 (results/showcase_selection.json):", "",
              sel["rule"].replace("\n", " "), "",
              "| 구분 | 시나리오 | 원인 유형 | 에이전트 Hit@1 엄격 | 에이전트 Hit@3 느슨 | 기준선 Hit@3 엄격 | 파일 |",
              "|---|---|---|---|---|---|---|"]
    for kind, sid in sel["selection"].items():
        a = per_set[(per_set["scenario_id"] == sid) & (per_set["method"] == "agent")].iloc[0]
        b = per_set[(per_set["scenario_id"] == sid) & (per_set["method"] == "baseline")].iloc[0]
        reason = (f"{LABEL[kind]} 사례. {a['fault_code']}, 에이전트 Hit@1 엄격 {a['hit1_strict']}, "
                  f"Hit@3 느슨 {a['hit3_loose']}, 기준선 Hit@3 엄격 {b['hit3_strict']}")
        md = render_trace(args.data / args.set / sid, run_dir / f"{sid}.json", per_set, reason,
                          showcase_dir / f"{sid}.json")
        (trace_dir / f"{sid}.md").write_text(md, encoding="utf-8")
        readme.append(f"| {LABEL[kind]} | {sid} | {a['fault_code']} | {a['hit1_strict']} | {a['hit3_loose']} | "
                      f"{b['hit3_strict']} | traces/{sid}.md |")
        print(trace_dir / f"{sid}.md")
    readme += ["", "fig6_case_timeline.png는 성공 사례를 그린다."]
    (trace_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
