"""결과 그림과 대표 시나리오 추적 기록을 만든다.

    python scripts/make_figures.py --set test2     # 2차: results/figures/, results/traces/ (+ fig8 1차 대 2차)
    python scripts/make_figures.py --set test      # 1차: results/figures/v1/, results/traces/v1/

대표 시나리오는 results/showcase_selection(_test2).json의 선정(성공 1, 실패 1, 새 유형 1)을 쓰고,
2차에는 반증 확인이 1순위 판단을 바꾼 사례 1건을 더한다 (1회차 오답 → 최종 정답인 것 중 번호가 가장 작은 것,
없으면 1순위가 바뀐 것 중 번호가 가장 작은 것). 선정 규칙과 채점 결과는 traces/README.md에 기록한다.
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
LABEL = {"success": "성공", "failure": "실패", "f5": "F5 (1차의 새 유형)", "new": "F6 (2차의 새 유형)",
         "changed": "반증 확인이 판단을 바꾼 사례"}
SETTINGS = {
    "test": {"keys": {"agent": "agent", "baseline": "baseline"}, "new": ("F5",), "fig": RESULTS / "figures" / "v1",
             "traces": RESULTS / "traces" / "v1", "selection": "showcase_selection.json"},
    "test2": {"keys": {"agent": "agent_v2", "baseline": "baseline_v2"}, "new": ("F6",), "fig": RESULTS / "figures",
              "traces": RESULTS / "traces", "selection": "showcase_selection_test2.json"},
}


def pick_changed(per_set: pd.DataFrame, agent: str) -> str | None:
    a = per_set[per_set["method"] == agent]
    if "top_changed" not in a:
        return None
    ch = a[a["top_changed"].astype("boolean").fillna(False)]
    cause = ~ch["fault_code"].isin(["F0a", "F0b"])
    fixed = ch[(cause & ch["prelim_hit3_strict"].astype("boolean").fillna(False).eq(False)
                & ch["hit3_strict"].astype("boolean").fillna(False))
               | (~cause & ch["prelim_false_alarm"].astype("boolean").fillna(False)
                  & ~ch["false_alarm"].astype("boolean").fillna(True))]
    pool = fixed if len(fixed) else ch
    return sorted(pool["scenario_id"])[0] if len(pool) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="test2", choices=["test", "test2"])
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--runs", type=Path, default=PROJECT_ROOT / "runs")
    args = ap.parse_args(argv)
    cfg = SETTINGS[args.set]
    keys = cfg["keys"]

    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    per = pd.read_csv(RESULTS / "per_scenario.csv")
    per_set = per[per["set"] == args.set]
    sel = json.loads((RESULTS / cfg["selection"]).read_text(encoding="utf-8"))
    run_dir = args.runs / sel["base_run"]
    showcase_dir = args.runs / f"{sel['base_run']}-showcase-ko"
    fig_dir = cfg["fig"]
    selection = dict(sel["selection"])
    if args.set == "test2":
        selection = {("new" if k == "f5" else k): v for k, v in selection.items()}
        changed = pick_changed(per_set, keys["agent"])
        if changed:
            selection["changed"] = changed

    F.setup()
    outs = [
        F.fig1_overall(summary, args.set, fig_dir / "fig1_overall.png", keys),
        F.fig2_difficulty(per, args.set, fig_dir / "fig2_difficulty.png", keys),
        F.fig3_by_fault(summary, args.set, fig_dir / "fig3_by_fault.png", keys, cfg["new"]),
        F.fig4_calibration(summary, args.set, fig_dir / "fig4_calibration.png", keys),
        (F.fig5_repeatability(summary, args.set, fig_dir / "fig5_repeatability.png", keys["agent"])
         if "repeatability" in summary[args.set].get(keys["agent"], {}) else "fig5: 반복성 결과 없음, 건너뜀"),
        F.fig6_case_timeline(args.data / args.set / selection["success"], run_dir / f"{selection['success']}.json",
                             per_set, fig_dir / "fig6_case_timeline.png", keys["agent"]),
        F.fig7_architecture(fig_dir / "fig7_architecture.png"),
    ]
    if args.set == "test2" and "test" in summary:
        outs.append(F.fig8_v1_v2(summary, fig_dir / "fig8_v1_v2.png"))
    for o in outs:
        print(o)

    trace_dir = cfg["traces"]
    trace_dir.mkdir(parents=True, exist_ok=True)
    readme = ["# 대표 시나리오 추적 기록", "",
              f"세트 {args.set}, 실행 `{rel(run_dir)}`. 선정 규칙 ({rel(RESULTS / cfg['selection'])}):", "",
              sel["rule"].replace("\n", " "), "",
              "반증 확인이 판단을 바꾼 사례: 1순위가 바뀐 것 중 1회차 오답(원인 시나리오) 또는 1회차 오경보(원인 없는 "
              "시나리오)가 최종에서 고쳐진 것의 번호가 가장 작은 것. 없으면 1순위가 바뀐 것 중 번호가 가장 작은 것."
              if args.set == "test2" else "", "",
              "| 구분 | 시나리오 | 원인 유형 | 에이전트 Hit@1 엄격 | 에이전트 Hit@3 느슨 | 기준선 Hit@3 엄격 | 파일 |",
              "|---|---|---|---|---|---|---|"]
    for kind, sid in selection.items():
        if not sid:
            continue
        a = per_set[(per_set["scenario_id"] == sid) & (per_set["method"] == keys["agent"])].iloc[0]
        b = per_set[(per_set["scenario_id"] == sid) & (per_set["method"] == keys["baseline"])].iloc[0]
        reason = (f"{LABEL[kind]}. {a['fault_code']}, 에이전트 Hit@1 엄격 {a['hit1_strict']}, "
                  f"Hit@3 느슨 {a['hit3_loose']}, 기준선 Hit@3 엄격 {b['hit3_strict']}")
        md = render_trace(args.data / args.set / sid, run_dir / f"{sid}.json", per_set, reason,
                          showcase_dir / f"{sid}.json", keys["agent"], keys["baseline"])
        (trace_dir / f"{sid}.md").write_text(md, encoding="utf-8")
        readme.append(f"| {LABEL[kind]} | {sid} | {a['fault_code']} | {a['hit1_strict']} | {a['hit3_loose']} | "
                      f"{b['hit3_strict']} | {sid}.md |")
        print(trace_dir / f"{sid}.md")
    (trace_dir / "index.json").write_text(json.dumps(
        {"set": args.set, "run": rel(run_dir), "selection": selection,
         "labels": {k: LABEL[k] for k in selection}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme += ["", "fig6_case_timeline.png는 성공 사례를 그린다."]
    (trace_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
