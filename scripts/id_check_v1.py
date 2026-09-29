"""1차(v1) 에이전트 실행 기록에서 가설 ID 표기 문제를 센다. 채점은 바꾸지 않는다.

    python scripts/id_check_v1.py --set test --run runs/<run_id>

에이전트가 Hit@3(엄격)을 놓친 원인 시나리오마다 상위 3개 가설의 ID가 fab.json에 없는 형식이었는지,
그 ID가 한 가지로만 해석될 때 정답 원인 중 하나와 엄격 일치하는지를 기록한다.
결과: results/v1_id_check.csv, results/v1_id_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from excursion_tracer.agent.schema import Hypothesis
from excursion_tracer.agent.validate_ids import FabIndex, normalize_hypothesis
from excursion_tracer.config import PROJECT_ROOT
from excursion_tracer.eval.ground_truth import load_ground_truth
from excursion_tracer.eval.match import NO_CAUSE_CODES, match_fault


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="test")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--per", type=Path, default=PROJECT_ROOT / "results" / "per_scenario.csv")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "v1_id_check")
    args = ap.parse_args(argv)

    per = pd.read_csv(args.per)
    per = per[per["set"] == args.set]
    ag = per[per["method"] == "agent"].set_index("scenario_id")
    bl = per[per["method"] == "baseline"].set_index("scenario_id")
    rows = []
    for sid, r in ag.iterrows():
        if r["fault_code"] in NO_CAUSE_CODES or bool(r["hit3_strict"]):
            continue
        d = args.data / args.set / sid
        idx = FabIndex.from_fab(json.loads((d / "fab.json").read_text(encoding="utf-8")))
        gt = load_ground_truth(d)
        rec = json.loads((args.run / f"{sid}.json").read_text(encoding="utf-8"))
        hyps = sorted((rec["final_report"] or {}).get("hypotheses", []), key=lambda h: h["rank"])[:3]
        bad_ids, fixed_hit, n_bad_h = [], False, 0
        for h in hyps:
            res = normalize_hypothesis(h, idx)
            if res.normalized or res.problems:
                n_bad_h += 1
                bad_ids += [x.split(": ", 1)[1].split(" -> ")[0] for x in res.normalized]
                bad_ids += [x for x in res.problems]
            if res.normalized and res.ok:
                hh = Hypothesis.model_validate(res.hypothesis)
                if any(match_fault(hh, f)[0] for f in gt["faults"]):
                    fixed_hit = True
        rows.append({
            "scenario_id": sid, "fault_code": r["fault_code"], "effect": r["effect"],
            "stickiness": r["stickiness"], "hypotheses_with_id_issue": n_bad_h,
            "id_issue": n_bad_h > 0, "ids": "; ".join(bad_ids),
            "unique_fix_matches_answer": fixed_hit,
            "baseline_hit3_strict": bool(bl.loc[sid, "hit3_strict"]) if sid in bl.index else None,
        })
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    by = df.groupby("fault_code").agg(wrong=("scenario_id", "size"), id_issue=("id_issue", "sum"),
                                      unique_fix_matches_answer=("unique_fix_matches_answer", "sum"))
    summary = {
        "set": args.set, "run": str(args.run),
        "wrong_cause_scenarios": int(len(df)),
        "with_id_issue": int(df["id_issue"].sum()),
        "unique_fix_matches_answer": int(df["unique_fix_matches_answer"].sum()),
        "by_fault": {k: {c: int(v) for c, v in row.items()} for k, row in by.iterrows()},
    }
    f1 = df[df["fault_code"] == "F1"]
    summary["f1"] = {
        "wrong": int(len(f1)),
        "baseline_hit": int(f1["baseline_hit3_strict"].sum()),
        "baseline_hit_large": int((f1["baseline_hit3_strict"] & (f1["effect"] == "large")).sum()),
        "id_issue": int(f1["id_issue"].sum()),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    print(df.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
