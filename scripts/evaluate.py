"""세트를 채점해 results/summary.json 과 results/per_scenario.csv 를 만든다.

    python scripts/evaluate.py --set dev --method baseline=runs/baseline/dev
"""

from __future__ import annotations

import argparse
from pathlib import Path

from excursion_tracer.config import PROJECT_ROOT, load_config
from excursion_tracer.eval.run_eval import evaluate


def _fmt(r: dict) -> str:
    if r.get("value") is None:
        return "-"
    lo, hi = r["ci95"]
    return f"{r['k']}/{r['n']} = {r['value']:.3f} [{lo:.3f}, {hi:.3f}]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", required=True, choices=["dev", "test", "test2"])
    ap.add_argument("--method", action="append", required=True,
                    help="이름=보고서 폴더 (여러 번 줄 수 있다)")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    ap.add_argument("--repeat", default=None,
                    help="반복성: 방법이름=기본폴더,추가폴더1,추가폴더2 (표본은 results/repeat_sample.json)")
    args = ap.parse_args(argv)

    methods = {}
    for m in args.method:
        name, _, path = m.partition("=")
        methods[name] = Path(path) if path else PROJECT_ROOT / "runs" / name / args.set
    summary = evaluate(load_config(), args.set, args.data, methods, args.results)[args.set]

    for name in methods:
        s = summary[name]
        print(f"\n[{args.set} / {name}]")
        for key in ("hit1_strict", "hit3_strict", "hit1_loose", "hit3_loose", "false_alarm",
                    "miss_rate", "onset_accuracy", "f0b_explained", "invalid_rate"):
            print(f"  {key:<15} {_fmt(s[key])}")
        print(f"  {'mrr_strict':<15} {s['mrr_strict']['value']:.3f}")
        print(f"  {'mrr_loose':<15} {s['mrr_loose']['value']:.3f}")
        print("\n  원인 유형별 Hit@3")
        print(f"  {'유형':<5}{'n':>3}  {'Hit@3 엄격':<28}{'Hit@3 느슨':<28}오경보")
        for code, e in s["by_fault"].items():
            if "hit3_strict" in e:
                print(f"  {code:<5}{e['n']:>3}  {_fmt(e['hit3_strict']):<28}{_fmt(e['hit3_loose']):<28}")
            else:
                print(f"  {code:<5}{e['n']:>3}  {'-':<28}{'-':<28}{_fmt(e['false_alarm'])}")
    if args.repeat:
        import json

        from excursion_tracer.eval.run_eval import evaluate_repeatability

        name, _, paths = args.repeat.partition("=")
        sample_file = args.results / ("repeat_sample.json" if args.set == "test" else f"repeat_sample_{args.set}.json")
        sample = json.loads(sample_file.read_text())["scenarios"]
        r = evaluate_repeatability(args.set, name, [Path(p) for p in paths.split(",")], sample,
                                   args.results)
        print(f"\n반복성 [{name}]: {_fmt(r)} (실행 {r['runs_per_scenario']}회, 누락 {r['missing']})")
    ch = summary["chance"]
    print(f"\n우연 수준 Hit@3: 엄격 {ch['hit3_strict']:.4f}, 느슨 {ch['hit3_loose']:.4f} "
          f"(원인 시나리오 {ch['n_cause']}개, 추첨 {ch['n_draws']}회)")
    print(f"\n저장: {args.results / 'summary.json'}, {args.results / 'per_scenario.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
