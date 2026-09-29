"""통계 기준선을 세트 전체에 실행하고 보고서를 runs/baseline/<set>/ (v2는 runs/baseline_v2/<set>/) 에 저장한다.

    python scripts/run_baseline.py --set dev
    python scripts/run_baseline.py --set dev --version v2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from excursion_tracer.config import PROJECT_ROOT, load_config
from excursion_tracer.stats.baseline import run_baseline
from excursion_tracer.stats.baseline_v2 import run_baseline_v2
from excursion_tracer.stats.data import load_scenario


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", required=True, choices=["dev", "test", "test2"])
    ap.add_argument("--version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config()
    name = "baseline" if args.version == "v1" else "baseline_v2"
    run = run_baseline if args.version == "v1" else run_baseline_v2
    out = (args.out or PROJECT_ROOT / "runs" / name) / args.set
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in (args.data / args.set).iterdir() if (p / "meta.json").is_file())
    if args.limit:
        dirs = dirs[: args.limit]
    seconds = {}
    for d in dirs:
        t0 = time.perf_counter()
        report = run(load_scenario(d), cfg)
        seconds[d.name] = round(time.perf_counter() - t0, 3)
        (out / f"{d.name}.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        top = report.hypotheses[0] if report.hypotheses else None
        desc = (f"{top.entity_type} {top.chamber_id or top.tool_id or top.recipe_id} "
                f"conf {top.confidence:.3f}") if top else ""
        print(f"{d.name}  {report.verdict:<19} {desc}  {seconds[d.name]:.2f}s")
    (out / "run_info.json").write_text(
        json.dumps({"method": name, "seconds": seconds}, indent=2) + "\n", encoding="utf-8")
    s = list(seconds.values())
    print(f"\n{len(s)}개, 시나리오당 평균 {sum(s) / len(s):.2f}초, 최대 {max(s):.2f}초 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
