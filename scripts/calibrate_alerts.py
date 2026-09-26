"""알림 보정: 원인 시나리오의 난이도 칸마다 시험용 시나리오를 임시 폴더에 만들어 폐기율을 잰다.

    python scripts/calibrate_alerts.py            # 칸마다 20개
    python scripts/calibrate_alerts.py --per-cell 40

폐기율 = 알림이 없어 버린 시도 수 / 전체 시도 수. 원인 유형은 F1·F2·F3·F4를 돌아가며
배정한다. 참고로 F0a·F0b도 stickiness별로 같은 수만큼 잰다. 결과는
results/alert_calibration.json에 저장한다.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from excursion_tracer.config import PROJECT_ROOT, load_config
from excursion_tracer.sim.generate import (
    EFFECTS,
    STICKINESS,
    ScenarioPlan,
    build_set_fab,
    generate_scenario,
)

CAUSE_CODES = ("F1", "F2", "F3", "F4")
SEED_BASE = 800000  # dev(1000~)·test(5000~) 번호와 겹치지 않는 시험용 seed


def run(per_cell: int, out: Path) -> dict:
    cfg = load_config()
    fab = build_set_fab(cfg, "dev")
    groups = [(e, s, CAUSE_CODES) for e in EFFECTS for s in STICKINESS]
    groups += [(None, s, ("F0a",)) for s in STICKINESS] + [(None, s, ("F0b",)) for s in STICKINESS]
    rows = []
    seed = SEED_BASE
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        for eff, stick, codes in groups:
            disc, levels, by_code = 0, Counter(), Counter()
            for i in range(per_cell):
                seed += 1
                code = codes[i % len(codes)]
                plan = ScenarioPlan(f"scn_{seed}", seed, code, eff, stick)
                r = generate_scenario(cfg, fab, "dev", plan, Path(tmp) / plan.scenario_id)
                disc += r.regen_count
                by_code[code] += r.regen_count
                levels[r.alert_level] += 1
            tries = disc + per_cell
            rows.append({
                "effect": eff, "stickiness": stick, "codes": list(codes), "n": per_cell,
                "discarded": disc, "attempts": tries, "discard_ratio": disc / tries,
                "alarm": levels["alarm"], "warning": levels["warning"],
                "discarded_by_code": dict(by_code),
            })
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "per_cell": per_cell,
        "effect": dict(cfg.faults.effect),
        "stickiness": dict(cfg.faults.stickiness),
        "max_discard_ratio": cfg.monitor.max_discard_ratio,
        "seconds": round(time.perf_counter() - t0, 1),
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def print_table(result: dict) -> None:
    lim = result["max_discard_ratio"]
    print(f"칸마다 {result['per_cell']}개, {result['seconds']}초. 효과 크기 {result['effect']}")
    print(f"{'구분':<10}{'효과':<8}{'stickiness':<11}{'폐기':>5}{'시도':>6}{'폐기율':>8}"
          f"{'alarm':>7}{'warning':>9}  유형별 폐기")
    for r in result["rows"]:
        kind = "원인" if r["effect"] else r["codes"][0]
        flag = "  > 한도" if r["effect"] and r["discard_ratio"] > lim else ""
        print(f"{kind:<10}{r['effect'] or '-':<8}{r['stickiness']:<11}{r['discarded']:>5}"
              f"{r['attempts']:>6}{r['discard_ratio']:>8.1%}{r['alarm']:>7}{r['warning']:>9}"
              f"  {r['discarded_by_code']}{flag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "alert_calibration.json")
    args = ap.parse_args(argv)
    print_table(run(args.per_cell, args.out))
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
