"""시나리오 세트를 생성한다.

    python scripts/gen_scenarios.py --set dev
    python scripts/gen_scenarios.py --set test --n 80

test 세트는 에이전트 프롬프트를 고정한 뒤에만 만든다. docs/DECISIONS.md의
"고정 시각" 기록이 비어 있으면 생성을 거부한다.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from excursion_tracer.config import PROJECT_ROOT, load_config
from excursion_tracer.sim.generate import EFFECTS, STICKINESS, generate_set

DECISIONS = PROJECT_ROOT / "docs" / "DECISIONS.md"


def prompt_frozen() -> bool:
    if not DECISIONS.is_file():
        return False
    m = re.search(r"^- 고정 시각:[ \t]*(\S.*)$", DECISIONS.read_text(encoding="utf-8"), re.M)
    return bool(m)


def print_summary(results, seconds: float) -> None:
    by_code: dict[str, list] = defaultdict(list)
    for r in results:
        by_code[r.plan.fault_code].append(r)

    print(f"\n시나리오 {len(results)}개, {seconds:.1f}초")
    print("\n[원인 유형별]")
    print(f"{'유형':<5}{'개수':>5}{'alarm':>7}{'warning':>9}{'폐기':>6}{'시도':>6}{'폐기율':>8}")
    for code in sorted(by_code):
        rs = by_code[code]
        lv = Counter(r.alert_level for r in rs)
        disc = sum(r.regen_count for r in rs)
        tries = disc + len(rs)
        print(f"{code:<5}{len(rs):>5}{lv['alarm']:>7}{lv['warning']:>9}{disc:>6}{tries:>6}"
              f"{disc / tries:>8.1%}")

    print("\n[원인 시나리오 난이도 칸별 (F0 제외)]")
    print(f"{'효과':<8}{'stickiness':<11}{'개수':>5}{'폐기':>6}{'시도':>6}{'폐기율':>8}  유형")
    for e in EFFECTS:
        for s in STICKINESS:
            rs = [r for r in results if r.plan.effect == e and r.plan.stickiness == s]
            disc = sum(r.regen_count for r in rs)
            tries = disc + len(rs)
            codes = dict(sorted(Counter(r.plan.fault_code for r in rs).items()))
            rate = f"{disc / tries:.1%}" if tries else "-"
            print(f"{e:<8}{s:<11}{len(rs):>5}{disc:>6}{tries:>6}{rate:>8}  {codes}")

    print("\n[원인 없음 (F0) stickiness별]")
    for s in STICKINESS:
        rs = [r for r in results if r.plan.effect is None and r.plan.stickiness == s]
        print(f"  {s:<5} {len(rs)}개  {dict(sorted(Counter(r.plan.fault_code for r in rs).items()))}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", required=True, choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=None, help="개수 (구성 비율 유지, 기본은 설정값)")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--skip-freeze-check", action="store_true",
                    help="test 세트 생성 전 프롬프트 고정 기록 확인을 건너뛴다")
    args = ap.parse_args(argv)

    if args.set == "test" and not args.skip_freeze_check and not prompt_frozen():
        print("test 세트는 프롬프트 고정 이후에만 생성한다: docs/DECISIONS.md의 '고정 시각'이 비어 있다.",
              file=sys.stderr)
        return 2

    cfg = load_config()
    t0 = time.perf_counter()
    try:
        from tqdm import tqdm
        progress = lambda it: tqdm(it, desc=f"{args.set}", unit="scn")  # noqa: E731
    except ImportError:
        progress = None
    results = generate_set(cfg, args.set, args.out, n=args.n, progress=progress)
    print_summary(results, time.perf_counter() - t0)
    print(f"\n출력: {args.out / args.set}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
