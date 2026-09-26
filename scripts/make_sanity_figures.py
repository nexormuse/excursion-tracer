"""생성한 세트에서 F1·F3·F0b 시나리오를 하나씩 골라 점검 그림을 그린다.

    python scripts/make_sanity_figures.py --set dev

유형마다 시나리오 번호가 가장 작은 것을 고른다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from excursion_tracer.config import PROJECT_ROOT, load_config
from excursion_tracer.eval.ground_truth import load_ground_truth
from excursion_tracer.viz.sanity import sanity_figure

CODES = ("F1", "F3", "F0b")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="dev", choices=["dev", "test"])
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "figures")
    args = ap.parse_args(argv)

    cfg = load_config()
    dirs = sorted(p for p in (args.data / args.set).iterdir() if (p / "meta.json").is_file())
    for code in CODES:
        pick = next((d for d in dirs if load_ground_truth(d)["fault_code"] == code), None)
        if pick is None:
            print(f"{code}: 해당 시나리오 없음")
            continue
        out = sanity_figure(cfg, pick, args.out / f"sanity_{code}_{pick.name}.png")
        print(f"{code}: {pick.name} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
