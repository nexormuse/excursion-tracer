"""LLM 에이전트를 세트에 실행한다. 이미 끝난 시나리오는 건너뛴다.

    python scripts/run_agent.py --set dev --limit 5
    python scripts/run_agent.py --estimate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from excursion_tracer.agent.loop import RUNS_DIR, find_run_id, run_set
from excursion_tracer.config import PROJECT_ROOT, load_config
from excursion_tracer.llm.usage import QuotaExhausted, UsageTracker


def measured_requests_per_scenario(runs_dir: Path) -> tuple[float | None, int]:
    vals = []
    if runs_dir.is_dir():
        for rec_path in runs_dir.glob("*/scn_*.json"):
            try:
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "requests" in rec and "evidence_pack" in rec:
                vals.append(rec["requests"])
    return (sum(vals) / len(vals) if vals else None), len(vals)


def estimate(cfg, tracker: UsageTracker, runs_dir: Path) -> None:
    used = tracker.used_in_window()
    avg, n = measured_requests_per_scenario(runs_dir)
    per = avg if avg else 2.0
    remaining = tracker.remaining()
    print(f"한도 창 (태평양 날짜): {tracker.window()}")
    print(f"하루 한도 {tracker.rpd_limit}, 멈춤 기준 {tracker.cap} ({cfg.llm.stop_at_ratio:.0%}), "
          f"이번 창 사용 {used}, 남은 요청 {remaining}")
    src = f"실측 평균 {avg:.2f} (실행 기록 {n}개)" if avg else "실행 기록 없음, 최대값 2로 가정"
    print(f"시나리오당 요청: {src}")
    print(f"지금 한도 창에서 더 돌릴 수 있는 시나리오: {math.floor(remaining / per)}개")
    print(f"분당 한도 {tracker.rpm_limit} 기준 최소 소요: 요청당 {60 / tracker.rpm_limit:.1f}초")
    print(f"다음 초기화: {tracker.next_reset():%Y-%m-%d %H:%M} (한국 시간)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", choices=["dev", "test", "test2"])
    ap.add_argument("--version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--runs", type=Path, default=RUNS_DIR)
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    tracker = UsageTracker.from_config(cfg.llm)
    if args.estimate:
        estimate(cfg, tracker, args.runs)
        return 0
    if not args.set:
        ap.error("--set 이 필요하다")

    from excursion_tracer.llm.gemini_client import GeminiClient

    from excursion_tracer.agent.loop_v2 import find_run_id_v2, run_set_v2

    client = GeminiClient(cfg.llm.model, tracker)
    client.check_model()
    finder, runner = (find_run_id, run_set) if args.version == "v1" else (find_run_id_v2, run_set_v2)
    run_id = args.run_id or finder(cfg.llm.model, args.runs)
    dirs = sorted(p for p in (args.data / args.set).iterdir() if (p / "meta.json").is_file())
    print(f"run_id {run_id}, 에이전트 {args.version}, 모델 {cfg.llm.model}, 시나리오 {len(dirs)}개 (limit {args.limit})")
    try:
        recs = runner(client, cfg, dirs, run_id, args.runs, limit=args.limit)
    except QuotaExhausted as e:
        print(str(e), file=sys.stderr)
        return 3
    if recs:
        req = [r["requests"] for r in recs]
        sec = [r["seconds"] for r in recs]
        print(f"\n{len(recs)}개 실행: 형식 실패 {sum(r['invalid'] for r in recs)}, "
              f"2회차 실패 {sum(r['round2_failed'] for r in recs)}, "
              f"ID 정규화 {sum(len(r.get('id_normalized') or []) for r in recs)}건, "
              f"ID 재요청 {sum(r.get('id_retry') is not None for r in recs)}, "
              f"추가 확인 요청 {sum(r['needs_checks'] for r in recs)}, "
              f"시나리오당 요청 평균 {sum(req) / len(req):.2f} (최대 {max(req)}), "
              f"시간 평균 {sum(sec) / len(sec):.1f}초 (최대 {max(sec):.1f}초)")
    print(f"기록: {args.runs / run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
