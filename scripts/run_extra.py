"""test 세트 추가 실행: 반복성(층화 추출 시나리오 × 추가 실행)과 한국어 쇼케이스.

    python scripts/run_extra.py repeat --set test          # 표본 선정 + 추가 2회 실행
    python scripts/run_extra.py showcase --set test        # 대표 3개 선정 + 한국어 실행

반복성 표본: 원인 유형별 개수 비율대로 eval.repeat_sample개를 seed 고정으로 뽑는다.
쇼케이스: 성공 1건(에이전트 Hit@1 엄격 적중, 기준선 Hit@3 엄격 실패인 것 중 번호가 가장 작은 것,
없으면 에이전트 Hit@1 엄격 적중 중 가장 작은 것), 실패 1건(원인 시나리오에서 에이전트 Hit@3 느슨 실패
중 가장 작은 것), 새 유형 1건(test는 F5, test2는 F6 중 번호가 가장 작은 것, 앞의 두 건과 겹치면 다음 것).
선정 결과는 results/repeat_sample.json, results/showcase_selection.json에 저장한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from excursion_tracer.agent.loop import RUNS_DIR, find_run_id, run_set
from excursion_tracer.agent.loop_v2 import find_run_id_v2, run_set_v2
from excursion_tracer.config import PROJECT_ROOT, load_config
from excursion_tracer.eval.ground_truth import load_ground_truth
from excursion_tracer.eval.run_eval import scenario_dirs
from excursion_tracer.llm.usage import QuotaExhausted, UsageTracker

RESULTS = PROJECT_ROOT / "results"


def stratified_sample(codes: dict[str, str], n: int, seed: int) -> list[str]:
    """원인 유형별 개수 비율대로 n개 (최대 나머지 방식), 유형 안에서는 무작위."""
    rng = np.random.default_rng(seed)
    by = pd.Series(codes).groupby(lambda k: codes[k]).apply(lambda s: sorted(s.index))
    total = len(codes)
    raw = {c: len(ids) * n / total for c, ids in by.items()}
    k = {c: int(np.floor(v)) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: (-(raw[c] - k[c]), c))[: n - sum(k.values())]:
        k[c] += 1
    out = []
    for c in sorted(by.index):
        ids = list(by[c])
        out += [ids[i] for i in sorted(rng.choice(len(ids), size=k[c], replace=False))]
    return sorted(out)


def pick_showcase(per: pd.DataFrame, set_name: str, agent: str, baseline: str = "baseline",
                  new_code: str = "F5") -> dict:
    p = per[per["set"] == set_name]
    a = p[p["method"] == agent].set_index("scenario_id")
    b = p[p["method"] == baseline].set_index("scenario_id")
    cause = a[~a["fault_code"].isin(["F0a", "F0b"])]
    win = [s for s in cause.index if cause.loc[s, "hit1_strict"] and not b.loc[s, "hit3_strict"]]
    if not win:
        win = [s for s in cause.index if cause.loc[s, "hit1_strict"]]
    fail = [s for s in cause.index if not cause.loc[s, "hit3_loose"]]
    chosen = {"success": sorted(win)[0] if win else None,
              "failure": sorted(fail)[0] if fail else None}
    new = [s for s in sorted(a.index) if a.loc[s, "fault_code"] == new_code and s not in chosen.values()]
    chosen["f5" if new_code == "F5" else "new"] = new[0] if new else None
    return chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["repeat", "showcase"])
    ap.add_argument("--set", default="test", choices=["dev", "test", "test2"])
    ap.add_argument("--version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--agent-method", default=None, help="per_scenario.csv의 에이전트 방법 이름")
    ap.add_argument("--baseline-method", default=None, help="per_scenario.csv의 기준선 방법 이름")
    ap.add_argument("--sample-n", type=int, default=None, help="반복성 표본 수 (기본은 eval.repeat_sample)")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--runs", type=Path, default=RUNS_DIR)
    args = ap.parse_args(argv)

    cfg = load_config()
    dirs = {d.name: d for d in scenario_dirs(args.data, args.set)}
    finder, runner = (find_run_id, run_set) if args.version == "v1" else (find_run_id_v2, run_set_v2)
    agent_method = args.agent_method or ("agent" if args.version == "v1" else "agent_v2")
    baseline_method = args.baseline_method or ("baseline" if args.version == "v1" else "baseline_v2")
    base_run = finder(cfg.llm.model, args.runs)
    tracker = UsageTracker.from_config(cfg.llm)
    from excursion_tracer.llm.gemini_client import GeminiClient

    client = GeminiClient(cfg.llm.model, tracker)
    RESULTS.mkdir(exist_ok=True)
    try:
        if args.what == "repeat":
            codes = {k: load_ground_truth(d)["fault_code"] for k, d in dirs.items()}
            n = args.sample_n or cfg.eval.repeat_sample
            sample = stratified_sample(codes, n, cfg.seed_for(args.set, 0))
            name = "repeat_sample.json" if args.version == "v1" else f"repeat_sample_{args.set}.json"
            (RESULTS / name).write_text(json.dumps(
                {"set": args.set, "base_run": base_run, "scenarios": sample}, indent=2) + "\n")
            print(f"반복성 표본 {len(sample)}개: {sample}")
            for k in range(1, cfg.eval.repeat_runs):
                rid = f"{base_run}-rep{k}"
                print(f"\n[{rid}]")
                runner(client, cfg, [dirs[s] for s in sample], rid, args.runs)
        else:
            per = pd.read_csv(RESULTS / "per_scenario.csv")
            chosen = pick_showcase(per, args.set, agent_method, baseline_method,
                                   new_code="F6" if args.set == "test2" else "F5")
            name = "showcase_selection.json" if args.version == "v1" else f"showcase_selection_{args.set}.json"
            (RESULTS / name).write_text(json.dumps(
                {"set": args.set, "base_run": base_run, "selection": chosen,
                 "rule": __doc__.split("쇼케이스: ")[1].split("\n선정")[0].strip()},
                ensure_ascii=False, indent=2) + "\n")
            print(f"쇼케이스 선정: {chosen}")
            rid = f"{base_run}-showcase-ko"
            runner(client, cfg, [dirs[s] for s in chosen.values() if s], rid, args.runs, language="ko")
    except QuotaExhausted as e:
        print(str(e), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
