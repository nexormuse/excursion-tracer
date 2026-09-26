"""세트 하나를 채점해 results/per_scenario.csv 와 results/summary.json 에 쓴다.

보고서 폴더에는 시나리오마다 <scenario_id>.json 이 있다. 파일 내용은 Report 하나이거나,
실행 기록({"final_report": ..., ...})이다. 없거나 스키마에 맞지 않으면 형식 실패(invalid)로 채점한다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from excursion_tracer.agent.schema import Report
from excursion_tracer.config import Config
from excursion_tracer.eval.compare import mcnemar_paired
from excursion_tracer.eval.ground_truth import load_ground_truth
from excursion_tracer.eval.match import score
from excursion_tracer.eval.metrics import breakdown, calibration, chance_level, core_metrics

RUN_INFO = "run_info.json"


def load_report(path: Path) -> tuple[Report | None, dict]:
    """(보고서, 실행 기록의 부가 정보). 실패하면 (None, {})."""
    if not path.is_file():
        return None, {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, {}
    extra = {}
    if isinstance(raw, dict) and "final_report" in raw:
        extra = {k: raw.get(k) for k in ("requests", "input_tokens", "output_tokens",
                                         "seconds", "numcheck_bad") if k in raw}
        if raw.get("invalid") or raw["final_report"] is None:
            return None, extra
        raw = raw["final_report"]
    try:
        return Report.model_validate(raw), extra
    except ValidationError:
        return None, extra


def scenario_dirs(data_root: Path, set_name: str) -> list[Path]:
    base = data_root / set_name
    return sorted(p for p in base.iterdir() if (p / "meta.json").is_file())


def score_method(cfg: Config, dirs: list[Path], gts: dict, report_dir: Path, method: str,
                 set_name: str) -> pd.DataFrame:
    info_path = report_dir / RUN_INFO
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    seconds = info.get("seconds", {})
    rows = []
    for d in dirs:
        gt = gts[d.name]
        rep, extra = load_report(report_dir / f"{d.name}.json")
        alert = json.loads((d / "alert.json").read_text(encoding="utf-8"))
        s = score(rep, gt, cfg.eval.tau, cfg.eval.onset_tolerance_days)
        rows.append({
            "set": set_name, "method": method, "scenario_id": d.name,
            "fault_code": gt["fault_code"], "effect": gt["cell"]["effect"],
            "stickiness": gt["cell"]["stickiness"], "alert_level": alert["level"],
            "regen_count": gt["regen_count"],
            **s,
            "seconds": extra.get("seconds", seconds.get(d.name)),
            **{k: extra.get(k) for k in ("requests", "input_tokens", "output_tokens", "numcheck_bad")},
        })
    return pd.DataFrame(rows)


def evaluate(cfg: Config, set_name: str, data_root: Path, methods: dict[str, Path],
             results_dir: Path, chance_draws: int = 2000) -> dict:
    dirs = scenario_dirs(data_root, set_name)
    gts = {d.name: load_ground_truth(d) for d in dirs}
    frames = {m: score_method(cfg, dirs, gts, p, m, set_name) for m, p in methods.items()}
    per = pd.concat(frames.values(), ignore_index=True)

    set_summary: dict = {}
    for m, df in frames.items():
        entry = core_metrics(df)
        entry["calibration"] = calibration(df)
        entry["by_fault"] = breakdown(df, "fault_code")
        entry["by_effect"] = breakdown(df[df["effect"].notna()], "effect")
        entry["by_stickiness"] = breakdown(df, "stickiness")
        entry["by_alert_level"] = breakdown(df, "alert_level")
        set_summary[m] = entry
    set_summary["chance"] = chance_level(dirs, gts, n_draws=chance_draws)
    names = list(frames)
    compare = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fa = frames[a].set_index("scenario_id")
            fb = frames[b].set_index("scenario_id")
            cause = ~fa["fault_code"].isin(["F0a", "F0b"])
            compare[f"{a}_vs_{b}"] = {"hit3_strict_mcnemar": mcnemar_paired(
                fa.loc[cause, "hit3_strict"], fb.loc[cause, "hit3_strict"])}
    set_summary["compare"] = compare
    regen = [g["regen_count"] for g in gts.values()]
    set_summary["info"] = {
        "n_scenarios": len(dirs),
        "fault_mix": dict(sorted(pd.Series([g["fault_code"] for g in gts.values()]).value_counts()
                                 .to_dict().items())),
        "regen_total": int(sum(regen)),
        "regen_by_fault": {k: int(v) for k, v in sorted(
            pd.Series(regen, index=[g["fault_code"] for g in gts.values()]).groupby(level=0).sum()
            .to_dict().items())},
        "tau": cfg.eval.tau,
        "methods": {m: str(p) for m, p in methods.items()},
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    csv = results_dir / "per_scenario.csv"
    if csv.is_file():
        old = pd.read_csv(csv)
        old = old[~((old["set"] == set_name) & old["method"].isin(names))]
        per = pd.concat([old, per], ignore_index=True)
    per.to_csv(csv, index=False)

    sj = results_dir / "summary.json"
    summary = json.loads(sj.read_text(encoding="utf-8")) if sj.is_file() else {}
    summary.setdefault(set_name, {})
    for k, v in set_summary.items():
        summary[set_name][k] = v
    sj.write_text(json.dumps(_clean(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
