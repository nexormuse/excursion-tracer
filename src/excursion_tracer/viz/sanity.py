"""생성기 점검 그림: 일별 평균 수율, 저수율 비율, 원인 시작 시각, 알림 구간."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from excursion_tracer.config import Config  # noqa: E402
from excursion_tracer.eval.ground_truth import load_ground_truth  # noqa: E402

SIM_START = pd.Timestamp("2026-01-01")
FONT_CANDIDATES = ["AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Malgun Gothic"]


def _set_font() -> None:
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in FONT_CANDIDATES:
        if name in names:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def _day(ts) -> float:
    return (pd.Timestamp(ts) - SIM_START) / pd.Timedelta(days=1)


def sanity_figure(cfg: Config, scenario_dir: Path, out_path: Path) -> Path:
    _set_font()
    d = Path(scenario_dir)
    alert = json.loads((d / "alert.json").read_text(encoding="utf-8"))
    gt = load_ground_truth(d)
    w = pd.read_parquet(d / "wafers.parquet")
    w["day"] = ((w["test_ts"] - SIM_START) / pd.Timedelta(days=1)).astype(int)
    thr = alert["baseline"]["low_yield_threshold"]
    w["low"] = w["yield"] < thr
    daily = w.groupby("day").agg(mean=("yield", "mean"), low=("low", "mean"), n=("yield", "size"))
    by_prod = w.groupby(["day", "product"])["yield"].mean().unstack()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=100, sharex=True)
    ax1.plot(daily.index, daily["mean"], "k-o", label="전체 평균")
    for p, style in zip(by_prod.columns, ["--s", ":^"]):
        ax1.plot(by_prod.index, by_prod[p], style, color="gray", label=f"{p} 평균")
    ax1.set_ylabel("일별 평균 수율")
    ax2.plot(daily.index, daily["low"], "k-o", label="저수율 비율")
    ax2.axhline(alert["baseline"]["low_rate"], color="gray", ls="--", label="기준 비율")
    ax2.set_ylabel(f"저수율 비율 (수율 < {thr})")
    ax2.set_xlabel("측정일 (기준일로부터 일수)")

    lo, hi = cfg.monitor.baseline_test_days
    ws, we = _day(alert["window_start"]), _day(alert["window_end"])
    for ax in (ax1, ax2):
        ax.axvspan(lo, hi + 1, color="0.92", label="기준 구간")
        ax.axvspan(ws, we, facecolor="none", edgecolor="0.4", hatch="//", lw=0,
                   label=f"알림 구간 ({alert['level']})")

    marks = []
    for f in gt["faults"]:
        where = f["chamber_id"] or f["tool_id"] or f["recipe_id"]
        if f["type"] == "F2":
            where = f"{f['tool_id']} × {f['recipe_id']}"
        marks.append((_day(f["onset_ts"]), f"{f['type']} 시작: {where} (δ={f['delta']})"))
    if gt["fault_code"] == "F0b":
        mc = cfg.faults.f0b_mix_change
        marks.append((mc.from_release_day, f"F0b: {mc.from_release_day}일차 투입분부터 P2 비중 {mc.p2_share}"))
    for (x, label), ls in zip(marks, ["-", "-."]):
        for ax in (ax1, ax2):
            ax.axvline(x, color="k", ls=ls, lw=1.5, label=label if ax is ax1 else None)

    cell = gt["cell"]
    fig.suptitle(f"{gt['scenario_id']}  {gt['fault_code']}  "
                 f"(효과 {cell['effect'] or '-'}, stickiness {cell['stickiness']})")
    ax1.legend(loc="lower left", fontsize=9)
    ax2.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
