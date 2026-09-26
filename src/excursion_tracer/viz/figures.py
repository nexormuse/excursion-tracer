"""결과 그림 7장. 지표 숫자는 results/summary.json과 results/per_scenario.csv에서만 읽는다.

fig6(사례 시간축)만 해당 시나리오의 관측 파일·정답·실행 기록을 함께 읽는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from excursion_tracer.eval.ground_truth import load_ground_truth  # noqa: E402

# 색: 범주 1 파랑(에이전트), 범주 2 주황(기준선), 순차 파랑 램프, 중립 잉크
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
METHOD_COLOR = {"agent": "#2a78d6", "baseline": "#eb6834"}
METHOD_LABEL = {"agent": "LLM 에이전트", "baseline": "통계 기준선"}
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
FONTS = ["AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Malgun Gothic"]
CAUSE_CODES = ["F1", "F2", "F3", "F4", "F5"]
FAULT_LABEL = {"F1": "F1 정비 후\n챔버 열화", "F2": "F2 설비×\n레시피", "F3": "F3 드리프트",
               "F4": "F4 두 원인", "F5": "F5 레시피 변경\n(test 전용)"}


def setup() -> None:
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in FONTS:
        if n in names:
            plt.rcParams["font.family"] = n
            break
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "normal", "legend.frameon": False,
    })


def _fig(w=16, h=9):
    return plt.figure(figsize=(w, h), dpi=100)


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def _err(r: dict) -> tuple[float, float]:
    lo, hi = r["ci95"]
    return r["value"] - lo, hi - r["value"]


def _bar(ax, x, r: dict, color: str, width: float, label=None):
    if r["value"] is None:
        return
    lo, hi = _err(r)
    ax.bar(x, r["value"], width=width, color=color, label=label, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, r["value"], yerr=[[lo], [hi]], fmt="none", ecolor=INK2, elinewidth=1.5, capsize=5)
    ax.text(x, r["ci95"][1] + 0.02, f"{r['k']}/{r['n']}\n{r['value']:.3f}", ha="center", va="bottom",
            fontsize=11, color=INK)


def _methods(s: dict) -> list[str]:
    return [m for m in ("agent", "baseline") if m in s]


# ---------------------------------------------------------------- fig1
def fig1_overall(summary: dict, set_name: str, out: Path) -> Path:
    s = summary[set_name]
    ms = _methods(s)
    metrics = [("hit3_strict", "Hit@3 (엄격)\n원인 시나리오"), ("hit3_loose", "Hit@3 (느슨)\n원인 시나리오"),
               ("false_alarm", "오경보율\n원인 없는 시나리오 (낮을수록 좋음)")]
    fig = _fig()
    ax = fig.add_subplot(111)
    w = 0.34
    for i, (key, _) in enumerate(metrics):
        for j, m in enumerate(ms):
            x = i + (j - (len(ms) - 1) / 2) * (w + 0.02)
            _bar(ax, x, s[m][key], METHOD_COLOR[m], w, METHOD_LABEL[m] if i == 0 else None)
    ch = s["chance"]
    for i, key in ((0, "hit3_strict"), (1, "hit3_loose")):
        ax.plot([i - 0.42, i + 0.42], [ch[key]] * 2, ls="--", color=INK, lw=1.5,
                label="우연 수준 (같은 수준 후보에서 무작위 3개)" if i == 0 else None)
        ax.text(i + 0.45, ch[key], f"{ch[key]:.3f}", va="center", fontsize=11, color=INK)
    ax.set_xticks(range(len(metrics)), [m[1] for m in metrics])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("비율 (95% Wilson 신뢰구간)")
    n = s[ms[0]]
    ax.set_title(f"{set_name} 세트 전체 성능: 원인 시나리오 {n['n_cause']}개, 원인 없는 시나리오 {n['n_no_cause']}개")
    ax.legend(loc="upper left", ncol=3)
    return _save(fig, out)


# ---------------------------------------------------------------- fig2
def fig2_difficulty(per: pd.DataFrame, set_name: str, out: Path) -> Path:
    p = per[(per["set"] == set_name) & per["effect"].notna()]
    ms = [m for m in ("agent", "baseline") if m in set(p["method"])]
    effects, sticks = ["small", "medium", "large"], ["low", "high"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig = _fig()
    axes = fig.subplots(1, len(ms), sharey=True, gridspec_kw={"wspace": 0.08})
    axes = np.atleast_1d(axes)
    for idx, (ax, m) in enumerate(zip(axes, ms)):
        g = p[p["method"] == m].groupby(["stickiness", "effect"])["hit3_strict"]
        k = g.sum().unstack().reindex(index=sticks, columns=effects)
        n = g.size().unstack().reindex(index=sticks, columns=effects)
        rate = (k / n).to_numpy(dtype=float)
        ax.imshow(rate, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        for i in range(len(sticks)):
            for j in range(len(effects)):
                v = rate[i, j]
                ax.text(j, i, f"{v:.2f}\n({int(k.iloc[i, j])}/{int(n.iloc[i, j])})", ha="center", va="center",
                        fontsize=14, color="#ffffff" if v >= 0.5 else INK)
        ax.set_xticks(range(3), ["효과 소", "효과 중", "효과 대"])
        ax.set_yticks(range(2), ["stickiness 저\n(교란 약함)", "stickiness 고\n(교란 강함)"])
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False, length=0)
        ax.set_title(METHOD_LABEL[m])
        for sp in ax.spines.values():
            sp.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=list(axes), shrink=0.7)
    cb.set_label("Hit@3 (엄격)", labelpad=12)
    sizes = p.groupby(["method", "stickiness", "effect"]).size()
    size_txt = f"칸마다 {sizes.min()}개" if sizes.min() == sizes.max() else f"칸마다 {sizes.min()}~{sizes.max()}개"
    fig.suptitle(f"{set_name} 세트 난이도 칸별 Hit@3 (엄격), 원인 시나리오 ({size_txt})", fontsize=16)
    return _save(fig, out)


# ---------------------------------------------------------------- fig3
def fig3_by_fault(summary: dict, set_name: str, out: Path) -> Path:
    s = summary[set_name]
    ms = _methods(s)
    codes = [c for c in CAUSE_CODES if c in s[ms[0]]["by_fault"]]
    fig = _fig()
    ax = fig.add_subplot(111)
    w = 0.34
    if "F5" in codes:
        i5 = codes.index("F5")
        ax.axvspan(i5 - 0.5, i5 + 0.5, color="#f1f0ec", zorder=0)
        ax.text(i5, 1.13, "dev에 없던 유형", ha="center", fontsize=12, color=INK2)
    for i, c in enumerate(codes):
        for j, m in enumerate(ms):
            x = i + (j - (len(ms) - 1) / 2) * (w + 0.02)
            _bar(ax, x, s[m]["by_fault"][c]["hit3_strict"], METHOD_COLOR[m], w,
                 METHOD_LABEL[m] if i == 0 else None)
    ax.set_xticks(range(len(codes)), [FAULT_LABEL[c] for c in codes])
    ax.set_xlim(-0.5, len(codes) - 0.5)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Hit@3 (엄격, 95% 신뢰구간)")
    ax.set_title(f"{set_name} 세트 원인 유형별 Hit@3 (엄격)")
    ax.legend(loc="upper left", ncol=2)
    return _save(fig, out)


# ---------------------------------------------------------------- fig4
def fig4_calibration(summary: dict, set_name: str, out: Path) -> Path:
    s = summary[set_name]
    ms = _methods(s)
    fig = _fig()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=1.2, label="완전 보정 (신뢰도 = 적중률)")
    markers = {"agent": "o", "baseline": "s"}
    for m in ms:
        pts = [b for b in s[m]["calibration"] if b["n"] and b["mean_conf"] is not None]
        x = [b["mean_conf"] for b in pts]
        y = [b["value"] for b in pts]
        lo = [b["value"] - b["ci95"][0] for b in pts]
        hi = [b["ci95"][1] - b["value"] for b in pts]
        ax.errorbar(x, y, yerr=[lo, hi], fmt=markers[m] + "-", color=METHOD_COLOR[m], ms=10, lw=2,
                    capsize=5, label=METHOD_LABEL[m], markeredgecolor=SURFACE, markeredgewidth=2)
        for b in pts:
            ax.annotate(f"n={b['n']}", (b["mean_conf"], b["value"]), textcoords="offset points",
                        xytext=(10, -14), fontsize=11, color=INK2)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    edges = [b["bin"] for b in s[ms[0]]["calibration"]]
    ax.set_xlabel(f"1순위 가설 신뢰도 (구간 {len(edges)}개의 평균, 구간 폭 {edges[0][1] - edges[0][0]:.1f})")
    ax.set_ylabel("실제 Hit@1 (엄격, 95% 신뢰구간)")
    ax.set_title(f"{set_name} 세트 신뢰도 보정 곡선 (원인 시나리오)")
    ax.legend(loc="upper left")
    return _save(fig, out)


# ---------------------------------------------------------------- fig5
def fig5_repeatability(summary: dict, set_name: str, out: Path) -> Path:
    r = summary[set_name]["agent"]["repeatability"]
    fig = _fig(16, 6)
    ax = fig.add_subplot(111)
    lo, hi = _err(r)
    ax.barh([0], [r["value"]], height=0.45, color=METHOD_COLOR["agent"], edgecolor=SURFACE, linewidth=2)
    ax.errorbar([r["value"]], [0], xerr=[[lo], [hi]], fmt="none", ecolor=INK2, elinewidth=1.5, capsize=6)
    ax.text(r["ci95"][1] + 0.015, 0, f"{r['k']}/{r['n']} = {r['value']:.3f}  "
            f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]", va="center", fontsize=14)
    ax.set_yticks([0], [METHOD_LABEL["agent"]])
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel("판정과 1순위 가설이 모든 실행에서 같았던 시나리오 비율 (95% 신뢰구간)")
    ax.set_title(f"{set_name} 세트 반복성: 시나리오 {r['n']}개 × {r['runs_per_scenario']}회 실행 "
                 f"(원인 유형별 층화 추출)")
    ax.grid(axis="y", visible=False)
    return _save(fig, out)


# ---------------------------------------------------------------- fig6
def _hyp_members(h: dict, hist: pd.DataFrame) -> pd.Series:
    hs = hist[hist["step_id"] == h["step_id"]]
    if h["entity_type"] == "chamber":
        m = hs["chamber_id"] == h["chamber_id"]
    elif h["entity_type"] == "recipe":
        m = hs["recipe_id"] == h["recipe_id"]
    elif h["entity_type"] == "tool_recipe":
        m = (hs["tool_id"] == h["tool_id"]) & (hs["recipe_id"] == h["recipe_id"])
    else:
        m = hs["tool_id"] == h["tool_id"]
    return hs.assign(inside=m.to_numpy())


def fig6_case_timeline(scenario_dir: Path, run_record: Path, per: pd.DataFrame, out: Path) -> Path:
    rec = json.loads(run_record.read_text(encoding="utf-8"))
    gt = load_ground_truth(scenario_dir)
    alert = json.loads((scenario_dir / "alert.json").read_text(encoding="utf-8"))
    wafers = pd.read_parquet(scenario_dir / "wafers.parquet")
    hist = pd.read_parquet(scenario_dir / "history.parquet")
    hyp = rec["final_report"]["hypotheses"][0]
    hs = _hyp_members(hyp, hist).merge(wafers[["wafer_id", "yield"]], on="wafer_id")
    hs["day"] = hs["track_in_ts"].dt.floor("D")
    g = hs.groupby(["day", "inside"])["yield"].mean().unstack()
    row = per[(per["scenario_id"] == rec["scenario_id"]) & (per["method"] == "agent")].iloc[0]

    fig = _fig()
    ax = fig.add_subplot(111)
    ent = hyp["chamber_id"] or hyp["tool_id"] or hyp["recipe_id"]
    if hyp["entity_type"] == "tool_recipe":
        ent = f"{hyp['tool_id']} × {hyp['recipe_id']}"
    ax.plot(g.index, g[True], "-o", color=METHOD_COLOR["agent"], lw=2, ms=8,
            label=f"에이전트 1순위 가설 {ent}을 거친 웨이퍼")
    ax.plot(g.index, g[False], "-s", color=INK2, lw=2, ms=8, label=f"같은 단계({hyp['step_id']}) 나머지 웨이퍼")
    ws, we = pd.Timestamp(alert["window_start"]), pd.Timestamp(alert["window_end"])
    ax.axvspan(ws, we, facecolor="none", edgecolor=GRID, hatch="//", lw=0, label=f"알림 구간 ({alert['level']})")
    for f in gt["faults"]:
        ax.axvline(pd.Timestamp(f["onset_ts"]), color=INK, lw=2,
                   label=f"정답 시작 시각 ({f['type']}, {f['step_id']})")
    if hyp["onset"]:
        ax.axvline(pd.Timestamp(hyp["onset"]), color=METHOD_COLOR["agent"], lw=2, ls="--",
                   label=f"에이전트가 낸 시작 시각 (신뢰도 {hyp['confidence']})")
    ax.set_xlabel("해당 단계 처리일")
    ax.set_ylabel("일별 평균 수율")
    truth = "; ".join(f"{f['type']} {f.get('chamber_id') or f.get('tool_id') or ''} {f.get('recipe_id') or ''}".strip()
                      for f in gt["faults"]) or "원인 없음"
    ax.set_title(f"{rec['scenario_id']}: 정답 {truth} / 에이전트 Hit@1(엄격) {bool(row['hit1_strict'])}, "
                 f"Hit@3(엄격) {bool(row['hit3_strict'])}")
    ax.legend(loc="lower left", fontsize=11)
    fig.autofmt_xdate()
    return _save(fig, out)


# ---------------------------------------------------------------- fig7
def fig7_architecture(out: Path) -> Path:
    fig = _fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, text, fill="#ffffff", edge=INK2, weight=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.2",
                                    facecolor=fill, edgecolor=edge, linewidth=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=13)

    def arrow(x0, y0, x1, y1, text="", at=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                                     color=INK2, linewidth=1.6))
        if text:
            tx, ty = at or ((x0 + x1) / 2, (y0 + y1) / 2 + 0.22)
            ax.text(tx, ty, text, ha="center", fontsize=11, color=INK2)

    code, llm, truth = "#eef4fc", "#e6f0fb", "#f4f3ef"
    box(0.4, 5.4, 2.6, 1.6, "시나리오 생성기\n(가상 팹, 정답 보유)", fill=truth, weight="bold")
    box(4.0, 6.4, 2.6, 1.2, "관측 데이터\n이력·수율·이벤트·알림")
    box(4.0, 3.9, 2.6, 1.2, "ground_truth.json\n(평가기만 읽음)", fill=truth, edge=INK)
    box(7.6, 6.4, 2.6, 1.2, "통계 엔진 (코드)\n층화·BH·변화점·교란", fill=code)
    box(11.0, 7.3, 2.6, 1.0, "통계 기준선\n(코드만)", fill=code)
    box(11.0, 5.2, 2.6, 1.6, "LLM 에이전트\nGemini, 호출 1~2회\n근거 묶음 → 보고서", fill=llm,
        edge=METHOD_COLOR["agent"], weight="bold")
    box(13.9, 5.9, 1.8, 1.2, "보고서\n(같은 스키마)")
    box(11.0, 2.2, 2.6, 1.4, "평가기\n정답 대조·지표·\n신뢰구간·McNemar", fill=code)
    box(13.9, 0.6, 1.8, 1.6, "summary.json\n그림·추적 기록")
    box(7.6, 0.6, 3.0, 1.4, "사람 (엔지니어)\n조치 승인 · 시스템 밖", fill="#ffffff", edge=INK)
    arrow(3.0, 6.6, 4.0, 7.0, "관측만")
    arrow(3.0, 5.8, 4.0, 4.5, "정답")
    arrow(6.6, 7.0, 7.6, 7.0)
    arrow(10.2, 7.3, 11.0, 7.8)
    arrow(10.2, 6.6, 11.0, 6.0, "근거 묶음·추가 확인", at=(9.3, 5.8))
    arrow(13.6, 7.8, 14.6, 7.1)
    arrow(13.6, 6.0, 13.9, 6.3)
    arrow(14.8, 5.9, 13.3, 3.6)
    arrow(6.6, 4.5, 11.0, 2.9, "평가기만", at=(8.2, 4.2))
    arrow(13.6, 2.4, 14.4, 2.2)
    arrow(11.3, 5.2, 10.0, 2.0, "권고 조치", at=(10.0, 3.9))
    ax.text(0.4, 8.5, "시스템 구조: 출제자(생성기)와 풀이자(통계 엔진·LLM)를 분리하고, 정답은 평가기만 읽는다",
            fontsize=16)
    ax.text(0.4, 0.3, "역할: 코드 = 데이터 생성·통계 검정·채점 / LLM = 가설 정리·추가 확인 계획·보고서 / 사람 = 최종 조치 승인",
            fontsize=12, color=INK2)
    return _save(fig, out)
