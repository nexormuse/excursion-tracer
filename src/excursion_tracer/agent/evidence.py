"""근거 묶음: 통계 코드가 만든 요약을 영어 텍스트로 만든다. 줄마다 근거 ID(E1, E2, …)를 붙인다.

항목: 알림, 팹 개요, 공통성 상위 후보(수준별), 상위 후보의 변화점, 후보 주변 이벤트,
교란(파이 계수·층화 비교), 상호작용, 제품 구성.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from excursion_tracer.config import Config
from excursion_tracer.stats.commonality import scan_all
from excursion_tracer.stats.confounding import confounding_check
from excursion_tracer.stats.data import ScenarioData
from excursion_tracer.stats.interaction import interaction_test
from excursion_tracer.stats.timewin import change_point

PHI_STRATIFY = 0.5
EVENT_DAYS = 2


def f3(x) -> str:
    """숫자는 소수 셋째 자리까지. 아주 작은 p값은 유효숫자 3자리 지수 표기."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    x = float(x)
    if x == float("inf"):
        return "inf"
    if x != 0 and abs(x) < 0.001:
        return f"{x:.2e}"
    return f"{x:.3f}"


def ts(t) -> str:
    if t is None or t == "" or (isinstance(t, float) and pd.isna(t)):
        return "n/a"
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M")


@dataclass
class Pack:
    lines: list[tuple[str, str]] = field(default_factory=list)  # (ID 또는 "", 문장)

    def head(self, text: str) -> None:
        self.lines.append(("", f"## {text}"))

    def add(self, text: str) -> None:
        self.lines.append(("E", text))

    def render(self) -> str:
        out, n = [], 0
        for kind, text in self.lines:
            if kind == "E":
                n += 1
                out.append(f"E{n} {text}")
            else:
                out.append(text)
        return "\n".join(out)


def _candidate_line(r) -> str:
    return (f"[{r['level']}] {r['entity_id']} (step {r['step_id']}): n {int(r['n_wafers'])}, "
            f"low-yield rate {f3(r['low_rate_in'])} vs rest {f3(r['low_rate_out'])}, "
            f"median diff {f3(r['median_diff'])}, OR {f3(r['odds_ratio'])}, "
            f"p {f3(r['p'])}, q {f3(r['q'])}")


def _events_near(data: ScenarioData, entity_id: str, level: str, when: pd.Timestamp) -> pd.DataFrame:
    ev = data.events
    step = entity_id.split("-")[0]
    if level == "tool":
        tool, chamber = entity_id, None
    elif level == "chamber":
        tool, chamber = entity_id.rsplit("-", 1)[0], entity_id
    else:
        tool = chamber = None
    m = (ev["event_type"] == "RECIPE_CHANGE") & (ev["step_id"] == step)
    if tool:
        m |= (ev["event_type"] == "PM") & (ev["tool_id"] == tool)
        if chamber:
            m |= (ev["event_type"] == "CHAMBER_PM") & (ev["chamber_id"] == chamber)
        else:
            m |= (ev["event_type"] == "CHAMBER_PM") & (ev["tool_id"] == tool)
    span = pd.Timedelta(days=EVENT_DAYS)
    m &= (ev["ts"] >= when - span) & (ev["ts"] <= when + span)
    return ev[m].sort_values("ts")


def _event_text(e) -> str:
    what = e["chamber_id"] if e["event_type"] == "CHAMBER_PM" else (
        e["tool_id"] if e["event_type"] == "PM" else e["recipe_id"])
    return f"{e['event_type']} {what} at {ts(e['ts'])}"


def build_evidence(data: ScenarioData, cfg: Config, top_n: int = 5) -> str:
    """근거 묶음 텍스트. evidence_max_chars를 넘으면 후보 수를 줄여 다시 만든다."""
    scan = scan_all(data, fdr_method=cfg.stats.fdr_method, stratify=cfg.stats.stratify_by_product)
    cps: dict[str, dict] = {}
    for n in range(top_n, 1, -1):
        text = _build(data, cfg, scan, n, cps)
        if len(text) <= cfg.agent.evidence_max_chars:
            return text
    raise ValueError(f"근거 묶음이 {cfg.agent.evidence_max_chars}자를 넘는다: {len(text)}")


def _build(data: ScenarioData, cfg: Config, scan: pd.DataFrame, n: int, cps: dict) -> str:
    p = Pack()
    a = data.alert

    p.head("Alert")
    p.add(f"level {a['level']}, window {ts(a['window_start'])} to {ts(a['window_end'])}, "
          f"low-yield threshold {f3(a['baseline']['low_yield_threshold'])} "
          f"(baseline low-yield rate {f3(a['baseline']['low_rate'])})")
    for prod, v in a["by_product"].items():
        drop = (v["mean_yield"] - v["baseline_mean"]) if v["mean_yield"] is not None else None
        p.add(f"{prod}: window mean yield {f3(v['mean_yield'])}, baseline mean {f3(v['baseline_mean'])}, "
              f"change {f3(drop)}, wafers {v['n_wafers']}, low-yield wafers {v['n_low']}")

    p.head("Fab overview")
    steps = data.fab["steps"]
    by_area: dict[str, Counter] = {}
    for s in steps:
        c = by_area.setdefault(s["area"], Counter())
        c["steps"] += 1
        c["tools"] += len(s["tools"])
        c["chambers"] += sum(len(t["chambers"]) for t in s["tools"])
    area_txt = "; ".join(f"{k} {v['steps']} steps/{v['tools']} tools/{v['chambers']} chambers"
                         for k, v in sorted(by_area.items()))
    p.add(f"{len(steps)} steps. {area_txt}")
    spec = [s["step_id"] for s in steps if s["product_specific_recipe"]]
    p.add(f"steps with product-specific recipes (P1->R1, P2->R2): {', '.join(spec) or 'none'}")

    p.head("Commonality candidates (stratified by product, BH q over all tests)")
    p.add(f"{len(scan)} comparisons; q <= 0.01: {int((scan['q'] <= 0.01).sum())}")
    for level in ("tool", "chamber", "recipe"):
        sub = scan[scan["level"] == level].head(n)
        if sub.empty:
            p.add(f"[{level}] no comparable candidates")
            continue
        for _, r in sub.iterrows():
            p.add(_candidate_line(r))

    top = scan.head(n)
    p.head(f"Change points of the top {len(top)} candidates (lot means relative to same-day rest of step)")
    for _, r in top.iterrows():
        ent = r["entity_id"]
        if ent not in cps:
            cps[ent] = change_point(data, ent, n_perm=cfg.stats.permutation_n)
        cp = cps[ent]
        p.add(f"{ent}: change point {ts(cp['ts'])}, mean before {f3(cp['mean_before'])}, "
              f"after {f3(cp['mean_after'])}, permutation p {f3(cp['p'])}, lots {cp['n_lots']}")

    p.head(f"Events within {EVENT_DAYS} days of each change point")
    for _, r in top.iterrows():
        ent = r["entity_id"]
        cp = cps[ent]
        if not cp["ts"]:
            p.add(f"{ent}: no change point")
            continue
        ev = _events_near(data, ent, r["level"], pd.Timestamp(cp["ts"]))
        txt = "; ".join(_event_text(e) for _, e in ev.iterrows()) or "none"
        p.add(f"{ent}: {txt}")

    p.head("Confounding among top candidates")
    ents = list(top["entity_id"])
    if len(ents) >= 2:
        cc = confounding_check(data, ents)
        phis = ", ".join(f"{q['a']}~{q['b']} {f3(q['phi'])}" for q in cc["pairs"])
        p.add(f"phi: {phis}")
        for q in cc["pairs"]:
            if q["phi"] >= PHI_STRATIFY:
                g = {k: q[k] for k in ("a_only", "b_only", "both", "neither")}
                parts = ", ".join(f"{k} n {v['n']} mean resid {f3(v['mean_yield'])} "
                                  f"low {f3(v['low_rate'])}" for k, v in g.items())
                p.add(f"{q['a']} vs {q['b']} (phi {f3(q['phi'])}): {parts}")

    p.head("Tool x recipe interaction")
    steps_seen = []
    for sid in top["step_id"]:
        if sid in steps_seen or not data.steps[sid]["product_specific_recipe"]:
            continue
        steps_seen.append(sid)
        it = interaction_test(data, sid)
        w = it["worst_cell"]
        cell = (f"most deviant cell {w['tool_id']} x {w['recipe_id']} n {w['n']}, "
                f"mean {f3(w['mean'])} vs additive {f3(w['additive_pred'])}") if w else "n/a"
        p.add(f"{sid}: interaction p {f3(it['p_interaction'])}; {cell}")
    if not steps_seen:
        p.add("no top candidate is at a step with product-specific recipes")

    p.head("Product mix")
    w = data.all_wafers.copy()
    w["day"] = w["test_ts"].dt.floor("D")
    share = w.groupby("day")["product"].apply(lambda s: (s == "P2").mean())
    lo, hi = cfg.monitor.baseline_test_days
    day0 = data.lots["release_ts"].min().floor("D")
    base = share[(share.index >= day0 + pd.Timedelta(days=lo))
                 & (share.index <= day0 + pd.Timedelta(days=hi))]
    win = share[share.index >= data.window_start.floor("D")]
    p.add(f"P2 share of measured wafers: baseline days mean {f3(base.mean())}, "
          f"alert window mean {f3(win.mean())}, daily min {f3(share.min())}, max {f3(share.max())}")
    daily = ", ".join(f"{d:%m-%d} {f3(v)}" for d, v in share.items())
    p.add(f"daily P2 share: {daily}")
    return p.render()
