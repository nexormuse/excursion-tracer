"""근거 묶음 v2: v1 항목에 제품 보정 하락, 알림 전 평소 차이와 평소 대비 배수, 이벤트 전후 스캔을 더한다.

후보 목록은 q ≤ stats.no_cause_q이고 수율이 낮아지는 쪽인 것만 평소 대비 배수 순으로 싣는다.
ID는 항상 전체 형식(S12-T3-C2, S07-R2)으로 쓴다.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from excursion_tracer.agent.evidence import (
    EVENT_DAYS, PHI_STRATIFY, Pack, _candidate_line, _event_text, _events_near, f3, ts,
)
from excursion_tracer.config import Config
from excursion_tracer.stats.commonality import scan_all
from excursion_tracer.stats.confounding import confounding_check
from excursion_tracer.stats.data import ScenarioData
from excursion_tracer.stats.events_scan import events_scan
from excursion_tracer.stats.interaction import interaction_test
from excursion_tracer.stats.nullref import add_ratio, null_reference, product_adjusted
from excursion_tracer.stats.timewin import change_point

MIN_TOP = 2


def build_evidence_v2(data: ScenarioData, cfg: Config, top_n: int = 5) -> str:
    ref = null_reference(data, cfg)
    scan = add_ratio(scan_all(data, fdr_method=cfg.stats.fdr_method,
                              stratify=cfg.stats.stratify_by_product), ref)
    ev = events_scan(data, cfg)
    pa = product_adjusted(data, cfg)
    cps: dict[str, dict] = {}
    text = ""
    for n in range(top_n, 1, -1):
        text = _build(data, cfg, scan, ev, ref, pa, n, cps)
        if len(text) <= cfg.v2.evidence_max_chars:
            return text
    raise ValueError(f"근거 묶음 v2가 {cfg.v2.evidence_max_chars}자를 넘는다: {len(text)}")


def _eligible(scan: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    e = scan[(scan["q"] <= cfg.stats.no_cause_q) & (scan["median_diff"] < 0)]
    return e.sort_values(["ratio", "q"], ascending=[False, True], kind="mergesort")


def _build(data, cfg, scan, ev, ref, pa, n, cps) -> str:
    p = Pack()
    a = data.alert
    lo, hi = cfg.v2.null_window_test_days

    p.head("Alert")
    p.add(f"level {a['level']}, window {ts(a['window_start'])} to {ts(a['window_end'])}, "
          f"low-yield threshold {f3(a['baseline']['low_yield_threshold'])} "
          f"(baseline low-yield rate {f3(a['baseline']['low_rate'])})")
    for prod, v in a["by_product"].items():
        drop = (v["mean_yield"] - v["baseline_mean"]) if v["mean_yield"] is not None else None
        p.add(f"{prod}: window mean yield {f3(v['mean_yield'])}, baseline mean {f3(v['baseline_mean'])}, "
              f"change {f3(drop)}, wafers {v['n_wafers']}, low-yield wafers {v['n_low']}")

    p.head(f"Product-adjusted change (alert window vs test days {lo}-{hi} before the alert)")
    for prod, v in pa["by_product"].items():
        p.add(f"{prod}: mean before {f3(v['mean_before'])}, alert {f3(v['mean_alert'])}, "
              f"change {f3(v['change'])}; share of wafers {f3(v['share_before'])} -> {f3(v['share_alert'])}")
    p.add(f"all products: change {f3(pa['overall_change'])}; change with the product mix held at the "
          f"pre-alert shares {f3(pa['mix_fixed_change'])}")

    p.head(f"Normal spread before the alert (test days {lo}-{hi}, {int(ref.quantile * 100)}th percentile "
           f"of |median diff| over the same comparisons)")
    parts = []
    for lv in ("tool", "chamber", "recipe"):
        v = ref.by_level.get(lv)
        parts.append(f"{lv} {f3(v)} ({ref.n_by_level[lv]} comparisons)" if v is not None
                     else f"{lv} no comparisons (tool value used)")
    p.add("; ".join(parts))
    if len(ev):
        refs = ev.groupby("event_type")["null_ref"].first()
        p.add("event before/after comparisons: " + "; ".join(f"{k} {f3(v)}" for k, v in refs.items()))

    p.head("Fab overview")
    steps = data.fab["steps"]
    by_area: dict[str, Counter] = {}
    for s in steps:
        c = by_area.setdefault(s["area"], Counter())
        c["steps"] += 1
        c["tools"] += len(s["tools"])
        c["chambers"] += sum(len(t["chambers"]) for t in s["tools"])
    p.add(f"{len(steps)} steps. " + "; ".join(
        f"{k} {v['steps']} steps/{v['tools']} tools/{v['chambers']} chambers" for k, v in sorted(by_area.items())))
    spec = [s["step_id"] for s in steps if s["product_specific_recipe"]]
    p.add(f"steps with product-specific recipes (P1->R1, P2->R2): {', '.join(spec) or 'none'}")

    p.head("Commonality candidates (stratified by product, BH q over all tests; "
           "ratio = |median diff| / normal spread)")
    elig = _eligible(scan, cfg)
    p.add(f"{len(scan)} comparisons; q <= {cfg.stats.no_cause_q} with lower yield: {len(elig)}; "
          f"of those beyond 1x normal spread: {int((elig['ratio'] > 1).sum())}, "
          f"beyond 3x: {int((elig['ratio'] > 3).sum())}")
    for level in ("tool", "chamber", "recipe"):
        sub = elig[elig["level"] == level].head(n)
        if sub.empty:
            p.add(f"[{level}] no candidate with q <= {cfg.stats.no_cause_q} and lower yield")
            continue
        for _, r in sub.iterrows():
            p.add(f"{_candidate_line(r)}, {r['ratio']:.2f}x normal spread")

    p.head(f"Event scan (every PM, chamber PM and recipe change: wafers of the target processed "
           f"{cfg.v2.event_scan_days:g} days after vs before, stratified by product, BH q)")
    evn = ev[ev["median_diff"] < 0].sort_values(["ratio", "q"], ascending=[False, True], kind="mergesort")
    p.add(f"{len(ev)} events compared; with lower yield after and q <= {cfg.stats.no_cause_q}: "
          f"{int((evn['q'] <= cfg.stats.no_cause_q).sum())}")
    for _, r in evn.head(n).iterrows():
        p.add(f"{r['event_id']} {r['event_type']} {r['target']} (step {r['step_id']}) at {ts(r['ts'])}: "
              f"n before {r['n_before']}, after {r['n_after']}, median diff {f3(r['median_diff'])}, "
              f"q {f3(r['q'])}, {r['ratio']:.2f}x normal spread")

    top = elig.head(n)
    if len(top) < MIN_TOP:
        top = scan[scan["median_diff"] < 0].sort_values("q", kind="mergesort").head(n)
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
        near = _events_near(data, ent, r["level"], pd.Timestamp(cp["ts"]))
        p.add(f"{ent}: " + ("; ".join(_event_text(e) for _, e in near.iterrows()) or "none"))

    p.head("Confounding among top candidates")
    ents = list(top["entity_id"])
    if len(ents) >= 2:
        cc = confounding_check(data, ents)
        p.add("phi: " + ", ".join(f"{q['a']}~{q['b']} {f3(q['phi'])}" for q in cc["pairs"]))
        for q in cc["pairs"]:
            if q["phi"] >= PHI_STRATIFY:
                g = {k: q[k] for k in ("a_only", "b_only", "both", "neither")}
                p.add(f"{q['a']} vs {q['b']} (phi {f3(q['phi'])}): " + ", ".join(
                    f"{k} n {v['n']} mean resid {f3(v['mean_yield'])} low {f3(v['low_rate'])}" for k, v in g.items()))
    else:
        p.add("fewer than two candidates")

    p.head("Tool x recipe interaction")
    seen = []
    for sid in top["step_id"]:
        if sid in seen or not data.steps[sid]["product_specific_recipe"]:
            continue
        seen.append(sid)
        it = interaction_test(data, sid)
        w = it["worst_cell"]
        cell = (f"most deviant cell {w['tool_id']} x {w['recipe_id']} n {w['n']}, "
                f"mean {f3(w['mean'])} vs additive {f3(w['additive_pred'])}") if w else "n/a"
        p.add(f"{sid}: interaction p {f3(it['p_interaction'])}; {cell}")
    if not seen:
        p.add("no top candidate is at a step with product-specific recipes")

    p.head("Daily product mix")
    w = data.all_wafers.copy()
    w["day"] = w["test_ts"].dt.floor("D")
    share = w.groupby("day")["product"].apply(lambda s: (s == "P2").mean())
    p.add("daily P2 share: " + ", ".join(f"{d:%m-%d} {f3(v)}" for d, v in share.items()))
    return p.render()
