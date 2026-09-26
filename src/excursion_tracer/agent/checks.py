"""추가 확인: 1회차에서 요청한 확인(최대 3개)을 통계 코드로 실행하고 텍스트로 돌려준다.

결과마다 check_output_max_chars 이내로 자른다. 잘못된 이름·필드는 오류 메시지를 결과로 돌려준다.
"""

from __future__ import annotations

import pandas as pd

from excursion_tracer.agent.evidence import f3, ts
from excursion_tracer.agent.schema import CheckRequest
from excursion_tracer.config import Config
from excursion_tracer.stats.commonality import scan_all
from excursion_tracer.stats.confounding import confounding_check
from excursion_tracer.stats.data import ScenarioData
from excursion_tracer.stats.interaction import interaction_test
from excursion_tracer.stats.timewin import before_after, change_point, time_trend


class CheckError(ValueError):
    pass


def _entity(data: ScenarioData, entity_id: str) -> str:
    if not entity_id:
        raise CheckError("entity_id is required")
    try:
        level = data.entity_level(entity_id)
    except ValueError as e:
        raise CheckError(f"unknown entity id '{entity_id}'") from e
    col = data.column_for(entity_id)
    if not (data.history[col] == entity_id).any():
        raise CheckError(f"'{entity_id}' does not appear in the lot history")
    return level


def _scan(data, cfg, req):
    level = req.level or ""
    df = scan_all(data, product=req.product or None, fdr_method=cfg.stats.fdr_method)
    if level:
        df = df[df["level"] == level]
    rows = [f"{r['entity_id']} [{r['level']}] n {int(r['n_wafers'])} low {f3(r['low_rate_in'])} vs "
            f"{f3(r['low_rate_out'])} median diff {f3(r['median_diff'])} q {f3(r['q'])}"
            for _, r in df.head(cfg.stats.top_k).iterrows()]
    scope = f"level {level or 'all'}, product {req.product or 'all'}"
    return f"commonality_scan ({scope}): " + ("; ".join(rows) or "no candidates")


def _trend(data, cfg, req):
    _entity(data, req.entity_id)
    tt = time_trend(data, req.entity_id)
    days = "; ".join(f"{d:%m-%d} in {f3(r['mean_in'])} (n {int(r['n_in'])}) rest {f3(r['mean_out'])}"
                     for d, r in tt.set_index("day").iterrows() if r["n_in"] > 0)
    cp = change_point(data, req.entity_id, n_perm=cfg.stats.permutation_n)
    return (f"time_trend {req.entity_id} (daily mean yield residual by processing day): {days}. "
            f"Change point {ts(cp['ts'])}, before {f3(cp['mean_before'])}, after {f3(cp['mean_after'])}, "
            f"p {f3(cp['p'])}")


def _events(data, cfg, req):
    ev = data.events
    if req.tool_id:
        _entity(data, req.tool_id)
        step = req.tool_id.split("-")[0]
        m = (ev["tool_id"] == req.tool_id) | ((ev["event_type"] == "RECIPE_CHANGE") & (ev["step_id"] == step))
        what = req.tool_id
    elif req.step_id:
        if req.step_id not in data.steps:
            raise CheckError(f"unknown step '{req.step_id}'")
        m = ev["step_id"] == req.step_id
        what = req.step_id
    else:
        raise CheckError("get_events needs tool_id or step_id")
    rows = ev[m].sort_values("ts")
    items = []
    for _, e in rows.iterrows():
        who = e["chamber_id"] or e["tool_id"] or e["recipe_id"]
        items.append(f"{e['event_type']} {who} {ts(e['ts'])}")
    return f"get_events {what}: " + ("; ".join(items) or "none")


def _before_after(data, cfg, req):
    _entity(data, req.entity_id)
    if not req.event_ts:
        raise CheckError("event_ts is required")
    try:
        pd.Timestamp(req.event_ts)
    except ValueError as e:
        raise CheckError(f"bad event_ts '{req.event_ts}'") from e
    r = before_after(data, req.entity_id, req.event_ts, days=req.days)
    return (f"before_after {req.entity_id} around {ts(req.event_ts)} +/-{req.days} days: "
            f"before n {r['n_before']} mean resid {f3(r['mean_before'])}, "
            f"after n {r['n_after']} mean resid {f3(r['mean_after'])}, one-sided p {f3(r['p'])}")


def _interaction(data, cfg, req):
    if req.step_id not in data.steps:
        raise CheckError(f"unknown step '{req.step_id}'")
    r = interaction_test(data, req.step_id)
    w = r["worst_cell"]
    cell = (f"most deviant cell {w['tool_id']} x {w['recipe_id']} n {w['n']} mean {f3(w['mean'])} "
            f"vs additive {f3(w['additive_pred'])}") if w else r.get("note", "")
    return f"interaction_test {req.step_id}: interaction p {f3(r['p_interaction'])}; {cell}"


def _confounding(data, cfg, req):
    ids = list(dict.fromkeys(req.entity_ids))
    if not 2 <= len(ids) <= 4:
        raise CheckError("confounding_check needs 2 to 4 entity_ids")
    for e in ids:
        _entity(data, e)
    cc = confounding_check(data, ids)
    parts = []
    for q in cc["pairs"]:
        g = ", ".join(f"{k} n {q[k]['n']} resid {f3(q[k]['mean_yield'])} low {f3(q[k]['low_rate'])}"
                      for k in ("a_only", "b_only", "both", "neither"))
        parts.append(f"{q['a']} vs {q['b']} phi {f3(q['phi'])}: {g}")
    return "confounding_check: " + "; ".join(parts)


def _drilldown(data, cfg, req):
    level = _entity(data, req.entity_id)
    if req.by not in ("chamber", "recipe", "product"):
        raise CheckError("drilldown needs by = chamber, recipe or product")
    step = req.entity_id.split("-")[0]
    h = data.analysis_history
    hs = h[h["step_id"] == step]
    inside = hs[hs[data.column_for(req.entity_id)] == req.entity_id]
    rest = hs[hs[data.column_for(req.entity_id)] != req.entity_id]
    col = {"chamber": "chamber_id", "recipe": "recipe_id", "product": "product"}[req.by]
    if level == "chamber" and req.by == "chamber":
        raise CheckError("a chamber cannot be split by chamber")
    rows = []
    for key, g in inside.groupby(col):
        r = rest[rest[col] == key] if req.by != "chamber" else rest
        rows.append(f"{key} n {len(g)} median resid {f3(g['resid'].median())} low {f3(g['low'].mean())} "
                    f"(rest of step {f3(r['resid'].median())}, low {f3(r['low'].mean())})")
    return f"drilldown {req.entity_id} by {req.by} (alert window): " + "; ".join(rows)


def _product_mix(data, cfg, req):
    w = data.all_wafers.copy()
    w["day"] = w["test_ts"].dt.floor("D")
    g = w.groupby("day").agg(p2=("product", lambda s: (s == "P2").mean()))
    m = w.groupby(["day", "product"])["yield"].mean().unstack()
    rows = [f"{d:%m-%d} P2 share {f3(g.loc[d, 'p2'])} P1 {f3(m.loc[d].get('P1'))} P2 {f3(m.loc[d].get('P2'))}"
            for d in g.index]
    return "product_mix (by test day: P2 share, mean yield per product): " + "; ".join(rows)


RUNNERS = {
    "commonality_scan": _scan,
    "time_trend": _trend,
    "get_events": _events,
    "before_after": _before_after,
    "interaction_test": _interaction,
    "confounding_check": _confounding,
    "drilldown": _drilldown,
    "product_mix": _product_mix,
}


def run_check(data: ScenarioData, cfg: Config, req: CheckRequest) -> str:
    runner = RUNNERS.get(req.name)
    try:
        if runner is None:
            raise CheckError(f"unknown check '{req.name}'")
        text = runner(data, cfg, req)
    except CheckError as e:
        text = f"error in {req.name}: {e}"
    limit = cfg.agent.check_output_max_chars
    return text if len(text) <= limit else text[: limit - 15] + " ...[truncated]"


def run_checks(data: ScenarioData, cfg: Config, reqs: list[CheckRequest]) -> list[dict]:
    out = []
    for i, req in enumerate(reqs[: cfg.agent.max_checks]):
        out.append({"check_id": f"C{i + 1}", "request": req.model_dump(),
                    "result": run_check(data, cfg, req)})
    return out


def render_checks(results: list[dict]) -> str:
    return "\n".join(f"{r['check_id']} {r['result']}" for r in results)
