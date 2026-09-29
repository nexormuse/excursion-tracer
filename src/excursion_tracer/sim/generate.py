"""시나리오 세트 계획, 시나리오 한 개 생성(알림이 없으면 재생성), 폴더 출력.

시나리오 폴더
  fab.json lots.parquet history.parquet events.parquet wafers.parquet alert.json meta.json
  ground_truth.json (평가기 전용)
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from excursion_tracer.config import Config
from excursion_tracer.sim.fab import Fab, build_fab
from excursion_tracer.sim.faults import (
    ALLOWED_SETS,
    FAULT_CODES,
    NO_CAUSE_CODES,
    SINGLE_CODES,
    TEST_ONLY_CODES,
    Fault,
    f5_from_change,
    fault_effect,
    pick_f1,
    pick_f2,
    pick_f3,
    pick_f5_step,
    pick_f6,
)
from excursion_tracer.sim.monitor import Alert, evaluate_alert
from excursion_tracer.sim.routing import (
    HOURS_PER_DAY,
    Event,
    Flow,
    make_maintenance_events,
    pick_recipe_change,
    simulate_flow,
)
from excursion_tracer.sim.yield_model import base_yield, draw_yield_state

SIM_START = np.datetime64("2026-01-01T00:00:00", "s")
EFFECTS = ("large", "medium", "small")
STICKINESS = ("low", "high")
OBSERVED_FILES = (
    "fab.json",
    "lots.parquet",
    "history.parquet",
    "events.parquet",
    "wafers.parquet",
    "alert.json",
    "meta.json",
)
GROUND_TRUTH_FILE = "ground_truth.json"


@dataclass(frozen=True)
class ScenarioPlan:
    scenario_id: str
    seed: int
    fault_code: str
    effect: str | None  # large | medium | small, 원인 없음이면 None
    stickiness: str  # low | high


@dataclass
class Attempt:
    flow: Flow
    events: list[Event]
    faults: list[Fault]
    yields: np.ndarray  # (L*W,) 측정 여부와 무관하게 계산한 수율
    measured: np.ndarray  # (L*W,) bool
    test_h: np.ndarray  # (L*W,)
    alert: Alert | None


@dataclass
class ScenarioResult:
    plan: ScenarioPlan
    regen_count: int
    alert_level: str
    path: Path | None = None
    seconds: float = 0.0


def check_set_allows(set_name: str, fault_code: str) -> None:
    if fault_code not in FAULT_CODES:
        raise ValueError(f"모르는 원인 유형: {fault_code}")
    if fault_code in TEST_ONLY_CODES and set_name not in ALLOWED_SETS[fault_code]:
        raise ValueError(f"{fault_code}는 {', '.join(ALLOWED_SETS[fault_code])} 세트에서만 생성한다 "
                         f"(요청한 세트: {set_name})")


def _scaled_mix(mix: dict[str, int], n: int) -> dict[str, int]:
    total = sum(mix.values())
    if n == total:
        return dict(mix)
    raw = {k: v * n / total for k, v in mix.items()}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    rest = n - sum(out.values())
    order = sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))
    for k in order[:rest]:
        out[k] += 1
    return out


def _balanced(options: list, n: int, rng: np.random.Generator) -> list:
    out: list = []
    while len(out) < n:
        out.extend(options[i] for i in rng.permutation(len(options)))
    return out[:n]


def plan_set(cfg: Config, set_name: str, n: int | None = None) -> list[ScenarioPlan]:
    """세트의 시나리오 목록. 원인 유형과 난이도 칸을 고르게 배분하고 번호 순서를 섞는다."""
    spec = cfg.set_spec(set_name)
    n = spec.n if n is None else n
    mix = _scaled_mix(spec.mix, n)
    for code in mix:
        check_set_allows(set_name, code)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed_for(set_name, 0), 1]))
    cells = [(e, s) for e in EFFECTS for s in STICKINESS]
    filled = Counter({c: 0 for c in cells})
    items: list[tuple[str, str | None, str]] = []
    for code in sorted(mix):
        k = mix[code]
        if code in NO_CAUSE_CODES:
            items.extend((code, None, s) for s in _balanced(list(STICKINESS), k, rng))
            continue
        # 유형 안에서 칸을 고르게 채우고, 남는 몫은 전체에서 가장 덜 찬 칸에 준다.
        full, rest = divmod(k, len(cells))
        chosen = cells * full
        order = sorted(rng.permutation(len(cells)), key=lambda i: filled[cells[i]])
        chosen += [cells[i] for i in order[:rest]]
        filled.update(chosen)
        items.extend((code, e, s) for e, s in chosen)
    order = rng.permutation(len(items))
    plans = []
    for i, j in enumerate(order):
        seed = cfg.seed_for(set_name, i + 1)
        code, eff, stick = items[j]
        plans.append(ScenarioPlan(f"scn_{seed}", seed, code, eff, stick))
    return plans


def simulate(cfg: Config, fab: Fab, plan: ScenarioPlan, attempt: int) -> Attempt:
    """시나리오 한 번 시도. 같은 (seed, attempt)는 같은 결과를 낸다."""
    ss = np.random.SeedSequence([plan.seed, attempt])
    r_flow, r_event, r_yield, r_fault = (np.random.default_rng(c) for c in ss.spawn(4))
    code = plan.fault_code
    stick = cfg.faults.stickiness[plan.stickiness]
    delta = cfg.faults.effect[plan.effect] if plan.effect is not None else 0.0
    mix = None
    if code == "F0b":
        mc = cfg.faults.f0b_mix_change
        mix = (mc.from_release_day, mc.p2_share)

    flow = simulate_flow(cfg, fab, r_flow, stick, mix)
    events = make_maintenance_events(cfg, fab, r_event)
    lo, hi = cfg.flow.benign_recipe_changes
    n_changes = int(r_event.integers(lo, hi + 1))

    faults: list[Fault] = []
    if code == "F5":
        change = pick_recipe_change(cfg, fab, r_fault, pick_f5_step(fab, r_fault))
        events.append(change)
        faults.append(f5_from_change(change, delta))
        n_changes -= 1
    elif code == "F6":
        faults.append(pick_f6(cfg, fab, r_fault, delta, set(), flow))
    elif code in SINGLE_CODES or code == "F4":
        types = [code] if code != "F4" else sorted(r_fault.choice(SINGLE_CODES, 2, replace=False))
        used: set[int] = set()
        for t in types:
            if t == "F1":
                f = pick_f1(cfg, fab, r_fault, delta, events, used)
            elif t == "F2":
                f = pick_f2(cfg, fab, r_fault, delta, used)
            else:
                f = pick_f3(cfg, fab, r_fault, delta, used)
            used.add(f.step_index)
            faults.append(f)
    for _ in range(n_changes):
        events.append(pick_recipe_change(cfg, fab, r_event))

    state = draw_yield_state(cfg, fab, r_yield)
    y = base_yield(cfg, fab, flow, state, r_yield)
    for f in faults:
        y = y - fault_effect(cfg, fab, flow, f)
    y = np.clip(y, 0.0, 1.0)

    W = flow.wafers_per_lot
    end_h = cfg.flow.observe_end_day * HOURS_PER_DAY
    done = flow.track_out_h[:, -1]
    measured = np.repeat(done <= end_h, W)
    test_h = np.repeat(done, W)
    alert = evaluate_alert(cfg, y[measured], test_h[measured])
    return Attempt(flow, events, faults, y, measured, test_h, alert)


def _ts(hours: np.ndarray) -> np.ndarray:
    secs = np.round(np.asarray(hours, dtype=float) * 3600.0).astype(np.int64)
    return (SIM_START + secs.astype("timedelta64[s]")).astype("datetime64[us]")


def _iso(hours: float) -> str:
    return str(_ts(np.array([hours]))[0].astype("datetime64[s]"))


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _alert_json(cfg: Config, att: Attempt, products: list[str]) -> dict:
    a = att.alert
    assert a is not None
    flow = att.flow
    W = flow.wafers_per_lot
    prod = np.repeat(flow.lot_product, W)
    m = att.measured
    day = np.floor(att.test_h / HOURS_PER_DAY)
    in_win = m & (day >= a.window_start_day) & (day < a.window_end_day)
    lo, hi = cfg.monitor.baseline_test_days
    in_base = m & (day >= lo) & (day <= hi)
    by_product = {}
    for i, p in enumerate(products):
        w = in_win & (prod == i)
        b = in_base & (prod == i)
        by_product[p] = {
            "mean_yield": round(float(att.yields[w].mean()), 4) if w.any() else None,
            "baseline_mean": round(float(att.yields[b].mean()), 4) if b.any() else None,
            "n_wafers": int(w.sum()),
            "n_low": int((w & (att.yields < a.threshold)).sum()),
        }
    return {
        "level": a.level,
        "window_start": _iso(a.window_start_day * HOURS_PER_DAY),
        "window_end": _iso(a.window_end_day * HOURS_PER_DAY),
        "baseline": {"low_yield_threshold": round(a.threshold, 4), "low_rate": a.low_rate},
        "by_product": by_product,
    }


def _fault_json(fab: Fab, f: Fault) -> dict:
    extra = {"windows": [[_iso(a), _iso(b)] for a, b in f.windows]} if f.windows else {}
    return {
        "type": f.type,
        "step_id": fab.steps[f.step_index].step_id,
        "tool_id": None if f.tool is None else str(fab.tool_ids[f.tool]),
        "chamber_id": None if f.chamber is None else str(fab.chamber_ids[f.chamber]),
        "recipe_id": f.recipe_id,
        "onset_ts": _iso(f.onset_h),
        "delta": f.delta,
        **extra,
    }


def write_scenario(
    cfg: Config, fab: Fab, set_name: str, plan: ScenarioPlan, att: Attempt,
    regen_count: int, out_dir: Path, created_at: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    flow = att.flow
    L, W, S = flow.n_lots, flow.wafers_per_lot, fab.n_steps
    products = flow.products
    end_h = cfg.flow.observe_end_day * HOURS_PER_DAY

    lot_ids = np.array([f"L{i + 1:04d}" for i in range(L)], dtype=object)
    prod_names = np.array(products, dtype=object)
    wafer_ids = np.array(
        [f"{lot}-W{s + 1:02d}" for lot in lot_ids for s in range(W)], dtype=object
    )

    _write_json(out_dir / "fab.json", fab.to_json(products))

    done = flow.track_out_h[:, -1]
    complete = pd.Series(_ts(done)).where(done <= end_h)
    pd.DataFrame(
        {
            "lot_id": lot_ids,
            "product": prod_names[flow.lot_product],
            "release_ts": _ts(flow.release_h),
            "complete_ts": complete.to_numpy(),
        }
    ).to_parquet(out_dir / "lots.parquet", index=False)

    step_ids = np.array([st.step_id for st in fab.steps], dtype=object)
    recipe_tab = np.array(
        [[st.recipe_ids()[min(p, len(st.recipe_ids()) - 1)] for p in range(len(products))]
         for st in fab.steps],
        dtype=object,
    )
    step_done = flow.track_out_h <= end_h  # (L, S)
    li, wi, si = np.nonzero(np.broadcast_to(step_done[:, None, :], (L, W, S)))
    pd.DataFrame(
        {
            "lot_id": lot_ids[li],
            "wafer_id": wafer_ids[li * W + wi],
            "slot": (wi + 1).astype(np.int64),
            "step_id": step_ids[si],
            "tool_id": fab.tool_ids[flow.tool[li, si]],
            "chamber_id": fab.chamber_ids[flow.chamber[li, wi, si]],
            "recipe_id": recipe_tab[si, flow.lot_product[li]],
            "track_in_ts": _ts(flow.track_in_h[li, si]),
            "track_out_ts": _ts(flow.track_out_h[li, si]),
        }
    ).to_parquet(out_dir / "history.parquet", index=False)

    evs = sorted(att.events, key=lambda e: (e.ts_h, e.event_type, e.step_index,
                                            -1 if e.tool is None else e.tool,
                                            -1 if e.chamber is None else e.chamber))
    pd.DataFrame(
        {
            "event_id": [f"EV{i + 1:04d}" for i in range(len(evs))],
            "event_type": [e.event_type for e in evs],
            "step_id": [fab.steps[e.step_index].step_id for e in evs],
            "tool_id": [None if e.tool is None else str(fab.tool_ids[e.tool]) for e in evs],
            "chamber_id": [None if e.chamber is None else str(fab.chamber_ids[e.chamber]) for e in evs],
            "recipe_id": [e.recipe_id for e in evs],
            "ts": _ts(np.array([e.ts_h for e in evs])),
        }
    ).to_parquet(out_dir / "events.parquet", index=False)

    m = att.measured
    widx = np.flatnonzero(m)
    pd.DataFrame(
        {
            "wafer_id": wafer_ids[widx],
            "lot_id": lot_ids[widx // W],
            "product": prod_names[flow.lot_product[widx // W]],
            "yield": att.yields[widx],
            "test_ts": _ts(att.test_h[widx]),
        }
    ).to_parquet(out_dir / "wafers.parquet", index=False)

    _write_json(out_dir / "alert.json", _alert_json(cfg, att, products))
    _write_json(
        out_dir / "meta.json",
        {"scenario_id": plan.scenario_id, "set": set_name, "seed": plan.seed,
         "created_at": created_at},
    )
    _write_json(
        out_dir / GROUND_TRUTH_FILE,
        {
            "scenario_id": plan.scenario_id,
            "fault_code": plan.fault_code,
            "faults": [_fault_json(fab, f) for f in att.faults],
            "cell": {"effect": plan.effect, "stickiness": plan.stickiness},
            "regen_count": regen_count,
        },
    )


def run_until_alert(
    cfg: Config, fab: Fab, plan: ScenarioPlan, max_attempts: int = 200
) -> tuple[Attempt, int]:
    """알림(alarm 또는 warning)이 나올 때까지 다시 만든다. (시도, 폐기 횟수)를 돌려준다."""
    for attempt in range(max_attempts):
        att = simulate(cfg, fab, plan, attempt)
        if att.alert is not None:
            return att, attempt
    raise RuntimeError(f"{plan.scenario_id}: {max_attempts}번 시도해도 알림이 없다")


def generate_scenario(
    cfg: Config, fab: Fab, set_name: str, plan: ScenarioPlan, out_dir: Path,
    created_at: str | None = None,
) -> ScenarioResult:
    check_set_allows(set_name, plan.fault_code)
    t0 = time.perf_counter()
    att, regen = run_until_alert(cfg, fab, plan)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    created_at = created_at or datetime.now().isoformat(timespec="seconds")
    write_scenario(cfg, fab, set_name, plan, att, regen, out_dir, created_at)
    assert att.alert is not None
    return ScenarioResult(plan, regen, att.alert.level, out_dir, time.perf_counter() - t0)


def build_set_fab(cfg: Config, set_name: str) -> Fab:
    return build_fab(cfg, cfg.seed_for(set_name, 0))


def generate_set(
    cfg: Config, set_name: str, out_root: Path, n: int | None = None, progress=None
) -> list[ScenarioResult]:
    fab = build_set_fab(cfg, set_name)
    plans = plan_set(cfg, set_name, n)
    results = []
    it = plans if progress is None else progress(plans)
    for plan in it:
        results.append(
            generate_scenario(cfg, fab, set_name, plan, out_root / set_name / plan.scenario_id)
        )
    return results
