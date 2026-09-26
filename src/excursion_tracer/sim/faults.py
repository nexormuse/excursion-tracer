"""원인 라이브러리 F0a·F0b·F1·F2·F3·F4·F5: 위치·시작 시각을 고르고 수율 효과를 계산한다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from excursion_tracer.config import Config
from excursion_tracer.sim.fab import Fab
from excursion_tracer.sim.routing import HOURS_PER_DAY, Event, Flow

FAULT_CODES = ("F0a", "F0b", "F1", "F2", "F3", "F4", "F5")
NO_CAUSE_CODES = ("F0a", "F0b")
TEST_ONLY_CODES = ("F5",)
SINGLE_CODES = ("F1", "F2", "F3")


@dataclass
class Fault:
    type: str
    step_index: int
    onset_h: float
    delta: float
    tool: int | None = None  # 전역 설비 번호
    chamber: int | None = None  # 전역 챔버 번호
    recipe_id: str | None = None


def _yield_steps(fab: Fab, **flags) -> list[int]:
    out = []
    for st in fab.steps:
        if not st.affects_yield:
            continue
        if "product_specific" in flags and st.product_specific != flags["product_specific"]:
            continue
        out.append(st.index)
    return out


def _onset(cfg: Config, rng: np.random.Generator) -> float:
    lo, hi = cfg.faults.onset_day
    return float(rng.uniform(lo, hi) * HOURS_PER_DAY)


def pick_f1(
    cfg: Config, fab: Fab, rng: np.random.Generator, delta: float,
    events: list[Event], exclude_steps: set[int],
) -> Fault:
    """정비 후 챔버 열화: 시작 시각 범위 안에 PM이 있는 다챔버 설비의 챔버를 고른다."""
    lo, hi = (d * HOURS_PER_DAY for d in cfg.faults.onset_day)
    candidates: list[tuple[int, float]] = []
    for ev in events:
        if ev.event_type not in ("PM", "CHAMBER_PM") or not lo <= ev.ts_h <= hi:
            continue
        if ev.step_index in exclude_steps or not fab.steps[ev.step_index].affects_yield:
            continue
        n_ch = int(fab.tool_n_chambers[ev.tool])
        if n_ch < 2:
            continue
        if ev.event_type == "CHAMBER_PM":
            candidates.append((ev.chamber, ev.ts_h))
        else:
            off = int(fab.tool_chamber_offset[ev.tool])
            candidates.extend((off + c, ev.ts_h) for c in range(n_ch))
    if not candidates:
        raise RuntimeError("F1을 넣을 PM 이벤트가 없다")
    chamber, ts = candidates[int(rng.integers(len(candidates)))]
    tool = int(fab.chamber_tool[chamber])
    return Fault("F1", int(fab.tool_step[tool]), ts, delta, tool=tool, chamber=chamber)


def pick_f2(
    cfg: Config, fab: Fab, rng: np.random.Generator, delta: float, exclude_steps: set[int]
) -> Fault:
    """설비×레시피 조합: 제품별 레시피 단계의 한 설비가 한 레시피로 처리한 웨이퍼."""
    steps = [s for s in _yield_steps(fab, product_specific=True) if s not in exclude_steps]
    s = steps[int(rng.integers(len(steps)))]
    st = fab.steps[s]
    tool = st.tool_offset + int(rng.integers(st.n_tools))
    recipes = st.recipe_ids()
    recipe = recipes[int(rng.integers(len(recipes)))]
    return Fault("F2", s, _onset(cfg, rng), delta, tool=tool, recipe_id=recipe)


def pick_f3(
    cfg: Config, fab: Fab, rng: np.random.Generator, delta: float, exclude_steps: set[int]
) -> Fault:
    """드리프트: 한 설비의 영향이 시작 시각부터 관측 종료까지 0 → δ로 선형 증가."""
    steps = [s for s in _yield_steps(fab) if s not in exclude_steps]
    s = steps[int(rng.integers(len(steps)))]
    st = fab.steps[s]
    tool = st.tool_offset + int(rng.integers(st.n_tools))
    return Fault("F3", s, _onset(cfg, rng), delta, tool=tool)


def pick_f5_step(fab: Fab, rng: np.random.Generator) -> int:
    steps = _yield_steps(fab, product_specific=False)
    return steps[int(rng.integers(len(steps)))]


def f5_from_change(change: Event, delta: float) -> Fault:
    """레시피 버전 변경: 변경 이후 그 단계를 거친 모든 웨이퍼."""
    return Fault("F5", change.step_index, change.ts_h, delta, recipe_id=change.recipe_id)


def fault_effect(cfg: Config, fab: Fab, flow: Flow, fault: Fault) -> np.ndarray:
    """웨이퍼별 수율 감소량 (L*W,)."""
    L, W = flow.n_lots, flow.wafers_per_lot
    s = fault.step_index
    t_in = flow.track_in_h[:, s]  # (L,)
    after = np.repeat(t_in >= fault.onset_h, W)
    if fault.type == "F1":
        hit = flow.chamber[:, :, s].reshape(L * W) == fault.chamber
        return np.where(hit & after, fault.delta, 0.0)
    if fault.type == "F2":
        r = fab.steps[s].recipe_ids().index(fault.recipe_id)
        lot_hit = (flow.tool[:, s] == fault.tool) & (flow.lot_product == r)
        return np.where(np.repeat(lot_hit, W) & after, fault.delta, 0.0)
    if fault.type == "F3":
        end_h = cfg.flow.observe_end_day * HOURS_PER_DAY
        frac = np.clip((t_in - fault.onset_h) / (end_h - fault.onset_h), 0.0, 1.0)
        lot_eff = np.where(flow.tool[:, s] == fault.tool, fault.delta * frac, 0.0)
        return np.repeat(lot_eff, W)
    if fault.type == "F5":
        return np.where(after, fault.delta, 0.0)
    raise ValueError(f"효과를 계산할 수 없는 원인 유형: {fault.type}")
