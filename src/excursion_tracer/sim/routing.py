"""로트 투입, 단계별 설비·챔버 배정과 처리 시각, PM·레시피 변경 이벤트.

시각은 모두 시뮬레이션 기준일 00:00부터의 시간(hour, float)으로 다룬다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from excursion_tracer.config import Config
from excursion_tracer.sim.fab import Fab

HOURS_PER_DAY = 24.0


@dataclass
class Flow:
    products: list[str]
    lot_product: np.ndarray  # (L,) 제품 번호
    release_h: np.ndarray  # (L,)
    track_in_h: np.ndarray  # (L, S)
    track_out_h: np.ndarray  # (L, S)
    tool: np.ndarray  # (L, S) 전역 설비 번호
    chamber: np.ndarray  # (L, W, S) 전역 챔버 번호
    wafers_per_lot: int

    @property
    def n_lots(self) -> int:
        return len(self.release_h)


def simulate_flow(
    cfg: Config,
    fab: Fab,
    rng: np.random.Generator,
    stickiness: float,
    mix_change: tuple[int, float] | None = None,
) -> Flow:
    """로트를 투입하고 40개 단계를 차례로 통과시킨다.

    mix_change=(투입일, P2 비중)이면 그 투입일부터 제품 비중을 바꾼다.
    """
    fl = cfg.flow
    products = list(cfg.fab.products)
    shares = np.array([cfg.fab.products[p].share for p in products])
    n_lots = fl.release_days * fl.lots_per_day
    lot_idx = np.arange(n_lots)
    day = lot_idx // fl.lots_per_day
    slot_in_day = lot_idx % fl.lots_per_day
    release_h = (day + (slot_in_day + rng.random(n_lots)) / fl.lots_per_day) * HOURS_PER_DAY

    p_second = np.full(n_lots, shares[1] if len(shares) > 1 else 0.0)
    if mix_change is not None:
        from_day, share2 = mix_change
        p_second[day >= from_day] = share2
    lot_product = (rng.random(n_lots) < p_second).astype(np.int64)

    S = fab.n_steps
    W = fl.wafers_per_lot
    queue = rng.exponential(fl.queue_mean_hours, size=(n_lots, S))
    proc = rng.uniform(fl.proc_hours[0], fl.proc_hours[1], size=(n_lots, S))
    track_in = np.empty((n_lots, S))
    track_out = np.empty((n_lots, S))
    t = release_h.copy()
    for s in range(S):
        track_in[:, s] = t + queue[:, s]
        track_out[:, s] = track_in[:, s] + proc[:, s]
        t = track_out[:, s]

    tool_local = np.empty((n_lots, S), dtype=np.int64)
    tool_local[:, 0] = rng.integers(0, fab.steps[0].n_tools, size=n_lots)
    for s in range(1, S):
        n_t = fab.steps[s].n_tools
        rand = rng.integers(0, n_t, size=n_lots)
        sticky = rng.random(n_lots) < stickiness
        pref = fab.preferred[s][tool_local[:, s - 1]]
        tool_local[:, s] = np.where(sticky, pref, rand)
    offsets = np.array([st.tool_offset for st in fab.steps])
    tool = tool_local + offsets[None, :]

    # 다챔버 설비는 로트 안 웨이퍼를 챔버에 순환 배정한다 (시작 챔버 무작위).
    nch = fab.tool_n_chambers[tool]  # (L, S)
    start = np.floor(rng.random((n_lots, S)) * nch).astype(np.int64)
    slot = np.arange(W)[None, :, None]
    local = (start[:, None, :] + slot) % nch[:, None, :]
    chamber = fab.tool_chamber_offset[tool][:, None, :] + local

    return Flow(
        products=products,
        lot_product=lot_product,
        release_h=release_h,
        track_in_h=track_in,
        track_out_h=track_out,
        tool=tool,
        chamber=chamber,
        wafers_per_lot=W,
    )


@dataclass
class Event:
    event_type: str  # PM | CHAMBER_PM | RECIPE_CHANGE
    step_index: int
    ts_h: float
    tool: int | None = None  # 전역 설비 번호
    chamber: int | None = None  # 전역 챔버 번호
    recipe_id: str | None = None


def _periodic(rng: np.random.Generator, lo: float, hi: float, end_h: float) -> list[float]:
    t = rng.uniform(0, hi) * HOURS_PER_DAY
    out = []
    while t < end_h:
        out.append(float(t))
        t += rng.uniform(lo, hi) * HOURS_PER_DAY
    return out


def make_maintenance_events(cfg: Config, fab: Fab, rng: np.random.Generator) -> list[Event]:
    """설비 PM과 챔버 PM 일정. 설비마다 5~9일 간격, 다챔버 설비의 챔버는 50% 확률로 따로 PM."""
    fl = cfg.flow
    lo, hi = fl.pm_interval_days
    end_h = fl.observe_end_day * HOURS_PER_DAY
    events: list[Event] = []
    for t in range(fab.n_tools):
        s = int(fab.tool_step[t])
        for ts in _periodic(rng, lo, hi, end_h):
            events.append(Event("PM", s, ts, tool=t))
        n_ch = int(fab.tool_n_chambers[t])
        if n_ch < 2:
            continue
        for c in range(n_ch):
            if rng.random() < fl.chamber_pm_prob:
                ch = int(fab.tool_chamber_offset[t]) + c
                for ts in _periodic(rng, lo, hi, end_h):
                    events.append(Event("CHAMBER_PM", s, ts, tool=t, chamber=ch))
    return events


def pick_recipe_change(
    cfg: Config, fab: Fab, rng: np.random.Generator, step_index: int | None = None
) -> Event:
    """레시피 변경 이벤트 하나. 시각은 원인 시작 시각과 같은 범위에서 뽑는다."""
    if step_index is None:
        step_index = int(rng.integers(fab.n_steps))
    st = fab.steps[step_index]
    recipes = st.recipe_ids()
    recipe = recipes[int(rng.integers(len(recipes)))]
    lo, hi = cfg.faults.onset_day
    ts = float(rng.uniform(lo, hi) * HOURS_PER_DAY)
    return Event("RECIPE_CHANGE", step_index, ts, recipe_id=recipe)
