"""웨이퍼 수율 모델.

wafer_yield = clip(base[product] + lot_effect + Σ benign_offset(챔버)
                   + wafer_noise − Σ fault_effect, 0, 1)

benign_offset은 설비·챔버마다 가진 고유 차이(원인이 아님)이고, 계측 단계는
수율에 영향을 주지 않는다. 일부 설비는 평균은 같고 웨이퍼 잡음만 크다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from excursion_tracer.config import Config
from excursion_tracer.sim.fab import Fab
from excursion_tracer.sim.routing import Flow


@dataclass
class YieldState:
    chamber_offset: np.ndarray  # (전역 챔버 수,)
    noisy_tool: np.ndarray  # (전역 설비 수,) bool


def draw_yield_state(cfg: Config, fab: Fab, rng: np.random.Generator) -> YieldState:
    ym = cfg.yield_model
    active_tool = np.array([fab.steps[s].affects_yield for s in fab.tool_step])
    active_chamber = active_tool[fab.chamber_tool]
    offset = rng.normal(0.0, ym.benign_offset_sd, size=fab.n_chambers)
    offset[~active_chamber] = 0.0
    candidates = np.flatnonzero(active_tool)
    n_noisy = int(round(ym.noisy_tool_ratio * len(candidates)))
    noisy = np.zeros(fab.n_tools, dtype=bool)
    noisy[rng.choice(candidates, size=n_noisy, replace=False)] = True
    return YieldState(chamber_offset=offset, noisy_tool=noisy)


def base_yield(
    cfg: Config, fab: Fab, flow: Flow, state: YieldState, rng: np.random.Generator
) -> np.ndarray:
    """원인 효과를 빼기 전의 웨이퍼 수율 (L*W,). 웨이퍼 순서는 로트, 슬롯 순."""
    ym = cfg.yield_model
    L, W = flow.n_lots, flow.wafers_per_lot
    base = np.array([cfg.fab.products[p].base_yield for p in flow.products])
    lot_eff = rng.normal(0.0, ym.lot_sd, size=L)
    y = np.repeat(base[flow.lot_product] + lot_eff, W)
    y += state.chamber_offset[flow.chamber].sum(axis=2).reshape(L * W)
    n_noisy = np.repeat(state.noisy_tool[flow.tool].sum(axis=1), W)
    extra_sd = ym.wafer_sd * np.sqrt((ym.noisy_tool_sd_mult**2 - 1.0) * n_noisy)
    y += rng.normal(0.0, 1.0, size=L * W) * np.sqrt(ym.wafer_sd**2 + extra_sd**2)
    return y
