"""수율 알림 규칙 (alarm / warning).

- 기준 구간(측정일 3~7일차) 수율의 10번째 백분위수 미만을 저수율 웨이퍼로 본다.
- 기준 구간 이후 측정일마다 저수율 비율을 기준 비율과 이항검정한다 (단측).
- alarm: p < alarm.binom_p 가 연속 일수만큼 이어지면. 구간 = 첫 위반일 − pre_window_days ~ 관측 종료.
- warning: alarm이 없을 때, 저수율 비율이 가장 높은 연속 window_days일의 p < warning.binom_p 이면.
  구간 = 그 첫날 − pre_window_days ~ 관측 종료.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

from excursion_tracer.config import Config


@dataclass
class Alert:
    level: str  # alarm | warning
    window_start_day: int
    window_end_day: int
    threshold: float
    low_rate: float
    first_day: int  # 알림을 일으킨 첫 측정일


@dataclass
class DailyCounts:
    days: np.ndarray
    n: np.ndarray
    n_low: np.ndarray
    p: np.ndarray


def _pvalue(k: np.ndarray, n: np.ndarray, rate: float) -> np.ndarray:
    return binom.sf(k - 1, n, rate)


def daily_counts(cfg: Config, test_day: np.ndarray, is_low: np.ndarray) -> DailyCounts:
    first = cfg.monitor.baseline_test_days[1] + 1
    days = np.arange(first, cfg.flow.observe_end_day)
    n = np.array([np.sum(test_day == d) for d in days])
    k = np.array([np.sum(is_low & (test_day == d)) for d in days])
    p = np.where(n > 0, _pvalue(k, n, cfg.monitor.low_yield_quantile), 1.0)
    return DailyCounts(days, n, k, p)


def low_yield_threshold(cfg: Config, yields: np.ndarray, test_day: np.ndarray) -> float:
    lo, hi = cfg.monitor.baseline_test_days
    base = yields[(test_day >= lo) & (test_day <= hi)]
    if len(base) == 0:
        raise ValueError("기준 구간에 측정된 웨이퍼가 없다")
    return float(np.quantile(base, cfg.monitor.low_yield_quantile))


def evaluate_alert(cfg: Config, yields: np.ndarray, test_h: np.ndarray) -> Alert | None:
    """측정된 웨이퍼의 수율과 측정 시각(hour)으로 알림을 판정한다. 없으면 None."""
    mc = cfg.monitor
    test_day = np.floor(test_h / 24.0).astype(np.int64)
    thr = low_yield_threshold(cfg, yields, test_day)
    dc = daily_counts(cfg, test_day, yields < thr)
    end_day = cfg.flow.observe_end_day

    viol = (dc.p < mc.alarm.binom_p) & (dc.n > 0)
    run = mc.alarm.consecutive_days
    for i in range(len(dc.days) - run + 1):
        if viol[i : i + run].all():
            d0 = int(dc.days[i])
            return Alert("alarm", d0 - mc.pre_window_days, end_day, thr, mc.low_yield_quantile, d0)

    w = mc.warning.window_days
    best_rate, best_i = -1.0, None
    for i in range(len(dc.days) - w + 1):
        n = dc.n[i : i + w].sum()
        if n == 0:
            continue
        rate = dc.n_low[i : i + w].sum() / n
        if rate > best_rate:
            best_rate, best_i = rate, i
    if best_i is not None:
        n = int(dc.n[best_i : best_i + w].sum())
        k = int(dc.n_low[best_i : best_i + w].sum())
        if _pvalue(np.array(k), np.array(n), mc.low_yield_quantile) < mc.warning.binom_p:
            d0 = int(dc.days[best_i])
            return Alert("warning", d0 - mc.pre_window_days, end_day, thr, mc.low_yield_quantile, d0)
    return None
