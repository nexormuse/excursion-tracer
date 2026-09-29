"""config/default.yaml을 pydantic으로 검증해 읽는다.

모든 모듈은 설정값을 이 모듈을 통해서만 가져온다. 알 수 없는 키나 범위를
벗어난 값은 읽는 시점에 오류로 막는다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import yaml
from dotenv import load_dotenv
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

T = TypeVar("T", int, float)


def _ordered(v: tuple[T, T]) -> tuple[T, T]:
    lo, hi = v
    if lo > hi:
        raise ValueError(f"범위의 하한이 상한보다 크다: {list(v)}")
    return v


IntRange = Annotated[tuple[int, int], AfterValidator(_ordered)]
FloatRange = Annotated[tuple[float, float], AfterValidator(_ordered)]

AreaName = Literal["litho", "etch", "depo", "cmp", "implant", "clean", "metro"]


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Product(Base):
    base_yield: float = Field(gt=0, le=1)
    share: float = Field(gt=0, le=1)


class FabConfig(Base):
    n_steps: int = Field(gt=0)
    areas: dict[AreaName, int]
    tools_per_step: dict[AreaName, IntRange]
    chambers_per_tool: dict[str, IntRange]
    product_specific_recipe_ratio: float = Field(ge=0, le=1)
    products: dict[str, Product]

    @model_validator(mode="after")
    def _check(self) -> "FabConfig":
        total = sum(self.areas.values())
        if total != self.n_steps:
            raise ValueError(f"areas 합({total})이 n_steps({self.n_steps})와 다르다")
        missing = set(self.areas) - set(self.tools_per_step)
        if missing:
            raise ValueError(f"tools_per_step에 빠진 영역: {sorted(missing)}")
        if "default" not in self.chambers_per_tool:
            raise ValueError("chambers_per_tool에 default 항목이 있어야 한다")
        share = sum(p.share for p in self.products.values())
        if abs(share - 1.0) > 1e-9:
            raise ValueError(f"제품 투입 비율 합이 1이 아니다: {share}")
        return self


class FlowConfig(Base):
    release_days: int = Field(gt=0)
    lots_per_day: int = Field(gt=0)
    wafers_per_lot: int = Field(gt=0)
    proc_hours: FloatRange
    queue_mean_hours: float = Field(gt=0)
    observe_end_day: int = Field(gt=0)
    pm_interval_days: IntRange
    chamber_pm_prob: float = Field(ge=0, le=1)
    benign_recipe_changes: IntRange

    @model_validator(mode="after")
    def _check(self) -> "FlowConfig":
        if self.observe_end_day <= self.release_days:
            raise ValueError("observe_end_day는 release_days보다 커야 한다")
        return self


class YieldModelConfig(Base):
    lot_sd: float = Field(ge=0)
    wafer_sd: float = Field(ge=0)
    benign_offset_sd: float = Field(ge=0)
    noisy_tool_ratio: float = Field(ge=0, le=1)
    noisy_tool_sd_mult: float = Field(ge=1)


class MixChange(Base):
    from_release_day: int = Field(ge=0)
    p2_share: float = Field(ge=0, le=1)


class FaultsConfig(Base):
    onset_day: IntRange
    effect: dict[Literal["large", "medium", "small"], float]
    stickiness: dict[Literal["low", "high"], float]
    f0b_mix_change: MixChange

    @model_validator(mode="after")
    def _check(self) -> "FaultsConfig":
        for name, value in self.effect.items():
            if not 0 < value < 1:
                raise ValueError(f"effect.{name}는 0과 1 사이여야 한다: {value}")
        for name, value in self.stickiness.items():
            if not 0 <= value <= 1:
                raise ValueError(f"stickiness.{name}는 0과 1 사이여야 한다: {value}")
        return self


class AlarmRule(Base):
    binom_p: float = Field(gt=0, lt=1)
    consecutive_days: int = Field(gt=0)


class WarningRule(Base):
    window_days: int = Field(gt=0)
    binom_p: float = Field(gt=0, lt=1)


class MonitorConfig(Base):
    baseline_test_days: IntRange
    low_yield_quantile: float = Field(gt=0, lt=1)
    alarm: AlarmRule
    warning: WarningRule
    pre_window_days: int = Field(ge=0)
    max_discard_ratio: float = Field(gt=0, le=1)


class SetSpec(Base):
    n: int = Field(gt=0)
    mix: dict[str, int]

    @model_validator(mode="after")
    def _check(self) -> "SetSpec":
        total = sum(self.mix.values())
        if total != self.n:
            raise ValueError(f"mix 합({total})이 n({self.n})과 다르다")
        return self


class SetsConfig(Base):
    dev: SetSpec
    test: SetSpec

    @model_validator(mode="after")
    def _check(self) -> "SetsConfig":
        if "F5" in self.dev.mix:
            raise ValueError("F5는 test 세트 전용이다")
        return self


class StatsConfig(Base):
    stratify_by_product: bool
    fdr_method: str
    permutation_n: int = Field(gt=0)
    no_cause_q: float = Field(gt=0, lt=1)
    no_cause_min_effect: float = Field(ge=0)
    top_k: int = Field(gt=0)


class AgentConfig(Base):
    evidence_max_chars: int = Field(gt=0)
    check_output_max_chars: int = Field(gt=0)
    max_checks: int = Field(gt=0)
    max_followup_rounds: int = Field(ge=0)
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(gt=0)
    report_language: Literal["en", "ko"]
    retry_on_invalid: int = Field(ge=0)


class LLMConfig(Base):
    provider: str
    model: str
    rpm_limit: int | None = Field(default=None, gt=0)
    rpd_limit: int | None = Field(default=None, gt=0)
    stop_at_ratio: float = Field(gt=0, le=1)
    quota_reset_tz: str
    workers: int = Field(gt=0)

    def require_rate_limits(self) -> tuple[int, int]:
        """한도를 적지 않은 채로는 LLM을 부르지 못하게 막는다."""
        if self.rpm_limit is None or self.rpd_limit is None:
            raise ValueError(
                "llm.rpm_limit과 llm.rpd_limit이 config/default.yaml에 있어야 한다"
            )
        return self.rpm_limit, self.rpd_limit


class EvalConfig(Base):
    tau: float = Field(ge=0, le=1)
    onset_tolerance_days: int = Field(ge=0)
    repeat_runs: int = Field(gt=0)
    repeat_sample: int = Field(gt=0)
    showcase_n: int = Field(gt=0)


class ConfByRatio(Base):
    below_1_5: float = Field(ge=0, le=1)
    from_1_5_to_3: float = Field(ge=0, le=1)
    above_3: float = Field(ge=0, le=1)


class F6Config(Base):
    n_windows: IntRange
    window_hours: FloatRange


class V2Config(Base):
    null_window_test_days: IntRange
    null_quantile: float = Field(gt=0, lt=1)
    baseline_no_cause_ratio: float = Field(gt=0)
    baseline_conf_by_ratio: ConfByRatio
    event_scan_days: float = Field(gt=0)
    evidence_max_chars: int = Field(gt=0)
    round1_max_tokens: int = Field(gt=0)
    round2_max_tokens: int = Field(gt=0)
    min_checks: int = Field(ge=0)
    f6: F6Config


class Test2Set(SetSpec):
    seed_base: int


class Config(Base):
    project: str
    seed_base: dict[str, int]
    fab: FabConfig
    flow: FlowConfig
    yield_model: YieldModelConfig
    faults: FaultsConfig
    monitor: MonitorConfig
    sets: SetsConfig
    stats: StatsConfig
    agent: AgentConfig
    llm: LLMConfig
    eval: EvalConfig
    v2: V2Config
    test2_set: Test2Set

    @model_validator(mode="after")
    def _check(self) -> "Config":
        missing = set(self.sets.model_dump()) - set(self.seed_base)
        if missing:
            raise ValueError(f"seed_base에 빠진 세트: {sorted(missing)}")
        if "F6" in self.sets.dev.mix or "F6" in self.sets.test.mix:
            raise ValueError("F6는 test2 세트 전용이다")
        return self

    def set_spec(self, set_name: str) -> "SetSpec":
        if set_name == "test2":
            return self.test2_set
        return getattr(self.sets, set_name)

    def seed_for(self, set_name: str, index: int = 0) -> int:
        """세트 이름과 시나리오 순번으로 seed를 만든다."""
        if set_name == "test2":
            return self.test2_set.seed_base + index
        if set_name not in self.seed_base:
            raise KeyError(f"모르는 세트: {set_name}")
        return self.seed_base[set_name] + index


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """YAML 설정을 읽어 검증한 Config를 돌려준다."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"설정 파일이 없다: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"설정 파일의 최상위가 매핑이 아니다: {cfg_path}")
    load_dotenv(PROJECT_ROOT / ".env")
    return Config.model_validate(raw)
