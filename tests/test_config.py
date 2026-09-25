"""설정 로딩과 검증 규칙 테스트."""

import pytest
import yaml
from pydantic import ValidationError

from excursion_tracer.config import DEFAULT_CONFIG_PATH, Config, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.project == "excursion-tracer"
    assert cfg.fab.n_steps == sum(cfg.fab.areas.values())
    assert cfg.sets.dev.n == sum(cfg.sets.dev.mix.values())
    assert cfg.sets.test.n == sum(cfg.sets.test.mix.values())
    assert cfg.llm.provider == "gemini"


def test_f5_is_test_only():
    cfg = load_config()
    assert "F5" not in cfg.sets.dev.mix
    assert "F5" in cfg.sets.test.mix


def test_rate_limits_must_be_filled_in_before_use():
    cfg = load_config()
    if cfg.llm.rpm_limit is None or cfg.llm.rpd_limit is None:
        with pytest.raises(ValueError):
            cfg.llm.require_rate_limits()
    else:
        assert cfg.llm.require_rate_limits() == (cfg.llm.rpm_limit, cfg.llm.rpd_limit)


def test_seed_for_derives_from_seed_base():
    cfg = load_config()
    assert cfg.seed_for("dev", 0) == cfg.seed_base["dev"]
    assert cfg.seed_for("test", 3) == cfg.seed_base["test"] + 3
    with pytest.raises(KeyError):
        cfg.seed_for("unknown")


def _raw():
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_unknown_key_is_rejected():
    raw = _raw()
    raw["llm"]["not_a_real_key"] = 1
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_area_sum_mismatch_is_rejected():
    raw = _raw()
    raw["fab"]["areas"]["etch"] += 1
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_set_mix_mismatch_is_rejected():
    raw = _raw()
    raw["sets"]["dev"]["mix"]["F1"] += 1
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_missing_config_file_is_reported():
    with pytest.raises(FileNotFoundError):
        load_config("config/does_not_exist.yaml")
