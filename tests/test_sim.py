"""가상 팹 생성기 검증 (§5.1.8): 재현성, 출력 스키마, 원인 주입, 누출, 알림, 세트 구성."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from excursion_tracer.eval.ground_truth import load_ground_truth
from excursion_tracer.eval.injection import fault_weights, injection_check
from excursion_tracer.sim import generate
from excursion_tracer.sim.fab import build_fab
from excursion_tracer.sim.generate import (
    EFFECTS,
    GROUND_TRUTH_FILE,
    OBSERVED_FILES,
    STICKINESS,
    ScenarioPlan,
    check_set_allows,
    generate_scenario,
    plan_set,
    simulate,
)
from excursion_tracer.sim.monitor import evaluate_alert

from .conftest import generated_dev_dirs, generated_test_dirs

OBSERVE_END = pd.Timestamp("2026-01-01") + pd.Timedelta(days=21)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 재현성

def test_same_seed_same_files(cfg, tmp_path):
    fab = build_fab(cfg, 990000)
    plan = ScenarioPlan("scn_990101", 990101, "F4", "medium", "high")
    a = generate_scenario(cfg, fab, "dev", plan, tmp_path / "a", created_at="2026-01-01T00:00:00")
    b = generate_scenario(cfg, build_fab(cfg, 990000), "dev", plan, tmp_path / "b",
                          created_at="2026-02-02T00:00:00")
    for name in OBSERVED_FILES + (GROUND_TRUTH_FILE,):
        if name == "meta.json":
            continue
        assert _hash(a.path / name) == _hash(b.path / name), name
    ma = json.loads((a.path / "meta.json").read_text())
    mb = json.loads((b.path / "meta.json").read_text())
    ma.pop("created_at"), mb.pop("created_at")
    assert ma == mb


def test_different_seed_different_files(cfg, tmp_path):
    fab = build_fab(cfg, 990000)
    a = generate_scenario(cfg, fab, "dev", ScenarioPlan("scn_1", 990201, "F1", "large", "low"),
                          tmp_path / "a")
    b = generate_scenario(cfg, fab, "dev", ScenarioPlan("scn_1", 990202, "F1", "large", "low"),
                          tmp_path / "b")
    assert _hash(a.path / "wafers.parquet") != _hash(b.path / "wafers.parquet")


def test_fab_is_deterministic_per_seed(cfg):
    assert build_fab(cfg, 1000).to_json(["P1", "P2"]) == build_fab(cfg, 1000).to_json(["P1", "P2"])
    assert build_fab(cfg, 1000).to_json(["P1", "P2"]) != build_fab(cfg, 5000).to_json(["P1", "P2"])


# ---------------------------------------------------------------- 출력 스키마 (부록 E)

def test_output_files_and_schema(fixture_set):
    for d in fixture_set.dirs():
        names = {p.name for p in d.iterdir()}
        assert names == set(OBSERVED_FILES) | {GROUND_TRUTH_FILE}

        lots = pd.read_parquet(d / "lots.parquet")
        assert list(lots.columns) == ["lot_id", "product", "release_ts", "complete_ts"]
        assert lots["lot_id"].str.fullmatch(r"L\d{4}").all()
        assert set(lots["product"]) <= {"P1", "P2"}
        assert pd.api.types.is_datetime64_any_dtype(lots["release_ts"])
        assert pd.api.types.is_datetime64_any_dtype(lots["complete_ts"])

        hist = pd.read_parquet(d / "history.parquet")
        assert list(hist.columns) == ["lot_id", "wafer_id", "slot", "step_id", "tool_id",
                                      "chamber_id", "recipe_id", "track_in_ts", "track_out_ts"]
        assert hist["wafer_id"].str.fullmatch(r"L\d{4}-W\d{2}").all()
        assert hist["slot"].between(1, 25).all()
        assert hist["step_id"].str.fullmatch(r"S\d{2}").all()
        assert (hist["tool_id"].str.split("-").str[0] == hist["step_id"]).all()
        assert (hist["chamber_id"].str.rsplit("-", n=1).str[0] == hist["tool_id"]).all()
        assert hist["recipe_id"].str.fullmatch(r"S\d{2}-R\d").all()
        assert (hist["track_out_ts"] > hist["track_in_ts"]).all()

        ev = pd.read_parquet(d / "events.parquet")
        assert list(ev.columns) == ["event_id", "event_type", "step_id", "tool_id",
                                    "chamber_id", "recipe_id", "ts"]
        assert set(ev["event_type"]) <= {"PM", "CHAMBER_PM", "RECIPE_CHANGE"}

        waf = pd.read_parquet(d / "wafers.parquet")
        assert list(waf.columns) == ["wafer_id", "lot_id", "product", "yield", "test_ts"]
        assert waf["yield"].between(0, 1).all()
        assert (waf["test_ts"] <= OBSERVE_END).all()

        alert = json.loads((d / "alert.json").read_text())
        assert set(alert) == {"level", "window_start", "window_end", "baseline", "by_product"}
        assert alert["level"] in ("alarm", "warning")
        assert set(alert["baseline"]) == {"low_yield_threshold", "low_rate"}
        for v in alert["by_product"].values():
            assert set(v) == {"mean_yield", "baseline_mean", "n_wafers", "n_low"}

        meta = json.loads((d / "meta.json").read_text())
        assert set(meta) == {"scenario_id", "set", "seed", "created_at"}
        assert meta["scenario_id"] == d.name

        gt = load_ground_truth(d)
        assert set(gt) == {"scenario_id", "fault_code", "faults", "cell", "regen_count"}
        assert set(gt["cell"]) == {"effect", "stickiness"}


def test_ground_truth_matches_fault_code(fixture_set):
    expected_n = {"F0a": 0, "F0b": 0, "F1": 1, "F2": 1, "F3": 1, "F4": 2, "F5": 1, "F6": 1}
    for d in fixture_set.dirs():
        gt = load_ground_truth(d)
        faults = gt["faults"]
        assert len(faults) == expected_n[gt["fault_code"]]
        if gt["fault_code"] == "F4":
            assert len({f["type"] for f in faults}) == 2
            assert len({f["step_id"] for f in faults}) == 2
            assert {f["type"] for f in faults} <= {"F1", "F2", "F3"}
        lo, hi = fixture_set.cfg.faults.onset_day
        for f in faults:
            onset = pd.Timestamp(f["onset_ts"])
            day = (onset - pd.Timestamp("2026-01-01")) / pd.Timedelta(days=1)
            assert lo <= day <= hi
            assert f["delta"] == fixture_set.cfg.faults.effect[gt["cell"]["effect"]]


def test_f1_onset_is_a_real_pm(fixture_set):
    for d in fixture_set.dirs():
        gt = load_ground_truth(d)
        ev = pd.read_parquet(d / "events.parquet")
        for f in gt["faults"]:
            if f["type"] != "F1":
                continue
            onset = pd.Timestamp(f["onset_ts"])
            pm = ev[(ev["ts"] == onset) & (
                ((ev["event_type"] == "PM") & (ev["tool_id"] == f["tool_id"]))
                | ((ev["event_type"] == "CHAMBER_PM") & (ev["chamber_id"] == f["chamber_id"]))
            )]
            assert len(pm) >= 1


def test_f5_onset_is_a_recipe_change_at_common_recipe_step(fixture_set):
    for d in fixture_set.dirs():
        gt = load_ground_truth(d)
        if gt["fault_code"] != "F5":
            continue
        f = gt["faults"][0]
        ev = pd.read_parquet(d / "events.parquet")
        rc = ev[(ev["event_type"] == "RECIPE_CHANGE") & (ev["ts"] == pd.Timestamp(f["onset_ts"]))]
        assert list(rc["recipe_id"]) == [f["recipe_id"]]
        fab = json.loads((d / "fab.json").read_text())
        step = next(s for s in fab["steps"] if s["step_id"] == f["step_id"])
        assert not step["product_specific_recipe"] and step["area"] != "metro"


# ---------------------------------------------------------------- 원인 주입

MIN_HIT_WAFERS = 500


def test_injected_effect_is_exact(fixture_set, monkeypatch):
    """원인을 뺀 같은 seed의 반사실 수율과 비교하면, 정답 요인을 거친 웨이퍼만 δ×가중치만큼 낮다."""
    cfg, fab = fixture_set.cfg, fixture_set.fab
    for r in fixture_set.results:
        gt = load_ground_truth(r.path)
        if not gt["faults"]:
            continue
        waf = pd.read_parquet(r.path / "wafers.parquet")
        hist = pd.read_parquet(r.path / "history.parquet")
        set_name = {"F5": "test", "F6": "test2"}.get(r.plan.fault_code, "dev")
        check_set_allows(set_name, r.plan.fault_code)
        with_fault = simulate(cfg, fab, r.plan, r.regen_count)
        with monkeypatch.context() as m:
            m.setattr(generate, "fault_effect", lambda *a: 0.0)
            without = simulate(cfg, fab, r.plan, r.regen_count)
        diff = (without.yields - with_fault.yields)[with_fault.measured]
        expected = np.zeros(len(waf))
        for f in gt["faults"]:
            w = fault_weights(hist, f, OBSERVE_END).reindex(waf["wafer_id"]).fillna(0.0)
            expected += f["delta"] * w.to_numpy()
        y0 = without.yields[with_fault.measured]
        y1 = with_fault.yields[with_fault.measured]
        unclipped = (y0 < 1.0) & (y1 > 0.0)
        assert unclipped.mean() > 0.95
        np.testing.assert_allclose(diff[unclipped], expected[unclipped], atol=1e-5)


def _injection_rows(dirs):
    rows = []
    for d in dirs:
        gt = load_ground_truth(d)
        for c in injection_check(d, OBSERVE_END):
            rows.append({"scenario": d.name, "code": gt["fault_code"], **c})
    return rows


def _assert_injection(rows):
    checked = [r for r in rows if r["n_hit"] >= MIN_HIT_WAFERS]
    assert checked
    bad = [r for r in checked if abs(r["measured"] / r["delta"] - 1) > 0.3]
    assert not bad, bad


def test_injected_effect_size_near_delta(fixture_set):
    """정답 요인을 거친 웨이퍼(시작 시각 이후)와 나머지의 수율 차이가 δ의 ±30% 안."""
    _assert_injection(_injection_rows(fixture_set.dirs()))


@pytest.mark.skipif(not generated_dev_dirs(), reason="data/dev 가 아직 없다")
def test_injected_effect_size_near_delta_dev_set():
    _assert_injection(_injection_rows(generated_dev_dirs()))


@pytest.mark.skipif(not generated_test_dirs(), reason="data/test 가 아직 없다")
def test_injected_effect_size_near_delta_test_set():
    _assert_injection(_injection_rows(generated_test_dirs()))


# ---------------------------------------------------------------- 누출 방지

FORBIDDEN_WORDS = (
    "fault", "inject", "truth", "ground", "cause", "delta", "stick", "regen", "attempt",
    "drift", "degrad", "anomal", "culprit", "effect", "cell", "difficult", "benign",
    "decoy", "noisy", "planted", "onset", "label", "answer",
)
FAULT_CODE_RE = re.compile(r"\bF[0-9][ab]?\b", re.IGNORECASE)


def _strings_in_json(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings_in_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings_in_json(v)
    elif isinstance(obj, str):
        yield obj


def _strings_in_parquet(path: Path):
    df = pd.read_parquet(path)
    yield from df.columns
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            yield from df[col].dropna().astype(str).unique()


def observed_strings(scenario_dir: Path):
    for name in OBSERVED_FILES:
        p = scenario_dir / name
        if name.endswith(".json"):
            yield from _strings_in_json(json.loads(p.read_text(encoding="utf-8")))
        else:
            yield from _strings_in_parquet(p)


def _assert_no_leak_words(dirs):
    for d in dirs:
        for s in set(observed_strings(d)):
            low = s.lower()
            hits = [w for w in FORBIDDEN_WORDS if w in low]
            assert not hits, f"{d.name}: '{s}' 에 금지 단어 {hits}"
            assert not FAULT_CODE_RE.search(s), f"{d.name}: '{s}' 에 원인 코드"


def test_leak_checker_catches_planted_word(tmp_path, fixture_set):
    src = fixture_set.dirs()[0]
    dst = tmp_path / src.name
    dst.mkdir()
    for name in OBSERVED_FILES:
        (dst / name).write_bytes((src / name).read_bytes())
    alert = json.loads((dst / "alert.json").read_text())
    alert["note"] = "Injected fault at S12"
    (dst / "alert.json").write_text(json.dumps(alert))
    with pytest.raises(AssertionError):
        _assert_no_leak_words([dst])


def test_observed_files_have_no_cause_words(fixture_set):
    _assert_no_leak_words(fixture_set.dirs())


@pytest.mark.skipif(not generated_dev_dirs(), reason="data/dev 가 아직 없다")
def test_observed_files_have_no_cause_words_dev_set():
    _assert_no_leak_words(generated_dev_dirs())


@pytest.mark.skipif(not generated_test_dirs(), reason="data/test 가 아직 없다")
def test_observed_files_have_no_cause_words_test_set():
    _assert_no_leak_words(generated_test_dirs())


def test_meta_has_only_allowed_keys(fixture_set):
    for d in fixture_set.dirs() + generated_dev_dirs() + generated_test_dirs():
        meta = json.loads((d / "meta.json").read_text())
        assert set(meta) == {"scenario_id", "set", "seed", "created_at"}
        assert meta["seed"] == int(meta["scenario_id"].split("_")[1])


def test_recipe_change_count_does_not_reveal_f5(fixture_set):
    lo, hi = fixture_set.cfg.flow.benign_recipe_changes
    for d in fixture_set.dirs():
        ev = pd.read_parquet(d / "events.parquet")
        assert lo <= int((ev["event_type"] == "RECIPE_CHANGE").sum()) <= hi


# ---------------------------------------------------------------- 알림

def _assert_alerts(dirs):
    assert dirs
    for d in dirs:
        alert = json.loads((d / "alert.json").read_text())
        assert alert["level"] in ("alarm", "warning"), d.name


def test_every_scenario_has_alert(fixture_set):
    _assert_alerts(fixture_set.dirs())


@pytest.mark.skipif(not generated_dev_dirs(), reason="data/dev 가 아직 없다")
def test_every_dev_scenario_has_alert():
    _assert_alerts(generated_dev_dirs())


@pytest.mark.skipif(not generated_test_dirs(), reason="data/test 가 아직 없다")
def test_every_test_scenario_has_alert():
    _assert_alerts(generated_test_dirs())


def _synthetic(cfg, rates: dict[int, float], n_per_day=400, seed=0):
    """측정일마다 n_per_day장. rates에 없는 날은 기준 분포 그대로."""
    rng = np.random.default_rng(seed)
    ys, hs = [], []
    for day in range(2, cfg.flow.observe_end_day):
        y = rng.uniform(0.8, 1.0, n_per_day)
        if day in rates:
            n_low = int(rates[day] * n_per_day)
            y[:n_low] = rng.uniform(0.5, 0.79, n_low)
        ys.append(y)
        hs.append(np.full(n_per_day, day * 24.0 + 12.0))
    return np.concatenate(ys), np.concatenate(hs)


def test_alarm_rule(cfg):
    y, h = _synthetic(cfg, {12: 0.3, 13: 0.3})
    a = evaluate_alert(cfg, y, h)
    assert a.level == "alarm" and a.first_day == 12
    assert a.window_start_day == 12 - cfg.monitor.pre_window_days
    assert a.window_end_day == cfg.flow.observe_end_day


def test_single_bad_day_is_not_alarm(cfg):
    y, h = _synthetic(cfg, {12: 0.3})
    a = evaluate_alert(cfg, y, h)
    assert a.level == "warning" and a.first_day in (10, 11, 12)


def test_no_alert_when_flat(cfg):
    y, h = _synthetic(cfg, {})
    assert evaluate_alert(cfg, y, h) is None


def test_warning_picks_worst_three_days(cfg):
    y, h = _synthetic(cfg, {15: 0.05, 16: 0.05, 17: 0.05}, n_per_day=300)
    a = evaluate_alert(cfg, y, h)
    assert a.level == "warning" and a.first_day == 15
    assert a.window_start_day == 15 - cfg.monitor.pre_window_days


# ---------------------------------------------------------------- 세트 구성

def test_dev_plan_matches_config(cfg):
    plans = plan_set(cfg, "dev")
    assert Counter(p.fault_code for p in plans) == Counter(cfg.sets.dev.mix)
    assert [p.seed for p in plans] == [cfg.seed_for("dev", i + 1) for i in range(len(plans))]
    assert all(p.scenario_id == f"scn_{p.seed}" for p in plans)
    assert "F5" not in {p.fault_code for p in plans}


def test_cells_are_balanced(cfg):
    for set_name in ("dev", "test"):
        plans = plan_set(cfg, set_name)
        for code in {p.fault_code for p in plans}:
            group = [p for p in plans if p.fault_code == code]
            if code in ("F0a", "F0b"):
                assert all(p.effect is None for p in group)
                c = Counter(p.stickiness for p in group)
                assert max(c.values()) - min(c.get(s, 0) for s in STICKINESS) <= 1
            else:
                c = Counter((p.effect, p.stickiness) for p in group)
                cells = [(e, s) for e in EFFECTS for s in STICKINESS]
                counts = [c.get(k, 0) for k in cells]
                assert max(counts) - min(counts) <= 1


def test_cells_are_balanced_across_cause_types(cfg):
    for set_name in ("dev", "test"):
        c = Counter((p.effect, p.stickiness) for p in plan_set(cfg, set_name) if p.effect)
        counts = [c.get((e, s), 0) for e in EFFECTS for s in STICKINESS]
        assert max(counts) - min(counts) <= 1, (set_name, counts)


def test_fault_order_is_shuffled(cfg):
    codes = [p.fault_code for p in plan_set(cfg, "dev")]
    runs = sum(1 for a, b in zip(codes, codes[1:]) if a != b)
    assert runs > len(set(codes)) * 2


def test_test_plan_has_f5_and_scales(cfg):
    assert Counter(p.fault_code for p in plan_set(cfg, "test"))["F5"] == cfg.sets.test.mix["F5"]
    small = Counter(p.fault_code for p in plan_set(cfg, "test", n=80))
    assert sum(small.values()) == 80 and small["F5"] == 12


def test_f6_windows_and_onset(fixture_set):
    lo, hi = fixture_set.cfg.v2.f6.n_windows
    wlo, whi = fixture_set.cfg.v2.f6.window_hours
    end = pd.Timestamp("2026-01-01") + pd.Timedelta(days=fixture_set.cfg.flow.observe_end_day)
    seen = 0
    for d in fixture_set.dirs():
        gt = load_ground_truth(d)
        if gt["fault_code"] != "F6":
            continue
        seen += 1
        f = gt["faults"][0]
        wins = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in f["windows"]]
        assert lo <= len(wins) <= hi
        assert wins[0][0] == pd.Timestamp(f["onset_ts"])
        for a, b in wins:
            assert wlo - 0.01 <= (b - a) / pd.Timedelta(hours=1) <= whi + 0.01
            assert b < end
        assert all(wins[i][1] <= wins[i + 1][0] for i in range(len(wins) - 1))
        assert f["chamber_id"].rsplit("-", 1)[0] == f["tool_id"]
    assert seen == 2


def test_test2_plan(cfg):
    plans = plan_set(cfg, "test2")
    assert Counter(p.fault_code for p in plans) == Counter(cfg.test2_set.mix)
    assert plans[0].scenario_id == "scn_7001" or min(p.seed for p in plans) == cfg.test2_set.seed_base + 1
    assert [p.seed for p in plans] == [cfg.test2_set.seed_base + i + 1 for i in range(len(plans))]


def test_f6_only_in_test2(cfg, tmp_path):
    for s in ("dev", "test"):
        with pytest.raises(ValueError):
            check_set_allows(s, "F6")
    check_set_allows("test2", "F6")
    check_set_allows("test2", "F5")
    assert "F6" not in {p.fault_code for p in plan_set(cfg, "dev") + plan_set(cfg, "test")}


def test_f5_is_rejected_outside_test(cfg, tmp_path):
    with pytest.raises(ValueError):
        check_set_allows("dev", "F5")
    fab = build_fab(cfg, 990000)
    with pytest.raises(ValueError):
        generate_scenario(cfg, fab, "dev", ScenarioPlan("scn_1", 990301, "F5", "large", "low"),
                          tmp_path / "x")
    assert not (tmp_path / "x").exists()
