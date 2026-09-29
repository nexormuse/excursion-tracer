"""생성기 테스트용 시나리오를 임시 폴더에 한 번 만들어 공유한다.

실제 dev/test 세트와 겹치지 않도록 별도 seed(99xxxx)와 별도 팹 seed를 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from excursion_tracer.config import PROJECT_ROOT, Config, load_config
from excursion_tracer.sim.fab import Fab, build_fab
from excursion_tracer.sim.generate import ScenarioPlan, ScenarioResult, generate_scenario

FIXTURE_FAB_SEED = 990000

# (원인 유형, 효과 크기, stickiness). 유형마다 하나, 칸은 여러 곳에 흩어 놓는다.
FIXTURE_PLANS = [
    ("F0a", None, "low"),
    ("F0a", None, "high"),
    ("F0b", None, "low"),
    ("F0b", None, "high"),
    ("F1", "large", "high"),
    ("F1", "small", "low"),
    ("F2", "medium", "low"),
    ("F2", "large", "high"),
    ("F3", "medium", "high"),
    ("F3", "large", "low"),
    ("F4", "medium", "high"),
    ("F4", "large", "low"),
    ("F5", "small", "low"),
    ("F5", "medium", "high"),
    ("F6", "large", "low"),
    ("F6", "medium", "high"),
]
SET_OF = {"F5": "test", "F6": "test2"}


@dataclass
class Fixture:
    cfg: Config
    fab: Fab
    root: Path
    results: list[ScenarioResult]

    def dirs(self) -> list[Path]:
        return [r.path for r in self.results]


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def fixture_set(cfg: Config, tmp_path_factory) -> Fixture:
    root = tmp_path_factory.mktemp("scenarios")
    fab = build_fab(cfg, FIXTURE_FAB_SEED)
    results = []
    for i, (code, eff, stick) in enumerate(FIXTURE_PLANS):
        seed = FIXTURE_FAB_SEED + 1 + i
        plan = ScenarioPlan(f"scn_{seed}", seed, code, eff, stick)
        set_name = SET_OF.get(code, "dev")
        results.append(
            generate_scenario(cfg, fab, set_name, plan, root / plan.scenario_id,
                              created_at="2026-01-01T00:00:00")
        )
    return Fixture(cfg, fab, root, results)


def generated_dev_dirs() -> list[Path]:
    """data/dev 에 생성해 둔 시나리오 폴더 (없으면 빈 목록)."""
    base = PROJECT_ROOT / "data" / "dev"
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if (p / "meta.json").is_file())


def generated_test_dirs() -> list[Path]:
    """data/test 에 생성해 둔 시나리오 폴더 (없으면 빈 목록)."""
    base = PROJECT_ROOT / "data" / "test"
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if (p / "meta.json").is_file())


def generated_test2_dirs() -> list[Path]:
    """data/test2 에 생성해 둔 시나리오 폴더 (없으면 빈 목록)."""
    base = PROJECT_ROOT / "data" / "test2"
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if (p / "meta.json").is_file())
