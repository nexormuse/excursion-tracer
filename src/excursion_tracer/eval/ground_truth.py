"""시나리오 정답 파일(ground_truth.json) 읽기. 정답은 평가기만 읽는다."""

from __future__ import annotations

import json
from pathlib import Path

GROUND_TRUTH_FILE = "ground_truth.json"


def load_ground_truth(scenario_dir: str | Path) -> dict:
    path = Path(scenario_dir) / GROUND_TRUTH_FILE
    with path.open(encoding="utf-8") as f:
        return json.load(f)
