"""팹 구조: 공정 단계, 영역, 설비, 챔버, 레시피, 선호 설비 매핑.

팹 구조는 세트마다 seed 하나로 만들고 그 세트의 모든 시나리오가 공유한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from excursion_tracer.config import Config

# 한 모듈의 기본 순서. 증착 → 노광 → 식각 → 세정이 반복된다.
_CORE = ("depo", "litho", "etch", "clean")
# 기본 모듈에 들어가지 못한 단계는 이 영역 뒤에 끼운다 (None이면 모듈 맨 앞).
_ANCHOR = {
    "cmp": ("depo",),
    "implant": ("etch", "litho"),
    "metro": ("litho", "etch"),
    "depo": (None,),
    "etch": ("etch",),
    "clean": ("etch", "implant"),
    "litho": ("clean",),
}


@dataclass(frozen=True)
class Step:
    index: int
    step_id: str
    area: str
    product_specific: bool
    tool_ids: tuple[str, ...]
    chamber_ids: tuple[tuple[str, ...], ...]  # 설비별 챔버 ID
    tool_offset: int  # 전역 설비 번호의 시작

    @property
    def n_tools(self) -> int:
        return len(self.tool_ids)

    @property
    def affects_yield(self) -> bool:
        return self.area != "metro"

    def recipe_ids(self) -> tuple[str, ...]:
        if self.product_specific:
            return (f"{self.step_id}-R1", f"{self.step_id}-R2")
        return (f"{self.step_id}-R1",)


@dataclass(frozen=True)
class Fab:
    steps: tuple[Step, ...]
    # 전역 설비 번호 기준 배열
    tool_step: np.ndarray  # 설비가 속한 단계 번호
    tool_ids: np.ndarray  # 설비 ID (object)
    tool_n_chambers: np.ndarray
    tool_chamber_offset: np.ndarray  # 전역 챔버 번호의 시작
    # 전역 챔버 번호 기준 배열
    chamber_tool: np.ndarray
    chamber_ids: np.ndarray  # 챔버 ID (object)
    # 단계 s의 선호 설비: preferred[s][이전 단계의 로컬 설비 번호] = 단계 s의 로컬 설비 번호
    preferred: tuple[np.ndarray, ...]

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def n_tools(self) -> int:
        return len(self.tool_ids)

    @property
    def n_chambers(self) -> int:
        return len(self.chamber_ids)

    def to_json(self, products: list[str]) -> dict:
        steps = []
        for st in self.steps:
            steps.append(
                {
                    "step_id": st.step_id,
                    "area": st.area,
                    "recipes": list(st.recipe_ids()),
                    "product_specific_recipe": st.product_specific,
                    "tools": [
                        {"tool_id": t, "chambers": list(ch)}
                        for t, ch in zip(st.tool_ids, st.chamber_ids)
                    ],
                }
            )
        return {
            "n_steps": self.n_steps,
            "products": products,
            "product_recipe": {p: f"R{i + 1}" for i, p in enumerate(products)},
            "steps": steps,
        }


def _area_sequence(counts: dict[str, int], rng: np.random.Generator) -> list[str]:
    remaining = dict(counts)
    n_mod = max(1, remaining.get("litho", 1))
    modules: list[list[str]] = []
    for _ in range(n_mod):
        mod = []
        for area in _CORE:
            if remaining.get(area, 0) > 0:
                mod.append(area)
                remaining[area] -= 1
        modules.append(mod)
    extras = [a for a, c in remaining.items() for _ in range(c)]
    extras = [extras[i] for i in rng.permutation(len(extras))]
    for area in extras:
        mod = modules[rng.integers(n_mod)]
        anchors = _ANCHOR.get(area, (None,))
        anchor = anchors[rng.integers(len(anchors))]
        if anchor is None:
            mod.insert(0, area)
        elif anchor in mod:
            pos = len(mod) - 1 - mod[::-1].index(anchor)
            mod.insert(pos + 1, area)
        else:
            mod.append(area)
    return [a for mod in modules for a in mod]


def build_fab(cfg: Config, seed: int) -> Fab:
    fc = cfg.fab
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    areas = _area_sequence(dict(fc.areas), rng)
    n = len(areas)
    n_specific = int(round(fc.product_specific_recipe_ratio * n))
    specific = set(rng.choice(n, size=n_specific, replace=False).tolist())

    steps: list[Step] = []
    tool_step, tool_ids, tool_nch, tool_choff = [], [], [], []
    chamber_tool, chamber_ids = [], []
    for i, area in enumerate(areas):
        step_id = f"S{i + 1:02d}"
        lo, hi = fc.tools_per_step[area]
        n_tools = int(rng.integers(lo, hi + 1))
        clo, chi = fc.chambers_per_tool.get(area, fc.chambers_per_tool["default"])
        t_ids, c_ids = [], []
        offset = len(tool_ids)
        for t in range(n_tools):
            tid = f"{step_id}-T{t + 1}"
            n_ch = int(rng.integers(clo, chi + 1))
            chs = tuple(f"{tid}-C{c + 1}" for c in range(n_ch))
            t_ids.append(tid)
            c_ids.append(chs)
            tool_step.append(i)
            tool_ids.append(tid)
            tool_nch.append(n_ch)
            tool_choff.append(len(chamber_ids))
            for cid in chs:
                chamber_tool.append(len(tool_ids) - 1)
                chamber_ids.append(cid)
        steps.append(
            Step(i, step_id, area, i in specific, tuple(t_ids), tuple(c_ids), offset)
        )

    preferred = [np.zeros(0, dtype=np.int64)]
    for i in range(1, n):
        n_prev, n_cur = steps[i - 1].n_tools, steps[i].n_tools
        reps = -(-n_prev // n_cur)
        targets = np.concatenate([rng.permutation(n_cur) for _ in range(reps)])[:n_prev]
        preferred.append(rng.permutation(targets))

    return Fab(
        steps=tuple(steps),
        tool_step=np.asarray(tool_step),
        tool_ids=np.asarray(tool_ids, dtype=object),
        tool_n_chambers=np.asarray(tool_nch),
        tool_chamber_offset=np.asarray(tool_choff),
        chamber_tool=np.asarray(chamber_tool),
        chamber_ids=np.asarray(chamber_ids, dtype=object),
        preferred=tuple(preferred),
    )
