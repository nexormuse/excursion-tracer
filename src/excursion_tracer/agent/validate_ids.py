"""가설 ID 검증과 정규화.

가설의 step_id·tool_id·chamber_id·recipe_id가 fab.json에 실제로 있고 서로 소속이 맞는지 본다.
줄여 쓴 ID(예: tool_id "S12-T3" + chamber_id "C2")가 한 가지로만 해석되면 전체 ID로 바꾸고,
해석이 안 되거나 여러 가지로 해석되면 문제 목록에 올린다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

REQUIRED = {
    "tool": ("tool_id",),
    "chamber": ("chamber_id",),
    "recipe": ("recipe_id",),
    "tool_recipe": ("tool_id", "recipe_id"),
}


@dataclass(frozen=True)
class FabIndex:
    steps: frozenset[str]
    tool_step: dict[str, str]
    chamber_tool: dict[str, str]
    recipe_step: dict[str, str]

    @classmethod
    def from_fab(cls, fab: dict) -> "FabIndex":
        tool_step, chamber_tool, recipe_step = {}, {}, {}
        for s in fab["steps"]:
            for r in s["recipes"]:
                recipe_step[r] = s["step_id"]
            for t in s["tools"]:
                tool_step[t["tool_id"]] = s["step_id"]
                for c in t["chambers"]:
                    chamber_tool[c] = t["tool_id"]
        return cls(frozenset(s["step_id"] for s in fab["steps"]), tool_step, chamber_tool, recipe_step)


@dataclass
class IdResult:
    hypothesis: dict
    normalized: list[str] = field(default_factory=list)  # "chamber_id: C2 -> S12-T3-C2"
    problems: list[str] = field(default_factory=list)  # 해석할 수 없거나 모호한 ID

    @property
    def ok(self) -> bool:
        return not self.problems


def _step_of(value: str) -> str | None:
    m = re.match(r"^(S\d{2})(?:-|$)", value)
    return m.group(1) if m else None


def normalize_hypothesis(h: dict, idx: FabIndex) -> IdResult:
    h = dict(h)
    res = IdResult(h)

    def change(key: str, new: str) -> None:
        res.normalized.append(f"{key}: {h[key] or '(빈 값)'} -> {new}")
        h[key] = new

    # 단계: 비었거나 숫자만 쓴 경우 다른 전체 ID에서 얻는다
    step = h.get("step_id", "")
    if step and step not in idx.steps:
        cand = f"S{int(step):02d}" if step.isdigit() else None
        if cand in idx.steps:
            change("step_id", cand)
        else:
            res.problems.append(f"step_id '{step}' not in fab")
    for key in ("chamber_id", "tool_id", "recipe_id"):
        if not h.get("step_id") and h.get(key) and _step_of(h[key]) in idx.steps:
            change("step_id", _step_of(h[key]))
    step = h.get("step_id", "")

    # 설비
    tool = h.get("tool_id", "")
    if tool and tool not in idx.tool_step:
        cand = f"{step}-{tool}" if step and re.fullmatch(r"T\d+", tool) else None
        if cand in idx.tool_step:
            change("tool_id", cand)
        else:
            res.problems.append(f"tool_id '{tool}' not in fab")
    tool = h.get("tool_id", "")
    if tool in idx.tool_step and step and idx.tool_step[tool] != step:
        res.problems.append(f"tool_id '{tool}' is not at step {step}")

    # 챔버
    ch = h.get("chamber_id", "")
    if ch and ch not in idx.chamber_tool:
        cands = []
        if re.fullmatch(r"C\d+", ch):
            if tool in idx.tool_step:
                cands = [f"{tool}-{ch}"]
            elif step:
                cands = [t + "-" + ch for t, s in idx.tool_step.items() if s == step]
            cands = [c for c in cands if c in idx.chamber_tool]
        if len(cands) == 1:
            change("chamber_id", cands[0])
        elif len(cands) > 1:
            res.problems.append(f"chamber_id '{ch}' is ambiguous at step {step}: {', '.join(cands)}")
        else:
            res.problems.append(f"chamber_id '{ch}' not in fab")
    ch = h.get("chamber_id", "")
    if ch in idx.chamber_tool:
        owner = idx.chamber_tool[ch]
        if not tool:
            change("tool_id", owner)
        elif tool in idx.tool_step and owner != tool:
            res.problems.append(f"chamber_id '{ch}' does not belong to tool {tool}")
        if step and idx.tool_step[owner] != step:
            res.problems.append(f"chamber_id '{ch}' is not at step {step}")

    # 레시피
    rc = h.get("recipe_id", "")
    if rc and rc not in idx.recipe_step:
        cand = f"{step}-{rc}" if step and re.fullmatch(r"R\d+", rc) else None
        if cand in idx.recipe_step:
            change("recipe_id", cand)
        else:
            res.problems.append(f"recipe_id '{rc}' not in fab")
    rc = h.get("recipe_id", "")
    if rc in idx.recipe_step and step and idx.recipe_step[rc] != step:
        res.problems.append(f"recipe_id '{rc}' is not at step {step}")

    if not h.get("step_id"):
        res.problems.append("step_id is empty")
    for key in REQUIRED.get(h.get("entity_type", ""), ()):
        if not h.get(key):
            res.problems.append(f"{key} is required for entity_type {h.get('entity_type')}")
    res.hypothesis = h
    return res


def normalize_report(report: dict, fab: dict) -> tuple[dict, list[str], list[str]]:
    """보고서의 모든 가설을 검사한다. (정규화된 보고서, 정규화 목록, 문제 목록)."""
    idx = FabIndex.from_fab(fab)
    hyps, normalized, problems = [], [], []
    for h in report.get("hypotheses", []):
        r = normalize_hypothesis(h, idx)
        hyps.append(r.hypothesis)
        normalized += [f"rank {h.get('rank')}: {x}" for x in r.normalized]
        problems += [f"rank {h.get('rank')}: {x}" for x in r.problems]
    return {**report, "hypotheses": hyps}, normalized, problems
