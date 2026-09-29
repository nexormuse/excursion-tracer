"""보고서·응답 스키마. 구조화 출력이 안정적이도록 null과 자유 형식 객체를 쓰지 않는다.
"해당 없음"은 빈 문자열("")로 둔다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    evidence_id: str  # 근거 묶음 E#, 추가 확인 C# (기준선은 검정 이름)
    summary: str


class Hypothesis(BaseModel):
    rank: int = Field(ge=1, le=3)
    entity_type: Literal["tool", "chamber", "recipe", "tool_recipe"]
    step_id: str
    tool_id: str = ""
    chamber_id: str = ""
    recipe_id: str = ""
    onset: str = ""  # ISO 8601, 모르면 ""
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence] = Field(min_length=2)
    falsification_test: str  # 이 가설이 틀렸다면 무엇이 보여야 하는가
    recommended_action: str  # 엔지니어에게 권하는 다음 조치


class Report(BaseModel):
    scenario_id: str
    method: str  # "baseline" | "agent:<model>"
    verdict: Literal["cause_found", "no_equipment_cause"]
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=3)
    no_cause_explanation: str = ""  # 예: 제품 구성 변화, 한계선 근처의 우연한 변동
    confounding_notes: str = ""
    limitations: str = ""


CheckName = Literal["commonality_scan", "time_trend", "get_events", "before_after",
                    "interaction_test", "confounding_check", "drilldown", "product_mix"]


class CheckRequest(BaseModel):
    name: CheckName
    entity_id: str = ""
    entity_ids: list[str] = Field(default_factory=list)
    step_id: str = ""
    tool_id: str = ""
    level: Literal["", "tool", "chamber", "recipe"] = ""
    product: Literal["", "P1", "P2"] = ""
    event_ts: str = ""
    days: int = 3
    by: Literal["", "chamber", "recipe", "product"] = ""
    reason: str  # 이 확인으로 무엇을 가리려는지


class Round1(BaseModel):
    report: Report  # 잠정 보고서 (추가 확인이 없으면 최종)
    needs_checks: bool
    checks: list[CheckRequest] = Field(default_factory=list, max_length=3)


class Round1V2(BaseModel):
    """v2 1회차: 잠정 보고서와 반드시 1~3개의 확인 요청."""

    report: Report  # 잠정 보고서
    checks: list[CheckRequest] = Field(min_length=1, max_length=3)
