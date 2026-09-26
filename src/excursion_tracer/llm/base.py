"""LLM 공통 인터페이스. 에이전트는 structured() 하나만 쓴다.

모델을 바꿀 때는 이 인터페이스를 따르는 어댑터만 새로 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class StructuredResponse:
    data: dict  # 스키마에 맞춘 JSON
    usage: Usage
    raw: str | None = None  # 디버깅용 원문


class SchemaError(Exception):
    """응답이 스키마 검증을 통과하지 못했다. 원문과 오류 내용, 쓴 토큰을 담는다."""

    def __init__(self, message: str, raw: str | None, usage: Usage):
        super().__init__(message)
        self.raw = raw
        self.usage = usage


@dataclass
class CallContext:
    """사용량 기록에 남길 호출 정보."""

    run_id: str = ""
    scenario_id: str = ""
    purpose: str = ""
    extra: dict = field(default_factory=dict)


class LLMClient(Protocol):
    model: str

    def structured(self, system: str, prompt: str, schema: type, max_tokens: int,
                   temperature: float, context: CallContext | None = None) -> StructuredResponse: ...
