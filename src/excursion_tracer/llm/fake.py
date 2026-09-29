"""시험용 가짜 클라이언트: 실제 요청 없이 같은 한도 관리와 기록을 거친다."""

from __future__ import annotations

import json
from typing import Callable

from pydantic import ValidationError

from excursion_tracer.llm.base import CallContext, SchemaError, StructuredResponse, Usage
from excursion_tracer.llm.usage import UsageTracker, guarded_call


class FakeRateLimit(Exception):
    pass


class FakeDailyQuota(Exception):
    pass


class FakeServerError(Exception):
    pass


class FakeClient:
    """responder(system, prompt, schema) -> JSON 문자열 또는 예외."""

    def __init__(self, tracker: UsageTracker, responder: Callable[[str, str, type], str],
                 model: str = "fake-model"):
        self.model = model
        self.tracker = tracker
        self.responder = responder
        self.calls: list[dict] = []

    def structured(self, system: str, prompt: str, schema: type, max_tokens: int,
                   temperature: float, context: CallContext | None = None) -> StructuredResponse:
        ctx = context or CallContext()

        def do_request():
            self.calls.append({"system": system, "prompt": prompt, "schema": schema.__name__})
            raw = self.responder(system, prompt, schema)
            return raw, len(prompt) // 4, len(raw) // 4, {}

        raw = guarded_call(
            self.tracker, do_request, model=self.model, run_id=ctx.run_id,
            scenario_id=ctx.scenario_id, purpose=ctx.purpose,
            is_rate_limit=lambda e: isinstance(e, FakeRateLimit),
            is_daily_quota=lambda e: isinstance(e, FakeDailyQuota),
            is_transient=lambda e: isinstance(e, FakeServerError),
        )
        usage = Usage(len(prompt) // 4, len(raw) // 4)
        try:
            data = schema.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as e:
            raise SchemaError(str(e), raw, usage) from e
        return StructuredResponse(data=data.model_dump(), usage=usage, raw=raw)
