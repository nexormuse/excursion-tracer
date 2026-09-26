"""Gemini 어댑터 (google-genai SDK).

응답 형식을 JSON으로, 응답 스키마를 pydantic 모델의 JSON 스키마로 지정하고,
받은 JSON은 pydantic으로 다시 검증한다. 키는 .env의 GEMINI_API_KEY에서 읽는다.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import errors as genai_errors
from pydantic import ValidationError

from excursion_tracer.llm.base import CallContext, SchemaError, StructuredResponse, Usage
from excursion_tracer.llm.usage import UsageTracker, guarded_call


def _is_rate_limit(e: Exception) -> bool:
    return isinstance(e, genai_errors.APIError) and e.code == 429


def _is_daily_quota(e: Exception) -> bool:
    if not _is_rate_limit(e):
        return False
    text = f"{getattr(e, 'details', '')} {e}".lower()
    return "perday" in text or "per day" in text or "requests_per_day" in text


class GeminiClient:
    def __init__(self, model: str, tracker: UsageTracker, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY가 .env에 없다")
        self.model = model
        self.tracker = tracker
        self._client = genai.Client(api_key=key)

    def list_models(self) -> list[str]:
        """사용 가능한 모델 이름 목록 (generateContent 지원 모델).
        생성 요청이 아니므로 한도에는 세지 않고(counted=False) 사용량 기록에만 남긴다."""
        self.tracker.record(model=self.model, purpose="list_models", counted=False)
        names = []
        for m in self._client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                names.append(m.name.removeprefix("models/"))
        return sorted(names)

    def check_model(self) -> None:
        names = self.list_models()
        if self.model not in names:
            near = [n for n in names if "flash-lite" in n]
            raise RuntimeError(f"모델 {self.model}이 목록에 없다. flash-lite 계열: {near}")

    def structured(self, system: str, prompt: str, schema: type, max_tokens: int,
                   temperature: float, context: CallContext | None = None) -> StructuredResponse:
        ctx = context or CallContext()

        def do_request():
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "system_instruction": system,
                    "response_mime_type": "application/json",
                    "response_json_schema": schema.model_json_schema(),
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            um = resp.usage_metadata
            tin = (um.prompt_token_count or 0) if um else 0
            tout = (um.candidates_token_count or 0) if um else 0
            extra = {"thoughts_tokens": (um.thoughts_token_count or 0) if um else 0}
            finish = ""
            if resp.candidates:
                finish = str(resp.candidates[0].finish_reason or "")
            extra["finish_reason"] = finish
            return resp, tin, tout, extra

        resp = guarded_call(
            self.tracker, do_request, model=self.model, run_id=ctx.run_id,
            scenario_id=ctx.scenario_id, purpose=ctx.purpose,
            is_rate_limit=_is_rate_limit, is_daily_quota=_is_daily_quota,
        )
        um = resp.usage_metadata
        usage = Usage((um.prompt_token_count or 0) if um else 0,
                      (um.candidates_token_count or 0) if um else 0)
        raw = resp.text
        if raw is None:
            raise SchemaError("응답에 텍스트가 없다", None, usage)
        try:
            data = schema.model_validate_json(raw)
        except ValidationError as e:
            raise SchemaError(str(e), raw, usage) from e
        return StructuredResponse(data=data.model_dump(), usage=usage, raw=raw)
