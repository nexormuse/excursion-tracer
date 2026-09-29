"""요청 한도 관리와 사용량 기록 (logs/usage.jsonl).

- 하루 한도는 quota_reset_tz(미국 태평양 시간) 자정에 새로 채워진다. 그 날짜를 한도 창으로 쓴다.
- 호출 전에 현재 한도 창에서 쓴 요청 수가 rpd_limit × stop_at_ratio에 닿았으면 호출하지 않고
  QuotaExhausted를 낸다 (다음 초기화 시각을 한국 시간으로 알려 준다).
- 호출 간격은 60 / rpm_limit 초 이상. 최근 60초 요청 수도 rpm_limit을 넘지 않게 기다린다.
- 기록은 파일에 남기므로 프로세스를 다시 시작해도 한도 창 사용량이 이어진다.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from excursion_tracer.config import PROJECT_ROOT, LLMConfig

KST = ZoneInfo("Asia/Seoul")
DEFAULT_LOG = PROJECT_ROOT / "logs" / "usage.jsonl"


class QuotaExhausted(Exception):
    def __init__(self, used: int, cap: int, next_reset_kst: datetime, reason: str = ""):
        self.used = used
        self.cap = cap
        self.next_reset_kst = next_reset_kst
        msg = (f"요청 한도에 닿아 멈춘다: 이번 한도 창 사용 {used} / 멈춤 기준 {cap}. "
               f"다음 초기화 {next_reset_kst:%Y-%m-%d %H:%M} (한국 시간)")
        super().__init__(f"{msg}. {reason}" if reason else msg)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class UsageTracker:
    rpm_limit: int
    rpd_limit: int
    stop_at_ratio: float
    reset_tz: str
    log_path: Path = DEFAULT_LOG
    clock: Callable[[], datetime] = _utcnow
    sleep: Callable[[float], None] = time.sleep
    _recent: list[datetime] = field(default_factory=list)

    @classmethod
    def from_config(cls, llm: LLMConfig, log_path: Path | None = None, **kw) -> "UsageTracker":
        rpm, rpd = llm.require_rate_limits()
        return cls(rpm, rpd, llm.stop_at_ratio, llm.quota_reset_tz,
                   log_path=log_path or DEFAULT_LOG, **kw)

    # ------------------------------------------------------------ 한도 창
    @property
    def cap(self) -> int:
        return math.floor(self.rpd_limit * self.stop_at_ratio)

    def window(self, now: datetime | None = None) -> str:
        now = now or self.clock()
        return now.astimezone(ZoneInfo(self.reset_tz)).date().isoformat()

    def next_reset(self, now: datetime | None = None) -> datetime:
        now = now or self.clock()
        tz = ZoneInfo(self.reset_tz)
        local = now.astimezone(tz)
        nxt = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), tzinfo=tz)
        return nxt.astimezone(KST)

    def _records(self) -> list[dict]:
        if not self.log_path.is_file():
            return []
        out = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def used_in_window(self, now: datetime | None = None) -> int:
        w = self.window(now)
        return sum(1 for r in self._records() if r.get("window") == w and r.get("counted", True))

    def remaining(self, now: datetime | None = None) -> int:
        return max(0, self.cap - self.used_in_window(now))

    # ------------------------------------------------------------ 호출 전후
    def before_call(self) -> None:
        """한도를 확인하고, 필요하면 분당 한도에 맞춰 기다린다. 넘으면 QuotaExhausted."""
        now = self.clock()
        used = self.used_in_window(now)
        if used >= self.cap:
            raise QuotaExhausted(used, self.cap, self.next_reset(now))
        if not self._recent:
            self._recent = [datetime.fromisoformat(r["ts"]) for r in self._records()[-self.rpm_limit:]]
        interval = 60.0 / self.rpm_limit
        if self._recent:
            wait = interval - (now - self._recent[-1]).total_seconds()
            if wait > 0:
                self.sleep(wait)
                now = self.clock()
        minute = [t for t in self._recent if (now - t).total_seconds() < 60.0]
        if len(minute) >= self.rpm_limit:
            wait = 60.0 - (now - minute[0]).total_seconds()
            if wait > 0:
                self.sleep(wait)

    def record(self, *, model: str, run_id: str = "", scenario_id: str = "", purpose: str = "",
               input_tokens: int = 0, output_tokens: int = 0, status: str = "ok",
               counted: bool = True, extra: dict | None = None) -> dict:
        now = self.clock()
        rec = {
            "ts": now.isoformat(),
            "window": self.window(now),
            "run_id": run_id,
            "scenario_id": scenario_id,
            "purpose": purpose,
            "model": model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "status": status,
            "counted": counted,
            "cost_usd": 0.0,
        }
        if extra:
            rec.update(extra)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._recent.append(now)
        self._recent = self._recent[-max(self.rpm_limit, 1):]
        return rec


def guarded_call(
    tracker: UsageTracker,
    do_request: Callable[[], tuple[object, int, int, dict]],
    *,
    model: str,
    run_id: str = "",
    scenario_id: str = "",
    purpose: str = "",
    is_rate_limit: Callable[[Exception], bool] = lambda e: False,
    is_daily_quota: Callable[[Exception], bool] = lambda e: False,
    is_transient: Callable[[Exception], bool] = lambda e: False,
    max_retries: int = 5,
    base_delay: float = 2.0,
):
    """한도 확인 → 요청 → 기록. 분당 한도 오류(429)와 서버의 일시 오류(5xx)는 지수 백오프로
    최대 max_retries번 다시 시도하고, 하루 한도 소진 오류면 다시 시도하지 않고 QuotaExhausted를 낸다.

    do_request()는 (결과, 입력 토큰, 출력 토큰, 추가 기록)을 돌려준다.
    """
    ids = {"model": model, "run_id": run_id, "scenario_id": scenario_id, "purpose": purpose}
    for attempt in range(max_retries + 1):
        tracker.before_call()
        try:
            result, tin, tout, extra = do_request()
        except Exception as e:  # noqa: BLE001 - 어댑터별 오류를 분류해 처리한다
            if is_daily_quota(e):
                tracker.record(**ids, status="daily_quota_exhausted", extra={"error": str(e)[:300]})
                now = tracker.clock()
                raise QuotaExhausted(tracker.used_in_window(now), tracker.cap,
                                     tracker.next_reset(now), reason="제공사가 하루 한도 소진을 알렸다") from e
            if (is_rate_limit(e) or is_transient(e)) and attempt < max_retries:
                status = "rate_limited" if is_rate_limit(e) else "server_error"
                tracker.record(**ids, status=status, extra={"error": str(e)[:300]})
                tracker.sleep(base_delay * 2**attempt)
                continue
            tracker.record(**ids, status="error", extra={"error": str(e)[:300]})
            raise
        tracker.record(**ids, input_tokens=tin, output_tokens=tout, status="ok", extra=extra)
        return result
    raise RuntimeError("unreachable")
