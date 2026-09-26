"""에이전트 실행: 근거 묶음 → 1회차(잠정 보고서 + 추가 확인 요청) → 확인 실행 → 2회차(최종 보고서).

- 스키마 검증에 실패하면 오류 내용을 붙여 1회 다시 요청하고, 그래도 실패하면 invalid.
- 2회차만 실패하면 1회차 잠정 보고서를 최종으로 쓰고 그 사실을 기록한다.
- 기록: runs/<run_id>/<scenario_id>.json. 이미 기록이 있는 시나리오는 건너뛴다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from excursion_tracer.agent.checks import render_checks, run_checks
from excursion_tracer.agent.evidence import build_evidence
from excursion_tracer.agent.numcheck import check_report
from excursion_tracer.agent.prompts import (
    RETRY_SUFFIX, ROUND1_TEMPLATE, ROUND2_TEMPLATE, prompt_hash, system_prompt,
)
from excursion_tracer.agent.schema import CheckRequest, Report, Round1
from excursion_tracer.config import PROJECT_ROOT, Config
from excursion_tracer.llm.base import CallContext, LLMClient, SchemaError
from excursion_tracer.stats.data import load_scenario

RUNS_DIR = PROJECT_ROOT / "runs"


def make_run_id(model: str, day: date | None = None) -> str:
    day = day or date.today()
    return f"{day.isoformat()}-{model}-{prompt_hash()[:8]}"


def find_run_id(model: str, runs_dir: Path = RUNS_DIR) -> str:
    """같은 모델·프롬프트의 기존 실행이 있으면 그 run_id를 이어 쓴다 (이어 달리기)."""
    suffix = f"-{model}-{prompt_hash()[:8]}"
    if runs_dir.is_dir():
        found = sorted(p.name for p in runs_dir.iterdir() if p.is_dir() and p.name.endswith(suffix))
        if found:
            return found[-1]
    return make_run_id(model)


@dataclass
class CallResult:
    parsed: dict | None
    attempts: list[dict]


def _call(client: LLMClient, cfg: Config, system: str, prompt: str, schema: type,
          ctx: CallContext) -> CallResult:
    attempts = []
    p = prompt
    for i in range(cfg.agent.retry_on_invalid + 1):
        t0 = time.perf_counter()
        try:
            res = client.structured(system, p, schema, cfg.agent.max_tokens, cfg.agent.temperature, ctx)
        except SchemaError as e:
            attempts.append({"prompt": p, "raw": e.raw, "error": str(e)[:2000],
                             "input_tokens": e.usage.input_tokens, "output_tokens": e.usage.output_tokens,
                             "seconds": round(time.perf_counter() - t0, 3)})
            p = prompt + RETRY_SUFFIX.format(error=str(e)[:800])
            continue
        attempts.append({"prompt": p, "raw": res.raw, "error": "",
                         "input_tokens": res.usage.input_tokens, "output_tokens": res.usage.output_tokens,
                         "seconds": round(time.perf_counter() - t0, 3)})
        return CallResult(res.data, attempts)
    return CallResult(None, attempts)


def _fix_ids(report: dict, scenario_id: str, method: str) -> dict:
    """시나리오 ID와 방법 이름은 실행기가 정한다 (모델이 쓴 값을 덮어쓴다)."""
    return {**report, "scenario_id": scenario_id, "method": method}


def run_scenario(client: LLMClient, cfg: Config, scenario_dir: Path, run_id: str,
                 out_dir: Path, language: str | None = None) -> dict:
    t_start = time.perf_counter()
    lang = language or cfg.agent.report_language
    data = load_scenario(scenario_dir)
    sid = data.scenario_id
    method = f"agent:{client.model}"
    pack = build_evidence(data, cfg)
    system = system_prompt(lang)
    a = data.alert
    prompt1 = ROUND1_TEMPLATE.format(scenario_id=sid, alert_level=a["level"],
                                     window_start=a["window_start"], window_end=a["window_end"],
                                     evidence_pack=pack)
    rec: dict = {
        "scenario_id": sid, "run_id": run_id, "model": client.model, "method": method,
        "prompt_hash": prompt_hash(), "language": lang, "evidence_pack": pack,
        "round1": None, "checks": [], "round2": None,
        "preliminary_report": None, "final_report": None,
        "invalid": False, "round2_failed": False, "needs_checks": False,
    }

    r1 = _call(client, cfg, system, prompt1, Round1, CallContext(run_id, sid, "round1"))
    rec["round1"] = {"attempts": r1.attempts, "parsed": r1.parsed}
    if r1.parsed is None:
        rec["invalid"] = True
    else:
        prelim = Report.model_validate(_fix_ids(r1.parsed["report"], sid, method))
        rec["preliminary_report"] = json.loads(prelim.model_dump_json())
        rec["final_report"] = rec["preliminary_report"]
        rec["needs_checks"] = bool(r1.parsed["needs_checks"] and r1.parsed["checks"])
        if rec["needs_checks"]:
            reqs = [CheckRequest.model_validate(c) for c in r1.parsed["checks"]]
            rec["checks"] = run_checks(data, cfg, reqs)
            prompt2 = ROUND2_TEMPLATE.format(
                scenario_id=sid, evidence_pack=pack,
                preliminary_report=prelim.model_dump_json(indent=1),
                check_results=render_checks(rec["checks"]),
            )
            r2 = _call(client, cfg, system, prompt2, Report, CallContext(run_id, sid, "round2"))
            rec["round2"] = {"attempts": r2.attempts, "parsed": r2.parsed}
            if r2.parsed is None:
                rec["round2_failed"] = True
            else:
                final = Report.model_validate(_fix_ids(r2.parsed, sid, method))
                rec["final_report"] = json.loads(final.model_dump_json())

    attempts = r1.attempts + (rec["round2"]["attempts"] if rec["round2"] else [])
    rec["requests"] = len(attempts)
    rec["input_tokens"] = sum(x["input_tokens"] for x in attempts)
    rec["output_tokens"] = sum(x["output_tokens"] for x in attempts)
    if rec["final_report"] is not None:
        sources = [pack] + [c["result"] for c in rec["checks"]]
        nc = check_report(Report.model_validate(rec["final_report"]), sources)
        rec["numcheck"] = nc
        rec["numcheck_bad"] = nc["bad"]
    else:
        rec["numcheck"] = None
        rec["numcheck_bad"] = None
    rec["seconds"] = round(time.perf_counter() - t_start, 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{sid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    return rec


def run_set(client: LLMClient, cfg: Config, scenario_dirs: list[Path], run_id: str,
            runs_dir: Path = RUNS_DIR, limit: int | None = None, log=print,
            language: str | None = None) -> list[dict]:
    """이미 기록이 있는 시나리오는 건너뛰고 차례로 실행한다. 한도 초과는 호출자에게 넘긴다."""
    out_dir = runs_dir / run_id
    done = []
    todo = [d for d in scenario_dirs if not (out_dir / f"{d.name}.json").is_file()]
    if limit is not None:
        todo = todo[:limit]
    for d in todo:
        rec = run_scenario(client, cfg, d, run_id, out_dir, language=language)
        done.append(rec)
        fr = rec["final_report"]
        top = ""
        if fr and fr["hypotheses"]:
            h = fr["hypotheses"][0]
            top = f"{h['entity_type']} {h['chamber_id'] or h['tool_id'] or h['recipe_id']} conf {h['confidence']}"
        log(f"{d.name}  requests {rec['requests']}  {rec['seconds']}s  "
            f"{'INVALID' if rec['invalid'] else (fr['verdict'] if fr else '')}  {top}"
            f"{'  (round2 failed)' if rec['round2_failed'] else ''}")
    return done
