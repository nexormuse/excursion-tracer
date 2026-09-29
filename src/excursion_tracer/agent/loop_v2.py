"""에이전트 v2 실행: 근거 묶음 v2 → 1회차(잠정 보고서 + 확인 1~3개, 필수) → 확인 실행 → 2회차(최종 보고서)
→ ID 검증·정규화.

- 스키마 검증에 실패하면 오류를 붙여 1회 다시 요청하고, 그래도 실패하면 1회차는 invalid,
  2회차는 1회차 잠정 보고서를 최종으로 쓴다.
- 보고서의 ID가 한 가지로만 해석되면 전체 ID로 바꾸고(정규화), 해석이 안 되거나 여러 가지로
  해석되는 ID가 최종 보고서에 있으면 그 목록을 붙여 2회차를 1회 다시 요청한다.
- 기록: runs/<run_id>/<scenario_id>.json. 이미 기록이 있는 시나리오는 건너뛴다.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

from excursion_tracer.agent.checks import render_checks, run_checks
from excursion_tracer.agent.evidence_v2 import build_evidence_v2
from excursion_tracer.agent.loop import RUNS_DIR, CallResult
from excursion_tracer.agent.numcheck import check_report
from excursion_tracer.agent.prompts_v2 import (
    ID_RETRY_SUFFIX_V2, RETRY_SUFFIX_V2, ROUND1_TEMPLATE_V2, ROUND2_TEMPLATE_V2, prompt_hash_v2,
    system_prompt_v2,
)
from excursion_tracer.agent.schema import CheckRequest, Report, Round1V2
from excursion_tracer.agent.validate_ids import normalize_report
from excursion_tracer.config import Config
from excursion_tracer.llm.base import CallContext, LLMClient, SchemaError
from excursion_tracer.stats.data import load_scenario

METHOD_PREFIX = "agent_v2"


def make_run_id_v2(model: str, day: date | None = None) -> str:
    day = day or date.today()
    return f"{day.isoformat()}-{model}-v2-{prompt_hash_v2()[:8]}"


def find_run_id_v2(model: str, runs_dir: Path = RUNS_DIR) -> str:
    """같은 모델·v2 프롬프트의 기존 실행이 있으면 그 run_id를 이어 쓴다."""
    suffix = f"-{model}-v2-{prompt_hash_v2()[:8]}"
    if runs_dir.is_dir():
        found = sorted(p.name for p in runs_dir.iterdir() if p.is_dir() and p.name.endswith(suffix))
        if found:
            return found[-1]
    return make_run_id_v2(model)


def _call(client: LLMClient, cfg: Config, system: str, prompt: str, schema: type, max_tokens: int,
          ctx: CallContext) -> CallResult:
    attempts, p = [], prompt
    for _ in range(cfg.agent.retry_on_invalid + 1):
        t0 = time.perf_counter()
        try:
            res = client.structured(system, p, schema, max_tokens, cfg.agent.temperature, ctx)
        except SchemaError as e:
            attempts.append({"prompt": p, "raw": e.raw, "error": str(e)[:2000],
                             "input_tokens": e.usage.input_tokens, "output_tokens": e.usage.output_tokens,
                             "seconds": round(time.perf_counter() - t0, 3)})
            p = prompt + RETRY_SUFFIX_V2.format(error=str(e)[:800])
            continue
        attempts.append({"prompt": p, "raw": res.raw, "error": "",
                         "input_tokens": res.usage.input_tokens, "output_tokens": res.usage.output_tokens,
                         "seconds": round(time.perf_counter() - t0, 3)})
        return CallResult(res.data, attempts)
    return CallResult(None, attempts)


def _finalize(raw: dict, sid: str, method: str, fab: dict) -> tuple[dict, list[str], list[str]]:
    """시나리오 ID·방법을 채우고 ID를 정규화한 보고서 (Report로 다시 검증)."""
    rep = {**raw, "scenario_id": sid, "method": method}
    rep, normalized, problems = normalize_report(rep, fab)
    rep = json.loads(Report.model_validate(rep).model_dump_json())
    return rep, normalized, problems


def run_scenario_v2(client: LLMClient, cfg: Config, scenario_dir: Path, run_id: str, out_dir: Path,
                    language: str | None = None) -> dict:
    t_start = time.perf_counter()
    lang = language or cfg.agent.report_language
    data = load_scenario(scenario_dir)
    sid = data.scenario_id
    method = f"{METHOD_PREFIX}:{client.model}"
    pack = build_evidence_v2(data, cfg)
    system = system_prompt_v2(lang)
    a = data.alert
    prompt1 = ROUND1_TEMPLATE_V2.format(scenario_id=sid, alert_level=a["level"],
                                        window_start=a["window_start"], window_end=a["window_end"],
                                        evidence_pack=pack)
    rec: dict = {
        "scenario_id": sid, "run_id": run_id, "model": client.model, "method": method, "version": "v2",
        "prompt_hash": prompt_hash_v2(), "language": lang, "evidence_pack": pack,
        "round1": None, "checks": [], "round2": None, "id_retry": None,
        "preliminary_report": None, "final_report": None, "needs_checks": True,
        "invalid": False, "round2_failed": False,
        "id_normalized": [], "id_problems": [], "prelim_id_normalized": [], "prelim_id_problems": [],
    }

    r1 = _call(client, cfg, system, prompt1, Round1V2, cfg.v2.round1_max_tokens,
               CallContext(run_id, sid, "round1"))
    rec["round1"] = {"attempts": r1.attempts, "parsed": r1.parsed}
    if r1.parsed is None:
        rec["invalid"] = True
    else:
        prelim, norm1, prob1 = _finalize(r1.parsed["report"], sid, method, data.fab)
        rec["preliminary_report"] = prelim
        rec["prelim_id_normalized"], rec["prelim_id_problems"] = norm1, prob1
        rec["final_report"] = prelim
        rec["id_normalized"], rec["id_problems"] = norm1, prob1
        reqs = [CheckRequest.model_validate(c) for c in r1.parsed["checks"]]
        rec["checks"] = run_checks(data, cfg, reqs)
        prompt2 = ROUND2_TEMPLATE_V2.format(
            scenario_id=sid, evidence_pack=pack,
            preliminary_report=json.dumps(prelim, ensure_ascii=False, indent=1),
            check_results=render_checks(rec["checks"]),
        )
        r2 = _call(client, cfg, system, prompt2, Report, cfg.v2.round2_max_tokens,
                   CallContext(run_id, sid, "round2"))
        rec["round2"] = {"attempts": r2.attempts, "parsed": r2.parsed}
        if r2.parsed is None:
            rec["round2_failed"] = True
        else:
            final, norm2, prob2 = _finalize(r2.parsed, sid, method, data.fab)
            if prob2:
                retry_prompt = prompt2 + ID_RETRY_SUFFIX_V2.format(problems="\n".join(prob2))
                r3 = _call(client, cfg, system, retry_prompt, Report, cfg.v2.round2_max_tokens,
                           CallContext(run_id, sid, "id_retry"))
                rec["id_retry"] = {"attempts": r3.attempts, "parsed": r3.parsed, "problems_before": prob2}
                if r3.parsed is not None:
                    final, norm2, prob2 = _finalize(r3.parsed, sid, method, data.fab)
            rec["final_report"] = final
            rec["id_normalized"], rec["id_problems"] = norm2, prob2

    attempts = (r1.attempts + (rec["round2"]["attempts"] if rec["round2"] else [])
                + (rec["id_retry"]["attempts"] if rec["id_retry"] else []))
    rec["requests"] = len(attempts)
    rec["input_tokens"] = sum(x["input_tokens"] for x in attempts)
    rec["output_tokens"] = sum(x["output_tokens"] for x in attempts)
    if rec["final_report"] is not None:
        nc = check_report(Report.model_validate(rec["final_report"]),
                          [pack] + [c["result"] for c in rec["checks"]])
        rec["numcheck"], rec["numcheck_bad"] = nc, nc["bad"]
    else:
        rec["numcheck"], rec["numcheck_bad"] = None, None
    rec["seconds"] = round(time.perf_counter() - t_start, 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{sid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rec


def run_set_v2(client: LLMClient, cfg: Config, scenario_dirs: list[Path], run_id: str,
               runs_dir: Path = RUNS_DIR, limit: int | None = None, log=print,
               language: str | None = None) -> list[dict]:
    out_dir = runs_dir / run_id
    todo = [d for d in scenario_dirs if not (out_dir / f"{d.name}.json").is_file()]
    if limit is not None:
        todo = todo[:limit]
    done = []
    for d in todo:
        rec = run_scenario_v2(client, cfg, d, run_id, out_dir, language=language)
        done.append(rec)
        fr = rec["final_report"]
        top = ""
        if fr and fr["hypotheses"]:
            h = fr["hypotheses"][0]
            top = f"{h['entity_type']} {h['chamber_id'] or h['tool_id'] or h['recipe_id']} conf {h['confidence']}"
        n_norm = len(rec["id_normalized"])
        log(f"{d.name}  requests {rec['requests']}  {rec['seconds']}s  "
            f"{'INVALID' if rec['invalid'] else (fr['verdict'] if fr else '')}  {top}"
            + ("  (round2 failed)" if rec["round2_failed"] else "")
            + ("  (id retry)" if rec["id_retry"] else "")
            + (f"  ids normalized {n_norm}" if n_norm else ""))
    return done
