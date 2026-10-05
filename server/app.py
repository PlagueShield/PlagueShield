"""PlagueShield dashboard API and static host.

The agent pipeline POSTs completed assessments to /api/assessments; the
dashboard reads them back. Assessments are append-only and persisted to disk
as newline-delimited JSON, because an audit trail that can be silently
rewritten is not an audit trail. Re-assessing a case adds a new version rather
than replacing the old one, so a reviewer can see how the picture changed as
results arrived.

Run:
    uvicorn server.app:app --reload --port 8000
"""

from __future__ import annotations

import json
import asyncio
import os
import pathlib
import threading
import inspect
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from plagueshield.data import available_cases, load_case
from plagueshield.data.loader import load_public_snapshot
from plagueshield.llm import DEFAULT_ANALYSIS_MODEL, is_configured
from plagueshield.knowledge.case_sources import case_reference_info
from plagueshield.models import CaseAssessment
from plagueshield.orchestrator import Pipeline
from server.worker import AGENT_NAMES, ResearchWorker
from server.x_publisher import XPublisher
from server.results import ResultsStore

BASE_DIR = pathlib.Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = pathlib.Path(os.getenv("PLAGUESHIELD_DATA_DIR", str(BASE_DIR)))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORE_PATH = DATA_DIR / "assessments.ndjson"
TOKEN_ADDRESS = os.getenv("PLAGUESHIELD_TOKEN_ADDRESS", "").strip()
TOKEN_SYMBOL = os.getenv("PLAGUESHIELD_TOKEN_SYMBOL", "PLAGUESHIELD")
TOKEN_MARKET_CAP = os.getenv("PLAGUESHIELD_TOKEN_MARKET_CAP")
WORKER_ENABLED = os.getenv("PLAGUESHIELD_WORKER_ENABLED", "true").lower() not in {"false", "0", "no"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(_backfill_results)
    if WORKER_ENABLED:
        _worker.start()
    _x_publisher.start()
    try:
        yield
    finally:
        await asyncio.to_thread(_x_publisher.stop)
        if WORKER_ENABLED:
            await asyncio.to_thread(_worker.stop)

app = FastAPI(
    title="PlagueShield",
    description="Multi-agent decision support for suspected Yersinia pestis cases",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

_lock = threading.Lock()
_records_cache: list[dict[str, Any]] = []
_records_signature: tuple | None = None
from plagueshield.prompt_evolution import PromptStore, review_history

_prompts = PromptStore(STORE_PATH.parent / "prompt_revisions.sqlite3")
_pipeline = Pipeline(
    live_evidence=os.getenv("PLAGUESHIELD_LIVE_EVIDENCE", "true").lower() not in {"false", "0", "no"},
    prompt_provider=_prompts.current, history_provider=lambda: review_history(_read_all()),
)


# ---------------------------------------------------------------------------
# Append-only store
# ---------------------------------------------------------------------------


def _append(record: dict[str, Any]) -> None:
    global _records_signature
    with _lock:
        with STORE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        _records_signature = None


def _publish_background(assessment: CaseAssessment) -> None:
    record = assessment.model_dump(mode="json")
    record["_received_at"] = datetime.now(timezone.utc).isoformat()
    record["_publisher_source"] = "research_worker"
    _append(record)
    try:
        _prompts.commit(record)
    except Exception:
        logging.getLogger(__name__).exception('Prompt revision rejected; previous prompt retained')
    try:
        _results.publish(record)
    except Exception:
        logging.getLogger(__name__).exception('Research article publication failed; stored assessment remains available for backfill')


_worker = ResearchWorker(
    _pipeline, _publish_background,
    interval=float(os.getenv("PLAGUESHIELD_RUN_INTERVAL_SECONDS", "120")),
)


def _read_all() -> list[dict[str, Any]]:
    global _records_cache, _records_signature
    with _lock:
        try:
            stat = STORE_PATH.stat()
        except FileNotFoundError:
            _records_cache, _records_signature = [], None
            return []
        signature = (str(STORE_PATH), stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if signature == _records_signature:
            return list(_records_cache)
        out: list[dict[str, Any]] = []
        with STORE_PATH.open(encoding="utf-8") as records:
            for line in records:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        _records_cache, _records_signature = out, signature
        return list(_records_cache)


def _latest_per_case(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for rec in records:
        cid = rec.get("case_id")
        if not cid:
            continue
        prev = latest.get(cid)
        if prev is None or rec.get("assessed_at", "") >= prev.get("assessed_at", ""):
            latest[cid] = rec
    return sorted(
        latest.values(), key=lambda r: r.get("assessed_at", ""), reverse=True
    )


_x_publisher = XPublisher(_read_all, DATA_DIR / 'x_outbox.sqlite3')
_results = ResultsStore(DATA_DIR / 'results.sqlite3')


def _backfill_results() -> None:
    for record in _read_all():
        try:
            _results.publish(record)
        except (FileNotFoundError, KeyError, ValueError):
            logging.getLogger(__name__).warning('Skipping an incompatible historical research article')


def _summarise(rec: dict[str, Any]) -> dict[str, Any]:
    """Compact row for the dashboard feed."""
    return {
        "case_id": rec.get("case_id"),
        "case_label": rec.get("case_label"),
        "assessed_at": rec.get("assessed_at"),
        "plague_likelihood": rec.get("plague_likelihood"),
        "plague_probability": rec.get("plague_probability"),
        "case_classification": rec.get("case_classification"),
        "standard_treatment_susceptibility": rec.get(
            "standard_treatment_susceptibility"
        ),
        "resistance_anomaly": rec.get("resistance_anomaly"),
        "confidence": rec.get("confidence"),
        "most_valuable_next_result": rec.get("most_valuable_next_result"),
        "human_review_required": rec.get("human_review_required"),
        "n_flags": len(rec.get("flags", [])),
        "n_critical": sum(
            1 for f in rec.get("flags", []) if f.get("severity") == "critical"
        ),
        "agent_activity": _agent_activity_for_record(rec),
    }


def _agent_activity_for_record(rec: dict[str, Any]) -> list[dict[str, Any]]:
    verdicts = rec.get("verdicts") or {}
    out: list[dict[str, Any]] = []
    for name, verdict in verdicts.items():
        data = verdict.get("data") or {}
        out.append(
            {
                "agent": name,
                "case_id": rec.get("case_id"),
                "headline": verdict.get("headline"),
                "confidence": verdict.get("confidence"),
                "abstained": verdict.get("abstained", False),
                "produced_at": verdict.get("produced_at") or rec.get("assessed_at"),
                "working_on": _working_on(name, verdict, rec),
                "model": data.get("model"),
                "configured": data.get("configured"),
            }
        )
    return out


def _working_on(name: str, verdict: dict[str, Any], rec: dict[str, Any]) -> str:
    data = verdict.get("data") or {}
    rationale = verdict.get("rationale") or []
    if name != "analysis":
        return verdict.get("headline") or (rationale[0] if rationale else "No published result.")
    if name == "analysis":
        return data.get("synthesis") or verdict.get("abstain_reason") or "Awaiting GPT-5.5 analysis."


def _agent_catalog() -> list[dict[str, Any]]:
    from plagueshield.agents.meta_review import MetaReviewAgent
    from plagueshield.agents import (
        ClinicalSummaryAgent,
        DiagnosticAgent,
        DiscordanceAgent,
        EvidenceAgent,
        LLMAnalysisAgent,
        CodeAnalysisAgent,
        NextTestAgent,
        ResistanceAgent,
        UncertaintyAgent,
    )

    stages = [
        (1, [DiagnosticAgent, ResistanceAgent]),
        (2, [EvidenceAgent, DiscordanceAgent]),
        (3, [UncertaintyAgent]),
        (4, [NextTestAgent]),
        (5, [CodeAnalysisAgent]),
        (6, [LLMAnalysisAgent]),
        (7, [ClinicalSummaryAgent]),
        (8, [MetaReviewAgent]),
    ]
    rows: list[dict[str, Any]] = []
    for stage, cls_list in stages:
        for c in cls_list:
            rows.append(
                {
                    "stage": stage,
                    "name": c.name,
                    "description": c.description,
                    "depends_on": list(c.depends_on),
                    "parallel": len(cls_list) > 1,
                    "execution_kind": "LLM / OpenAI Responses" if c.name in {"analysis", "meta_review"} else "Python / deterministic",
                }
            )
    rows.append({
        "stage": 9, "name": "x_publisher", "parallel": False, "scheduled": True,
        "description": "Publishes compact threads of research agents' findings on X every five minutes",
        "depends_on": list(AGENT_NAMES), "execution_kind": "Python / scheduled X API publisher",
    })
    return rows


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def require_writable_api() -> None:
    if os.getenv("PLAGUESHIELD_PUBLIC_READ_ONLY", "false").lower() in {"true", "1", "yes"}:
        raise HTTPException(403, "This public deployment is read-only. Research runs are managed by the backend worker.")


@app.post("/api/assessments", dependencies=[Depends(require_writable_api)])
def post_assessment(assessment: CaseAssessment) -> JSONResponse:
    """Endpoint the agent pipeline posts its results to."""
    record = assessment.model_dump(mode="json")
    record["_received_at"] = datetime.now(timezone.utc).isoformat()
    _append(record)
    return JSONResponse(
        status_code=201,
        content={
            "status": "accepted",
            "case_id": assessment.case_id,
            "assessed_at": record["assessed_at"],
        },
    )


@app.get("/api/assessments")
def list_assessments(
    all_versions: bool = Query(False, description="Include superseded versions"),
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    records = _read_all()
    rows = records if all_versions else _latest_per_case(records)
    rows = sorted(rows, key=lambda r: r.get("assessed_at", ""), reverse=True)[:limit]
    return {
        "count": len(rows),
        "total_versions": len(records),
        "assessments": [_summarise(r) for r in rows],
    }


@app.get("/api/assessments/{case_id}")
def get_assessment(case_id: str, version: int | None = None) -> dict[str, Any]:
    versions = [r for r in _read_all() if r.get("case_id") == case_id]
    if not versions:
        raise HTTPException(404, f"No assessment posted for case {case_id!r}")
    versions.sort(key=lambda r: r.get("assessed_at", ""))
    if version is not None:
        if not (0 <= version < len(versions)):
            raise HTTPException(
                404, f"Version {version} out of range (0..{len(versions) - 1})"
            )
        chosen = versions[version]
    else:
        chosen = versions[-1]
    return {
        "assessment": chosen,
        "version": versions.index(chosen),
        "n_versions": len(versions),
    }


@app.get("/api/assessments/{case_id}/report", response_class=PlainTextResponse)
def get_report(case_id: str, fmt: str = Query("markdown", pattern="^(markdown|ascii)$")):
    payload = get_assessment(case_id)["assessment"]
    key = "report_markdown" if fmt == "markdown" else "report_ascii"
    return PlainTextResponse(payload.get(key) or "(no report)")


@app.get("/api/cases")
def list_cases() -> dict[str, Any]:
    """The bundled de-identified case records available to assess."""
    assessed = {r["case_id"] for r in _latest_per_case(_read_all())}
    rows = available_cases()
    for row in rows:
        row["assessed"] = row["case_id"] in assessed
        row.update(case_reference_info(load_case(row["case_id"])))
    return {"count": len(rows), "cases": rows}


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> dict[str, Any]:
    try:
        case = load_case(case_id)
        return {**case.model_dump(mode="json"), **case_reference_info(case)}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/run/{case_id}", dependencies=[Depends(require_writable_api)])
def run_case(case_id: str) -> dict[str, Any]:
    """Runs the agent pipeline over a bundled case and posts the result."""
    try:
        case = load_case(case_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    assessment = _pipeline.run(case)
    record = assessment.model_dump(mode="json")
    record["_received_at"] = datetime.now(timezone.utc).isoformat()
    _append(record)
    return {"status": "assessed", "assessment": record}


@app.post("/api/run-all", dependencies=[Depends(require_writable_api)])
def run_all() -> dict[str, Any]:
    results = []
    for row in available_cases():
        try:
            results.append(run_case(row["case_id"])["assessment"])
        except HTTPException:
            continue
    return {"count": len(results), "assessments": [_summarise(r) for r in results]}


@app.get("/api/public-data")
def public_data() -> dict[str, Any]:
    """Snapshots of the public sources backing the system."""
    out: dict[str, Any] = {}
    for name in ("ncbi_biosamples", "ncbi_assembly_counts", "who_outbreak_news"):
        snap = load_public_snapshot(name)
        if snap is None:
            out[name] = {"ok": False, "note": "No snapshot — run scripts/fetch_public_data.py"}
        else:
            # Trim payloads for the dashboard.
            out[name] = {
                "source": snap.get("source"),
                "ok": snap.get("ok"),
                "fetched_at": snap.get("fetched_at"),
                "count": snap.get("count"),
                "note": snap.get("note"),
                "url": snap.get("url"),
                "records": snap.get("records", [])[:25],
            }
    return out


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    records = _read_all()
    latest = _latest_per_case(records)
    return {
        "cases_available": len(available_cases()),
        "cases_assessed": len(latest),
        "total_assessments": len(records),
        "review_required": sum(1 for r in latest if r.get("human_review_required")),
        "anomalies": sum(
            1
            for r in latest
            if r.get("resistance_anomaly") in ("Suspected", "Confirmed")
        ),
        "abstentions": sum(
            1
            for r in latest
            if r.get("resistance_anomaly") == "Insufficient evidence"
        ),
        "critical_flags": sum(
            sum(1 for f in r.get("flags", []) if f.get("severity") == "critical")
            for r in latest
        ),
    }


_token_status_lock = threading.Lock()
_token_status_cache: dict[str, Any] = {}
_token_status_expires = 0.0


@app.get("/api/token-status")
def token_status(response: Response) -> dict[str, Any]:
    """Short shared cache for frequent public market-cap updates."""
    global _token_status_cache, _token_status_expires
    response.headers['Cache-Control'] = 'no-store'
    with _token_status_lock:
        if time.monotonic() >= _token_status_expires:
            fresh = _fetch_token_status()
            if fresh.get('address') and fresh.get('address') == _token_status_cache.get('address') and fresh.get('market_cap_source') != 'dexscreener' and _token_status_cache.get('market_cap_source') == 'dexscreener':
                fresh = {**_token_status_cache, 'stale': True,
                         'market_cap_error': fresh.get('market_cap_error', 'Market data temporarily unavailable.')}
            else:
                fresh.update(stale=False, refreshed_at=datetime.now(timezone.utc).isoformat())
            _token_status_cache = fresh
            _token_status_expires = time.monotonic() + 4.0
        return dict(_token_status_cache)


def _fetch_token_status() -> dict[str, Any]:
    """Public token status. DexScreener is best-effort; env vars are fallback."""
    status: dict[str, Any] = {
        "symbol": TOKEN_SYMBOL,
        "address": TOKEN_ADDRESS or None,
        "configured": bool(TOKEN_ADDRESS),
        "market_cap": TOKEN_MARKET_CAP if TOKEN_ADDRESS else None,
        "market_cap_source": "env" if TOKEN_ADDRESS and TOKEN_MARKET_CAP else None,
        "fees_note": "Token fees are used to pay compute costs for this research.",
        "x_account": "@plagueshieldX",
        "x_url": "https://x.com/plagueshieldX",
    }
    if not TOKEN_ADDRESS:
        return status
    try:
        import httpx

        url = f"https://api.dexscreener.com/latest/dex/tokens/{TOKEN_ADDRESS}"
        with httpx.Client(timeout=4.0) as client:
            resp = client.get(url)
        if resp.status_code < 400:
            pairs = resp.json().get("pairs") or []
            if pairs:
                pair = max(pairs, key=lambda p: float(p.get("fdv") or p.get("marketCap") or 0))
                status["market_cap"] = pair.get("marketCap") or pair.get("fdv") or TOKEN_MARKET_CAP
                status["price_usd"] = pair.get("priceUsd")
                status["pair_url"] = pair.get("url")
                status["market_cap_source"] = "dexscreener"
    except Exception as exc:  # noqa: BLE001 - public status should degrade
        status["market_cap_error"] = f"{type(exc).__name__}: {exc}"
    return status


@app.get("/api/agents")
def agents() -> dict[str, Any]:
    """Describes the pipeline, for the dashboard's architecture panel."""
    catalog = _agent_catalog()
    return {
        "llm": {
            "provider": "openai",
            "model": DEFAULT_ANALYSIS_MODEL,
            "configured": is_configured(),
            "purpose": "GPT-5.5 analysis of upstream agent verdicts",
        },
        "agents": catalog,
        "stages": [
            {
                "stage": stage,
                "parallel": any(a["parallel"] for a in catalog if a["stage"] == stage),
                "agents": [a for a in catalog if a["stage"] == stage],
            }
            for stage in sorted({a["stage"] for a in catalog})
        ]
    }


@app.get("/api/agent-status")
def agent_status() -> dict[str, Any]:
    latest = _latest_per_case(_read_all())
    activity = [
        item
        for rec in latest[:20]
        for item in _agent_activity_for_record(rec)
    ]
    activity.sort(key=lambda x: x.get("produced_at") or "", reverse=True)
    return {
        "agents": _agent_catalog(),
        "activity": activity[:120],
        "latest_cases": [_summarise(r) for r in latest[:12]],
        "live": _worker.snapshot(),
    }


@app.get("/api/agents/{name}/runs")
def agent_runs(name: str) -> dict[str, Any]:
    """Published verdicts, reproducible inputs, prompts and linked sources."""
    if name not in {agent["name"] for agent in _agent_catalog()}:
        raise HTTPException(404, "Agent not found")
    if name == 'x_publisher':
        status = _x_publisher.snapshot()
        preview = _x_publisher.preview() if not status['history'] else None
        previews = status['history'] or ([preview] if preview else [])
        runs = [{
            "case_id": item['case_id'], "case_label": 'X research thread',
            "assessed_at": item.get('created_at'), "source_note": 'Summaries of completed experimental research; see the research agents for full evidence.',
            "references": [*case_reference_info(load_case(item['case_id']))['references'],
                           *[{"source": "X", "title": "Published post", "url": url} for url in item.get('post_urls', [])]],
            "verdict": {"headline": f"X publishing / {item.get('status', 'draft preview')}",
                        "publication_status": item.get('status', 'draft preview'),
                        "confidence": "Not applicable", "rationale": item['posts'],
                        "data": {"publishing_policy": status['policy'], "mode": item.get('mode', 'preview'), "posts": item['posts']},
                        "flags": [], "abstained": False},
        } for item in previews]
        return {"agent": name, "implementation": inspect.getsource(inspect.getmodule(XPublisher)), "runs": runs, "publisher": status}
    runs = []
    for record in _latest_per_case(_read_all())[:20]:
        verdict = (record.get("verdicts") or {}).get(name)
        if not verdict:
            continue
        case_id = record["case_id"]
        try:
            reference_info = case_reference_info(load_case(case_id))
        except FileNotFoundError:
            reference_info = {"references": [], "source_note": "No bundled source record."}
        runs.append({
            "case_id": case_id, "case_label": record.get("case_label"),
            "assessed_at": record.get("assessed_at"), "verdict": verdict,
            "citations": verdict.get("citations", []), **reference_info,
        })
    implementation = inspect.getsource(inspect.getmodule(type(getattr(_pipeline, name))))
    return {"agent": name, "implementation": implementation, "runs": runs,
            "prompt_history": _prompts.history() if name in {"analysis", "meta_review"} else [],
            "current_prompt": _prompts.current() if name in {"analysis", "meta_review"} else None}


@app.get('/api/x-status')
def x_status() -> dict[str, Any]:
    """Read-only publishing configuration and history; never returns credentials."""
    return _x_publisher.snapshot()


@app.get('/api/results')
def research_results(
    limit: int = Query(24, ge=1, le=100), offset: int = Query(0, ge=0),
    q: str = Query('', max_length=200), origin: str = Query('', max_length=40),
) -> dict[str, Any]:
    """Immutable research articles, one for each eligible completed iteration."""
    return _results.list(limit=limit, offset=offset, query=q, origin=origin)


@app.get('/api/results/{article_id}')
def research_article(article_id: str) -> dict[str, Any]:
    article = _results.get(article_id)
    if article is None:
        raise HTTPException(404, 'Research article not found')
    return {key: value for key, value in article.items() if key != 'artifact'}


@app.get('/api/results/{article_id}/artifact')
def research_artifact(article_id: str) -> JSONResponse:
    article = _results.get(article_id)
    if article is None:
        raise HTTPException(404, 'Research article not found')
    return JSONResponse(article['artifact'], headers={
        'Content-Disposition': f'attachment; filename="plagueshield-{article["id"]}.json"',
    })


@app.get("/api/live")
def live_progress() -> dict[str, Any]:
    """Current backend run, per-agent progress, and recent execution events."""
    return _worker.snapshot()


@app.get("/api/live/stream")
async def live_stream(request: Request) -> StreamingResponse:
    """Streams real agent progress as server-sent events."""
    async def events():
        revision = -1
        heartbeat = asyncio.get_running_loop().time()
        while not await request.is_disconnected():
            if _worker.revision() != revision:
                snapshot = _worker.snapshot()
                revision = snapshot["revision"]
                yield f"event: progress\ndata: {json.dumps(snapshot)}\n\n"
                heartbeat = asyncio.get_running_loop().time()
            elif asyncio.get_running_loop().time() - heartbeat >= 15:
                yield ": keep-alive\n\n"
                heartbeat = asyncio.get_running_loop().time()
            await asyncio.sleep(0.25)

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Static dashboard
# ---------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/{path:path}")
def spa(path: str) -> FileResponse:
    if path.startswith("api/") or path in {"healthz", "openapi.json"}:
        raise HTTPException(404, "Not found")
    return FileResponse(STATIC_DIR / "index.html")
