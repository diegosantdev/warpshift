from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .pipeline import (
    export_risk_report,
    get_demo_repo_candidates,
    get_history,
    RUNS_DIR,
    run_analysis,
    stage_events,
    create_pr_for_run,
)
from .real_anchor import load_real_anchor, prepare_real_anchor
from .settings import settings
from .schemas import AnalysisRequest

app = FastAPI(title="WarpShift API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalysisRequest):
    return run_analysis(req)


@app.get("/history")
def history():
    return {"items": get_history()}


@app.get("/runs/{run_id}")
def run_evidence(run_id: str):
    path = f"{RUNS_DIR}/{run_id}.json"
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return {"run_id": run_id, "evidence": json.load(fp)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run evidence not found") from exc

@app.post("/runs/{run_id}/create-pr")
def create_pr_endpoint(run_id: str):
    url = create_pr_for_run(run_id)
    if not url:
        raise HTTPException(status_code=500, detail="Failed to create PR. Ensure GitHub CLI is authenticated and repo exists.")
    return {"status": "ok", "pr_url": url}


@app.get("/demo-repos")
def demo_repos():
    return {"items": get_demo_repo_candidates()}


@app.post("/export/risk-report")
def export_report(req: AnalysisRequest, format: str = "json"):
    result = run_analysis(req)
    if format == "json":
        return {"format": "json", "content": export_risk_report(result)}
    if format == "markdown":
        return {"format": "markdown", "content": export_risk_report(result, as_markdown=True)}
    raise HTTPException(status_code=400, detail="format must be json or markdown")


@app.post("/anchor/prepare")
def prepare_anchor():
    try:
        artifact = prepare_real_anchor()
        return {"status": "ok", "artifact": artifact}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to prepare anchor: {exc}") from exc


@app.get("/anchor/status")
def anchor_status():
    artifact = load_real_anchor()
    return {
        "available": artifact is not None,
        "mode": f"{settings.backend_mode.upper()} (anchor cached)",
        "artifact": artifact,
    }


@app.post("/analyze/stream")
def analyze_stream(req: AnalysisRequest):
    def event_stream():
        for event_name, payload in stage_events(req):
            yield f"event: {event_name}\n"
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/analyze/stream")
def analyze_stream_get(github_url: str, mode: str = "live"):
    req = AnalysisRequest(github_url=github_url, mode=mode)

    def event_stream():
        for event_name, payload in stage_events(req):
            yield f"event: {event_name}\n"
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
