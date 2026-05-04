from __future__ import annotations

import json
import os
import subprocess
import sys
import importlib

from fastapi import FastAPI, HTTPException, UploadFile, File
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

app = FastAPI(title="WarpShift API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health & Status ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend_mode": settings.backend_mode,
        "hipcc": settings.hipcc_bin,
        "hipify": settings.hipify_bin,
    }


@app.get("/admin/rocm-info")
def rocm_info():
    """Query real ROCm toolchain availability on this machine."""
    info: dict = {"backend_mode": settings.backend_mode}

    def _run(cmd: list[str]) -> tuple[int, str]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            return r.returncode, (r.stdout + r.stderr).strip()
        except Exception as exc:
            return -1, str(exc)

    rc, out = _run(["hipcc", "--version"])
    info["hipcc_available"] = rc == 0
    info["hipcc_version"] = out.splitlines()[0] if out else "unavailable"

    rc2, out2 = _run(["hipify-perl", "--version"])
    # hipify-perl exits non-zero even on success sometimes
    info["hipify_available"] = "hipify" in out2.lower() or rc2 == 0
    info["hipify_version"] = out2.splitlines()[0] if out2 else "unavailable"

    rc3, out3 = _run(["rocm-smi", "--showproductname"])
    info["rocm_smi"] = out3[:300] if out3 else "unavailable"

    rc4, out4 = _run(["rocminfo"])
    # Look for MI300X / gfx942
    mi300x = any("MI300" in line or "gfx942" in line for line in out4.splitlines())
    info["mi300x_detected"] = mi300x
    info["rocminfo_snippet"] = "\n".join(
        l for l in out4.splitlines() if "MI300" in l or "gfx942" in l or "Name:" in l
    )[:400]

    return info


@app.post("/admin/patch-file")
async def patch_file(file: UploadFile = File(...), path: str = ""):
    """
    Hot-patch a Python source file on this server without SSH.
    Accepts a multipart file upload and writes it to the given relative path
    under the app directory. Triggers module reload.

    Usage (from dev machine):
        Invoke-WebRequest -Uri http://<host>:8000/admin/patch-file `
            -Method POST `
            -Form @{file=Get-Item pipeline.py; path="app/pipeline.py"}
    """
    # Security: only allow .py files in the app directory
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not path:
        raise HTTPException(400, "path query parameter required")
    if not path.endswith(".py"):
        raise HTTPException(400, "Only .py files allowed")
    abs_path = os.path.abspath(os.path.join(base, path))
    if not abs_path.startswith(base):
        raise HTTPException(400, "Path traversal rejected")

    content = await file.read()
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "wb") as fp:
        fp.write(content)

    # Invalidate module cache so uvicorn --reload picks it up
    mod_key = None
    rel = os.path.relpath(abs_path, base).replace(os.sep, "/").replace(".py", "").replace("/", ".")
    for k in list(sys.modules.keys()):
        if rel in k:
            mod_key = k
            try:
                importlib.reload(sys.modules[k])
            except Exception:
                pass

    return {
        "status": "patched",
        "path": abs_path,
        "bytes": len(content),
        "module_reloaded": mod_key,
    }


@app.post("/admin/git-pull")
def git_pull():
    """Pull latest code from git and return diff summary."""
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        r = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=app_dir, capture_output=True, text=True, timeout=60,
        )
        return {
            "exit_code": r.returncode,
            "stdout": r.stdout.strip(),
            "stderr": r.stderr.strip(),
            "success": r.returncode == 0,
        }
    except Exception as exc:
        raise HTTPException(500, f"git pull failed: {exc}")


# ── Core API ──────────────────────────────────────────────────────────────────

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
        raise HTTPException(
            status_code=500,
            detail="Failed to create PR. Ensure GitHub CLI is authenticated and repo exists.",
        )
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


# ── Anchor ────────────────────────────────────────────────────────────────────

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


# ── Streaming (SSE) ───────────────────────────────────────────────────────────

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
