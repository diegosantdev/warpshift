#!/usr/bin/env python3
"""Docker container entrypoint for WarpShift runner.

Reads REPO_URL and MODE from env, runs the pipeline stages,
and writes structured JSON logs to /workspace/logs/.
"""

import json
import os
import sys
import time

# Ensure the backend package is importable
sys.path.insert(0, "/opt/warpshift/backend")

from app.stages import (
    run_hipify_stage,
    run_static_analysis_stage,
    run_runtime_validation_stage,
    run_ai_explanation_stage,
)


LOGS_DIR = "/workspace/logs"


def write_stage_log(stage_num: int, execution):
    """Persist a StageLog as JSON to /workspace/logs/."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"stage_{stage_num:02d}.json")
    log_data = execution.log.to_dict() if execution.log else {
        "stage": execution.name,
        "exit_code": 0 if execution.status == "done" else 1,
        "duration_ms": 0,
        "stdout": execution.detail,
        "stderr": "",
        "toolchain": {},
    }
    log_data["status"] = execution.status
    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(log_data, fp, indent=2)
    print(f"[WarpShift] Stage {stage_num} ({execution.name}): {execution.status}")


def main():
    repo_url = os.getenv("REPO_URL", "https://github.com/NVIDIA/cuda-samples")
    mode = os.getenv("MODE", "live")

    print(f"[WarpShift] Starting pipeline — repo={repo_url}, mode={mode}")
    t0 = time.perf_counter()

    # Stage 1: HIPIFY
    s1 = run_hipify_stage(repo_url)
    write_stage_log(1, s1)

    # Stage 2: Static Analysis
    s2 = run_static_analysis_stage()
    write_stage_log(2, s2)

    # Stage 3: Runtime Validation
    s3 = run_runtime_validation_stage(mode)
    write_stage_log(3, s3)

    # Stage 4: AI Explanation
    s4 = run_ai_explanation_stage()
    write_stage_log(4, s4)

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    summary = {
        "total_duration_ms": elapsed,
        "stages_completed": 4,
        "repo_url": repo_url,
        "mode": mode,
    }
    summary_path = os.path.join(LOGS_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[WarpShift] Pipeline complete in {elapsed:.0f}ms")


if __name__ == "__main__":
    main()
