"""Docker sandbox executor for WarpShift.

Runs the migration pipeline inside an isolated container so that
every execution is reproducible regardless of host toolchain.

Two modes:
  - Dry Run (default): compile-only validation, no GPU required.
  - GPU Real: full benchmark on AMD hardware (requires --device flag).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from app.settings import settings


@dataclass
class SandboxResult:
    """Aggregated output from a single container run."""
    exit_code: int
    duration_ms: float
    stage_logs: list[dict] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""


def _docker_available() -> bool:
    """Check if Docker daemon is reachable."""
    try:
        proc = subprocess.run(
            ["docker", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _image_exists(image: str) -> bool:
    """Check if the runner image exists locally."""
    try:
        proc = subprocess.run(
            ["docker", "images", "-q", image],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(proc.stdout.strip())
    except Exception:
        return False


def run_in_sandbox(
    repo_url: str,
    mode: str = "live",
    *,
    gpu: bool = False,
) -> SandboxResult:
    """Spin up a container, run the pipeline, collect structured logs.

    Parameters
    ----------
    repo_url : str
        GitHub URL to analyze.
    mode : str
        "live" or "full".
    gpu : bool
        If True, pass ``--device /dev/kfd --device /dev/dri`` for AMD GPU access.

    Returns
    -------
    SandboxResult
        Aggregated container output with per-stage structured logs.
    """
    if not _docker_available():
        return SandboxResult(
            exit_code=1,
            duration_ms=0,
            stderr="Docker daemon not reachable. Falling back to host execution.",
        )

    image = settings.docker_image
    if not _image_exists(image):
        return SandboxResult(
            exit_code=1,
            duration_ms=0,
            stderr=f"Docker image '{image}' not found. Build with: docker build -t {image} .",
        )

    # Create a temp directory for this run's workspace
    run_dir = tempfile.mkdtemp(prefix="warpshift_run_")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        f"--memory={settings.docker_memory_limit}",
        f"--cpus={settings.docker_cpu_limit}",
        "-v", f"{run_dir}:/workspace",
        "-e", f"REPO_URL={repo_url}",
        "-e", f"MODE={mode}",
        "-e", "MIGRATEAI_BACKEND_MODE=real",
    ]

    if gpu:
        cmd.extend([
            "--device", "/dev/kfd",
            "--device", "/dev/dri",
            "--group-add", "video",
        ])

    cmd.append(image)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=settings.docker_timeout_seconds,
        )
        elapsed = round((time.perf_counter() - t0) * 1000, 2)

        # Collect structured logs from the container workspace
        stage_logs: list[dict] = []
        for log_file in sorted(os.listdir(logs_dir)) if os.path.exists(logs_dir) else []:
            if log_file.endswith(".json"):
                try:
                    with open(os.path.join(logs_dir, log_file), "r", encoding="utf-8") as fp:
                        stage_logs.append(json.load(fp))
                except Exception:
                    continue

        return SandboxResult(
            exit_code=proc.returncode,
            duration_ms=elapsed,
            stage_logs=stage_logs,
            stdout=proc.stdout[:2000],
            stderr=proc.stderr[:2000],
        )
    except subprocess.TimeoutExpired:
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SandboxResult(
            exit_code=124,
            duration_ms=elapsed,
            stderr=f"Container timed out after {settings.docker_timeout_seconds}s.",
        )
    except Exception as exc:
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SandboxResult(
            exit_code=1,
            duration_ms=elapsed,
            stderr=f"Docker execution error: {exc}",
        )
