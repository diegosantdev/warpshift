from __future__ import annotations

import json
import os
import re
import subprocess
from difflib import unified_diff

from .settings import settings


def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)


def _extract_warp_line(source_lines: list[str]) -> tuple[int | None, str | None]:
    for idx, line in enumerate(source_lines, start=1):
        if re.search(r"\bwarpSize\b\s*=\s*32\b", line):
            return idx, line.rstrip("\n")
    for idx, line in enumerate(source_lines, start=1):
        if " / 32" in line or "& 31" in line:
            return idx, line.rstrip("\n")
    return None, None


def _hipify_convert(source_path: str) -> tuple[str | None, str | None]:
    # Try hipify-clang first, then hipify-perl as fallback.
    try:
        result = _run([settings.hipify_bin, source_path])
        return result.stdout, settings.hipify_bin
    except Exception:
        try:
            perl_candidate = "hipify-perl"
            result = _run([perl_candidate, source_path])
            return result.stdout, perl_candidate
        except Exception:
            local_hipify = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "tools", "HIPIFY", "bin", "hipify-perl")
            )
            if os.path.exists(local_hipify):
                try:
                    result = _run(["perl", local_hipify, source_path])
                    return result.stdout, "perl hipify-perl(local)"
                except Exception:
                    return None, None
            return None, None


def prepare_real_anchor() -> dict:
    os.makedirs(settings.anchor_cache_dir, exist_ok=True)

    if not os.path.exists(os.path.join(settings.anchor_cache_dir, ".git")):
        _run(["git", "clone", "--depth", "1", settings.anchor_repo_url, settings.anchor_cache_dir])

    _run(["git", "fetch", "--depth", "1", "origin", settings.anchor_repo_ref], cwd=settings.anchor_cache_dir)
    _run(["git", "checkout", settings.anchor_repo_ref], cwd=settings.anchor_cache_dir)
    pinned_commit = _run(["git", "rev-parse", "HEAD"], cwd=settings.anchor_cache_dir).stdout.strip()

    source_path = os.path.join(settings.anchor_cache_dir, settings.anchor_relative_path)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Anchor source file not found: {source_path}")

    with open(source_path, "r", encoding="utf-8", errors="ignore") as fp:
        original = fp.read()
    original_lines = original.splitlines(keepends=True)

    warp_line, warp_text = _extract_warp_line(original_lines)
    converted, hipify_tool = _hipify_convert(source_path)
    converted_lines = converted.splitlines(keepends=True) if converted else []

    diff_lines = (
        list(
            unified_diff(
                original_lines,
                converted_lines,
                fromfile="reduction_kernel.cu",
                tofile="reduction_kernel.hip.cpp",
                n=2,
            )
        )
        if converted
        else []
    )

    artifact = {
        "repo_url": settings.anchor_repo_url,
        "repo_ref": settings.anchor_repo_ref,
        "repo_commit": pinned_commit,
        "source_relative_path": settings.anchor_relative_path,
        "hipify_executed": bool(converted),
        "hipify_tool": hipify_tool,
        "warp_detection": {
            "found": warp_line is not None,
            "line": warp_line,
            "content": warp_text,
        },
        "diff_preview": "".join(diff_lines[:160]),
        "original_preview": "".join(original_lines[:120]),
        "converted_preview": "".join(converted_lines[:120]) if converted else "",
    }

    os.makedirs(os.path.dirname(settings.anchor_artifact_file), exist_ok=True)
    with open(settings.anchor_artifact_file, "w", encoding="utf-8") as fp:
        json.dump(artifact, fp, ensure_ascii=True, indent=2)
    return artifact


def load_real_anchor() -> dict | None:
    if not os.path.exists(settings.anchor_artifact_file):
        return None
    with open(settings.anchor_artifact_file, "r", encoding="utf-8") as fp:
        return json.load(fp)
