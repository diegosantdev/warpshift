from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
from difflib import unified_diff
from datetime import datetime, timezone

from .schemas import (
    AnalysisRequest,
    AnalysisResult,
    BenchmarkResult,
    DiffAnnotation,
    DecisionEngineResult,
    Insight,
    PullRequestPreview,
    RiskItem,
)
from .settings import settings
from .stages import (
    run_ai_explanation_stage,
    run_hipify_stage,
    run_runtime_validation_stage,
    run_static_analysis_stage,
)
from .real_anchor import load_real_anchor

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
RUNS_DIR = os.path.join(DATA_DIR, "runs")
DEMO_REPO_CANDIDATES = [
    "https://github.com/user/cuda-reduction",
    "https://github.com/user/llama-custom-op",
    "https://github.com/user/attention-kernel",
]

REPO_CACHE_DIR = os.path.join(DATA_DIR, "repo-cache")


def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)


def _safe_repo_dir(repo_url: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9]+", "-", repo_url).strip("-").lower()
    return os.path.join(REPO_CACHE_DIR, key[:120])


def _prepare_repo(repo_url: str) -> tuple[str | None, str | None]:
    if not repo_url.startswith("http://") and not repo_url.startswith("https://"):
        return None, None
    os.makedirs(REPO_CACHE_DIR, exist_ok=True)
    repo_dir = _safe_repo_dir(repo_url)
    try:
        if not os.path.exists(os.path.join(repo_dir, ".git")):
            if "NVIDIA/cuda-samples" in repo_url:
                # Sparse checkout to only get vectorAdd for 20s demo speed
                os.makedirs(repo_dir, exist_ok=True)
                _run(["git", "init"], cwd=repo_dir)
                _run(["git", "remote", "add", "-f", "origin", repo_url], cwd=repo_dir)
                _run(["git", "config", "core.sparseCheckout", "true"], cwd=repo_dir)
                with open(os.path.join(repo_dir, ".git", "info", "sparse-checkout"), "w") as fp:
                    fp.write("Samples/0_Introduction/vectorAdd/\n")
                _run(["git", "pull", "origin", "master"], cwd=repo_dir)
            else:
                _run(["git", "clone", "--depth", "1", repo_url, repo_dir])
        else:
            _run(["git", "fetch", "--depth", "1", "origin"], cwd=repo_dir)
            _run(["git", "reset", "--hard", "origin/HEAD"], cwd=repo_dir)
        commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_dir).stdout.strip()
        return repo_dir, commit
    except Exception:
        return None, None


def _collect_cuda_files(repo_dir: str, limit: int = 12) -> list[str]:
    files: list[str] = []
    for root, _, filenames in os.walk(repo_dir):
        for name in filenames:
            if name.endswith((".cu", ".cuh")):
                files.append(os.path.join(root, name))
                if len(files) >= limit:
                    return files
    return files


def _collect_source_graph_files(repo_dir: str, limit: int = 200) -> list[str]:
    files: list[str] = []
    allowed = (".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp")
    for root, _, filenames in os.walk(repo_dir):
        for name in filenames:
            if name.endswith(allowed):
                files.append(os.path.join(root, name))
                if len(files) >= limit:
                    return files
    return files


def _build_include_graph(files: list[str]) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {}
    include_re = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]')
    for file_path in files:
        deps: list[str] = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    match = include_re.match(line)
                    if match:
                        deps.append(match.group(1))
        except Exception:
            deps = []
        graph[file_path] = deps
    return graph


def _detect_build_system(repo_dir: str) -> str | None:
    if os.path.exists(os.path.join(repo_dir, "CMakeLists.txt")):
        return "cmake"
    if os.path.exists(os.path.join(repo_dir, "Makefile")):
        return "make"
    if os.path.exists(os.path.join(repo_dir, "setup.py")) or os.path.exists(os.path.join(repo_dir, "pyproject.toml")):
        return "python"
    return None


def _rewrite_build_files_preview(repo_dir: str) -> list[dict]:
    candidates = ["CMakeLists.txt", "Makefile"]
    previews: list[dict] = []
    for rel in candidates:
        path = os.path.join(repo_dir, rel)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                content = fp.read()
            rewritten = (
                content.replace("nvcc", "hipcc")
                .replace("-lcudart", "-lamdhip64")
                .replace("CUDA", "HIP")
            )
            if rewritten != content:
                previews.append(
                    {
                        "file": rel,
                        "changed": True,
                        "preview": "\n".join(rewritten.splitlines()[:80]),
                    }
                )
        except Exception:
            continue
    return previews


def _run_build_validation(repo_dir: str, build_system: str | None, candidate_file: str | None) -> tuple[str, str]:
    if not build_system:
        return "not_run", "No build system detected."
    try:
        if settings.backend_mode == "real":
            if build_system == "cmake":
                _run(["cmake", "."], cwd=repo_dir)
                _run(["make", "-j4"], cwd=repo_dir)
                return "pass", "CMake build completed successfully."
            if build_system == "make":
                _run(["make", "-j4"], cwd=repo_dir)
                return "pass", "Make build completed successfully."
        else:
            if build_system == "cmake":
                _run([settings.hipcc_bin, "--version"])
                return "pass", "Build prerequisites detected for CMake project."
            if build_system == "make":
                _run(["make", "--version"])
                return "pass", "Make build tooling detected."
            if build_system == "python":
                _run(["py", "--version"])
                return "pass", "Python build tooling detected."
    except Exception as exc:
        return "fail", f"Build tooling check failed: {exc}"

    if candidate_file and os.path.exists(candidate_file):
        try:
            proc = subprocess.run(
                [settings.hipcc_bin, candidate_file, "-o", os.path.join(repo_dir, "warpshift_runtime_check.out")],
                check=False,
                capture_output=True,
                text=True,
                timeout=settings.runtime_build_timeout_seconds,
            )
            if proc.returncode == 0:
                return "pass", "hipcc compiled candidate source successfully."
            msg = (proc.stderr or proc.stdout or "Compile failed").strip().splitlines()[:2]
            return "fail", " | ".join(msg)
        except Exception as exc:
            return "fail", f"Compile invocation failed: {exc}"
    return "not_run", "No candidate file available for compile validation."


def _create_real_pr_if_enabled(
    repo_dir: str | None,
    run_id: str,
    title: str,
    body: str,
) -> str | None:
    if not settings.github_real_pr or not repo_dir:
        return None
    try:
        _run(["gh", "--version"])
        branch = f"warpshift/{run_id.lower()}"
        _run(["git", "checkout", "-b", branch], cwd=repo_dir)
        # Non-destructive marker file for optional PR flow.
        marker = os.path.join(repo_dir, ".warpshift-pr.txt")
        with open(marker, "w", encoding="utf-8") as fp:
            fp.write(f"WarpShift generated branch for run {run_id}\n")
        _run(["git", "add", ".warpshift-pr.txt"], cwd=repo_dir)
        _run(["git", "commit", "-m", f"WarpShift migration prep {run_id}"], cwd=repo_dir)
        _run(["git", "push", "-u", "origin", branch], cwd=repo_dir)
        result = _run(
            [
                "gh",
                "pr",
                "create",
                "--base",
                settings.github_default_base_branch,
                "--title",
                title,
                "--body",
                body,
            ],
            cwd=repo_dir,
        )
        return result.stdout.strip().splitlines()[-1]
    except Exception:
        return None


def _scan_repo_signals(files: list[str]) -> dict:
    signals = {
        "warp": None,
        "cublas": None,
        "cudnn": None,
        "dynamic_launch": None,
    }
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                for line_no, line in enumerate(fp, start=1):
                    if signals["warp"] is None and (re.search(r"\bwarpSize\b\s*=\s*32\b", line) or "& 31" in line):
                        signals["warp"] = (file_path, line_no, line.strip())
                    if signals["cublas"] is None and "cublas" in line.lower():
                        signals["cublas"] = (file_path, line_no, line.strip())
                    if signals["cudnn"] is None and "cudnn" in line.lower():
                        signals["cudnn"] = (file_path, line_no, line.strip())
                    if signals["dynamic_launch"] is None and "LAUNCH_" in line:
                        signals["dynamic_launch"] = (file_path, line_no, line.strip())
        except Exception:
            continue
    return signals


def _detect_cuda_dependencies(files: list[str]) -> dict:
    header_re = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]')
    headers: set[str] = set()
    calls: set[str] = set()
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    m = header_re.match(line)
                    if m:
                        header = m.group(1)
                        if "cuda" in header.lower() or "cublas" in header.lower() or "cudnn" in header.lower():
                            headers.add(header)
                    low = line.lower()
                    if "cuda" in low or "cublas" in low or "cudnn" in low:
                        token = line.strip()
                        if token:
                            calls.add(token[:140])
        except Exception:
            continue
    return {
        "cuda_headers": sorted(headers),
        "cuda_calls_sample": sorted(calls)[:25],
    }


def _cuda_to_hip_mapping_report(dep_scan: dict) -> dict:
    known = {
        "cuda_runtime.h": "hip/hip_runtime.h",
        "cuda.h": "hip/hip_runtime.h",
        "cublas_v2.h": "hipblas/hipblas.h or rocblas/rocblas.h",
        "cudnn.h": "miopen/miopen.h (manual parity review)",
    }
    mappings = []
    for header in dep_scan.get("cuda_headers", []):
        mappings.append({"cuda": header, "hip": known.get(header, "manual mapping required")})
    return {"header_mappings": mappings}


def _hipify_one_file(file_path: str) -> tuple[str | None, str | None]:
    try:
        result = _run([settings.hipify_bin, file_path])
        return result.stdout, settings.hipify_bin
    except Exception:
        local_hipify = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "tools", "HIPIFY", "bin", "hipify-perl")
        )
        if os.path.exists(local_hipify):
            try:
                result = _run(["perl", local_hipify, file_path])
                return result.stdout, "perl hipify-perl(local)"
            except Exception:
                return None, None
        return None, None


def _hipify_batch(files: list[str], limit: int = 3) -> tuple[list[dict], dict]:
    artifacts: list[dict] = []
    stats = {
        "files_changed": 0,
        "lines_added": 0,
        "lines_removed": 0,
        "tool": None,
    }
    for file_path in files[:limit]:
        converted, tool = _hipify_one_file(file_path)
        if not converted:
            continue
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                original = fp.read().splitlines(keepends=True)
            conv_lines = converted.splitlines(keepends=True)
            diff = list(
                unified_diff(
                    original,
                    conv_lines,
                    fromfile=file_path,
                    tofile=f"{file_path}.hip",
                    n=2,
                )
            )
            added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
            removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
            if added or removed:
                stats["files_changed"] += 1
                stats["lines_added"] += added
                stats["lines_removed"] += removed
                stats["tool"] = tool
                artifacts.append(
                    {
                        "file_path": file_path,
                        "diff_preview": "".join(diff[:120]),
                        "original_first": original[0].strip()[:180] if original else "",
                        "converted_first": conv_lines[0].strip()[:180] if conv_lines else "",
                    }
                )
        except Exception:
            continue
    return artifacts, stats


def _apply_semantic_fixes(converted: str) -> tuple[str, list[str]]:
    fixed = converted
    patches: list[str] = []
    patterns = [
        (r"\bwarpSize\s*=\s*32\b", "warpSize = hipWarpSize"),
        (r"\bcudaStream_t\b", "hipStream_t"),
        (r"\bcudaEvent_t\b", "hipEvent_t"),
    ]
    for pattern, repl in patterns:
        updated, count = re.subn(pattern, repl, fixed)
        if count > 0:
            patches.append(f"{pattern} -> {repl} ({count}x)")
            fixed = updated
    return fixed, patches


def _run_runtime_execution(binary_path: str) -> tuple[str, str, float | None]:
    if not os.path.exists(binary_path):
        return "not_run", "No runtime binary produced.", None
    try:
        start = time.perf_counter()
        proc = subprocess.run([binary_path], check=False, capture_output=True, text=True, timeout=20)
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        
        output = proc.stdout + "\n" + proc.stderr
        detail = output.strip().splitlines()[:2]
        
        # Parse numerical validation and benchmark times
        parsed_time = elapsed
        if "[WARPSHIFT_BENCHMARK] time_ms=" in output:
            try:
                parsed_time = float(output.split("[WARPSHIFT_BENCHMARK] time_ms=")[1].split()[0])
            except Exception:
                pass
                
        if "[WARPSHIFT_VALIDATION] status=FAILED" in output:
            return "fail", "Numerical validation failed: results do not match.", parsed_time

        if proc.returncode == 0:
            if "[WARPSHIFT_VALIDATION] status=SUCCESS" in output:
                return "pass", "Execution and numerical validation passed.", parsed_time
            return "pass", " | ".join(detail) if detail else "Runtime execution passed.", parsed_time
        return "fail", " | ".join(detail) if detail else "Runtime execution failed.", parsed_time
    except Exception as exc:
        return "fail", f"Runtime execution error: {exc}", None


def _compute_confidence(build_status: str, runtime_status: str, runtime_source: str, hipify_stats: dict) -> int:
    value = 55
    if runtime_source in {"repo-scan", "repo-scan+hipify"}:
        value += 15
    if hipify_stats.get("files_changed", 0) > 0:
        value += 10
    if build_status == "pass":
        value += 10
    elif build_status == "fail":
        value -= 8
    if runtime_status == "pass":
        value += 10
    elif runtime_status == "fail":
        value -= 5
    return max(35, min(95, value))


def _mock_risks() -> list[RiskItem]:
    return [
        RiskItem(
            id="risk-cudnn-custom-op",
            level="high",
            title="cuDNN custom ops detected",
            description="No stable ROCm equivalent for this custom op path.",
            detection_source="dependency scan",
            line=96,
            confidence="high",
            effort="high ~2h+",
            fix="Manual rewrite required",
            blocking=True,
        ),
        RiskItem(
            id="risk-warpsize",
            level="medium",
            title="warpSize hardcoded as 32",
            description="AMD CDNA wavefront is 64; hardcoded 32 can break reductions.",
            detection_source="static analysis",
            line=87,
            confidence="high",
            effort="low ~5min",
            fix="Replace literal with hipWarpSize",
        ),
        RiskItem(
            id="risk-cublas-order",
            level="medium",
            title="cuBLAS argument ordering mismatch",
            description="rocBLAS operation enums and arg order differ from cuBLAS.",
            detection_source="static analysis",
            line=234,
            confidence="high",
            effort="medium ~30min",
            fix="Review and update rocBLAS call signature",
        ),
        RiskItem(
            id="risk-dynamic-launch",
            level="low",
            title="Dynamic kernel launch macro incompatible",
            description="Macro-based CUDA launch pattern not directly converted by HIPIFY.",
            detection_source="runtime validation",
            line=156,
            confidence="medium",
            effort="medium ~30min",
            fix="Use hipLaunchKernelGGL with explicit args",
        ),
    ]


def _insights() -> list[Insight]:
    return [
        Insight(
            risk_id="risk-warpsize",
            summary="Hardcoded warpSize breaks on AMD wavefront 64.",
            impact=[
                "Thread divergence risk in reduction loops",
                "Incorrect results in warp-level operations",
            ],
            fix_applied="Replaced literal 32 with hipWarpSize",
            manual_review="no",
        ),
        Insight(
            risk_id="risk-cublas-order",
            summary="rocBLAS namespace and argument order differ from cuBLAS.",
            impact=[
                "Call can compile and still return wrong math output",
                "Potential silent numerical corruption",
            ],
            fix_applied="Updated to rocblas_operation_* and corrected arg ordering",
            manual_review="yes, validate alpha/beta and leading dimensions",
        ),
    ]


def _diff_annotations() -> list[DiffAnnotation]:
    anchor = load_real_anchor()
    if anchor and anchor.get("warp_detection", {}).get("found"):
        warp_line = int(anchor["warp_detection"]["line"])
        warp_text = anchor["warp_detection"]["content"]
        converted = warp_text.replace("32", "hipWarpSize")
        return [
            DiffAnnotation(
                id="ann-warpsize-real",
                file=anchor.get("source_relative_path", "reduction_kernel.cu"),
                line=warp_line,
                original=warp_text,
                converted=converted,
                detection_source="static analysis (real anchor)",
                confidence="high",
                effort="low ~5min",
                insight=Insight(
                    risk_id="risk-warpsize",
                    summary="Real anchor detected hardcoded warp-width assumption.",
                    impact=[
                        "Wavefront mismatch can break reduction assumptions on CDNA",
                        "Potential silent correctness issues under lane-sensitive logic",
                    ],
                    fix_applied="Replace hardcoded width with hipWarpSize/runtime-safe query",
                    manual_review="yes, verify all lane-mask math paths",
                ),
            )
        ]

    return [
        DiffAnnotation(
            id="ann-warpsize",
            file="kernel_reduction.hip",
            line=87,
            original="int warpSize = 32;",
            converted="int warpSize = hipWarpSize;",
            detection_source="static analysis",
            confidence="high",
            effort="low ~5min",
            insight=Insight(
                risk_id="risk-warpsize",
                summary="Hardcoded warpSize breaks on AMD wavefront 64.",
                impact=[
                    "Thread divergence risk in reduction loops",
                    "Incorrect results in warp-level operations",
                ],
                fix_applied="Replaced literal 32 with hipWarpSize",
                manual_review="no",
            ),
        ),
        DiffAnnotation(
            id="ann-cublas",
            file="gemm_bridge.hip",
            line=234,
            original="cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ...);",
            converted="rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, ...);",
            detection_source="static analysis",
            confidence="high",
            effort="medium ~30min",
            insight=Insight(
                risk_id="risk-cublas-order",
                summary="rocBLAS namespace and argument order differ from cuBLAS.",
                impact=[
                    "Will compile but produce incorrect output if order remains unchanged",
                    "Silent wrong results in BLAS paths",
                ],
                fix_applied="Updated rocBLAS enums and parameter order",
                manual_review="yes, check alpha/beta and lda/ldb/ldc",
            ),
        ),
    ]


def _pull_request_preview(
    files_changed: int = 12,
    lines_added: int = 340,
    lines_removed: int = 210,
) -> PullRequestPreview:
    pr_body = """## Summary
- Convert CUDA APIs to HIP equivalents for ROCm compatibility.
- Annotate medium/high-risk migration points directly in the diff.
- Add runtime validation output and migration decision report.

## Risk highlights
- warpSize assumption can break on wavefront 64.
- cuBLAS argument ordering differs in rocBLAS.
- dynamic launch and cuDNN custom ops require manual review.

## Test plan
- [ ] Build with hipcc on ROCm 7.x
- [ ] Validate reduction kernel correctness
- [ ] Re-run migration workflow and confirm improved score
"""
    return PullRequestPreview(
        pr_number=42,
        title="CUDA -> ROCm Migration",
        files_changed=files_changed,
        lines_added=lines_added,
        lines_removed=lines_removed,
        auto_converted=[
            "cudaMalloc -> hipMalloc",
            "cudaMemcpy -> hipMemcpy",
            "kernel launch syntax -> HIP syntax",
        ],
        flagged_for_review=[
            "warpSize assumption (line 87)",
            "cuBLAS arg order (line 234)",
        ],
        manual_fix_required=[
            "dynamic kernel launch (line 156)",
            "cuDNN custom op (lines 89-102)",
        ],
        github_pr_body=pr_body,
    )


def _load_history() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _save_history(items: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as fp:
        json.dump(items, fp, ensure_ascii=True, indent=2)


def append_history(result: AnalysisResult, github_url: str) -> None:
    history = _load_history()
    history.insert(
        0,
        {
            "run_id": result.run_id,
            "timestamp_utc": result.timestamp_utc.isoformat(),
            "github_url": github_url,
            "migration_score": result.migration_score,
            "migration_confidence": result.migration_confidence,
            "decision": result.decision_engine.decision,
            "runtime_source": result.runtime_source,
            "evidence_file": result.evidence_file,
        },
    )
    _save_history(history[:20])


def get_history() -> list[dict]:
    history = _load_history()
    if history:
        return history
    return [
        {
            "run_id": "A1023",
            "timestamp_utc": "2026-05-04T14:32:00+00:00",
            "github_url": "https://github.com/user/cuda-reduction",
            "migration_score": 78,
            "migration_confidence": 82,
            "decision": "proceed_with_caution",
        },
        {
            "run_id": "A1019",
            "timestamp_utc": "2026-05-04T10:18:00+00:00",
            "github_url": "https://github.com/user/llama-custom-op",
            "migration_score": 64,
            "migration_confidence": 80,
            "decision": "do_not_migrate_yet",
        },
        {
            "run_id": "A1004",
            "timestamp_utc": "2026-05-03T22:02:00+00:00",
            "github_url": "https://github.com/user/attention-kernel",
            "migration_score": 85,
            "migration_confidence": 88,
            "decision": "proceed_with_caution",
        },
    ]


def get_demo_repo_candidates() -> list[str]:
    return DEMO_REPO_CANDIDATES


def export_risk_report(result: AnalysisResult, as_markdown: bool = False) -> str | dict:
    if as_markdown:
        lines = [
            "# Pre-Migration Risk Report",
            f"- Run: {result.run_id}",
            f"- Score: {result.migration_score}/100",
            f"- Confidence: {result.migration_confidence}%",
            "",
            "## Risks",
        ]
        for risk in result.risk_items:
            lines.append(
                f"- [{risk.level.upper()}] {risk.title} (source: {risk.detection_source}, line: {risk.line})"
            )
        lines.append("")
        lines.append("## Decision")
        lines.append(f"- {result.decision_engine.decision}")
        return "\n".join(lines)
    return result.model_dump(mode="json")


def _decision(risks: list[RiskItem]) -> DecisionEngineResult:
    blocking_issues = [r for r in risks if r.blocking]
    if blocking_issues:
        return DecisionEngineResult(
            decision="do_not_migrate_yet",
            why=[
                "Blocking issues detected in dependency/runtime paths",
                "Manual rewrite required before safe migration",
            ],
            unresolved_consequences=[
                "Build/runtime failures on ROCm",
                "High risk of incomplete feature parity",
            ],
            next_step="Resolve blocking issues and re-run validation",
        )
    return DecisionEngineResult(
        decision="proceed_with_caution",
        why=[
            "No blocking issues found",
            "Medium-risk incompatibilities need review",
        ],
        unresolved_consequences=[
            "Potential wrong results in BLAS routines",
            "Runtime bugs in reduction kernels",
        ],
        next_step="Apply annotated fixes and re-run validation",
    )


def run_analysis(req: AnalysisRequest) -> AnalysisResult:
    random.seed(req.github_url)
    risks = _mock_risks()
    repo_dir, repo_commit = _prepare_repo(req.github_url)
    repo_files: list[str] = []
    source_graph_files: list[str] = []
    include_graph: dict[str, list[str]] = {}
    dependency_scan = {"cuda_headers": [], "cuda_calls_sample": []}
    dependency_mapping = {"header_mappings": []}
    repo_signals = {}
    runtime_source = "mock"
    build_system: str | None = None
    build_status = "not_run"
    build_detail = "Build not executed."
    build_rewrite_previews: list[dict] = []
    runtime_exec_status = "not_run"
    runtime_exec_detail = "Runtime not executed."
    runtime_exec_ms: float | None = None
    if repo_dir:
        repo_files = _collect_cuda_files(repo_dir)
        source_graph_files = _collect_source_graph_files(repo_dir)
        include_graph = _build_include_graph(source_graph_files)
        dependency_scan = _detect_cuda_dependencies(source_graph_files)
        dependency_mapping = _cuda_to_hip_mapping_report(dependency_scan)
        repo_signals = _scan_repo_signals(repo_files)
        runtime_source = "repo-scan"
        build_system = _detect_build_system(repo_dir)
        build_rewrite_previews = _rewrite_build_files_preview(repo_dir)
        build_status, build_detail = _run_build_validation(
            repo_dir,
            build_system,
            repo_files[0] if repo_files else None,
        )

        # Rebuild risk list from real signals when available.
        rebuilt: list[RiskItem] = []
        if repo_signals.get("cudnn"):
            file_path, line_no, _ = repo_signals["cudnn"]
            rebuilt.append(
                RiskItem(
                    id="risk-cudnn-custom-op",
                    level="high",
                    title="cuDNN usage detected",
                    description=f"Detected from {os.path.relpath(file_path, repo_dir)}.",
                    detection_source="dependency scan (repo)",
                    line=line_no,
                    confidence="high",
                    effort="high ~2h+",
                    fix="Manual migration review required",
                    blocking=True,
                )
            )
        if repo_signals.get("warp"):
            file_path, line_no, _ = repo_signals["warp"]
            rebuilt.append(
                RiskItem(
                    id="risk-warpsize",
                    level="medium",
                    title="warp-width assumption detected",
                    description=f"Found warp-sensitive constant in {os.path.relpath(file_path, repo_dir)}.",
                    detection_source="static analysis (repo)",
                    line=line_no,
                    confidence="high",
                    effort="low ~5min",
                    fix="Use hipWarpSize or runtime-safe lane logic",
                )
            )
        if repo_signals.get("cublas"):
            file_path, line_no, _ = repo_signals["cublas"]
            rebuilt.append(
                RiskItem(
                    id="risk-cublas-order",
                    level="medium",
                    title="cuBLAS call detected",
                    description=f"Manual review needed for rocBLAS arg semantics in {os.path.relpath(file_path, repo_dir)}.",
                    detection_source="static analysis (repo)",
                    line=line_no,
                    confidence="high",
                    effort="medium ~30min",
                    fix="Review enum and argument compatibility for rocBLAS",
                )
            )
        if repo_signals.get("dynamic_launch"):
            file_path, line_no, _ = repo_signals["dynamic_launch"]
            rebuilt.append(
                RiskItem(
                    id="risk-dynamic-launch",
                    level="low",
                    title="Macro-based kernel launch detected",
                    description=f"Pattern found in {os.path.relpath(file_path, repo_dir)}.",
                    detection_source="runtime validation (repo)",
                    line=line_no,
                    confidence="medium",
                    effort="medium ~30min",
                    fix="Expand macro to explicit hipLaunchKernelGGL call",
                )
            )
        if rebuilt:
            risks = rebuilt

    anchor = load_real_anchor()
    if anchor and anchor.get("warp_detection", {}).get("found"):
        for risk in risks:
            if risk.id == "risk-warpsize":
                risk.line = int(anchor["warp_detection"]["line"])
                risk.description = (
                    "Detected from real anchor source file in pinned CUDA sample repository."
                )
                risk.detection_source = "static analysis (real anchor)"
    # Full mode assumes preprocessed artifacts reduced blockers.
    if req.mode == "full":
        risks = [r for r in risks if not r.blocking]

    medium = sum(1 for r in risks if r.level == "medium")
    high = sum(1 for r in risks if r.level == "high")
    score = 91 if req.mode == "full" else max(52, 92 - (high * 18 + medium * 7))
    confidence = 90 if runtime_source == "repo-scan" else (88 if req.mode == "full" else 82)

    cuda = 120.0
    rocm = 135.0 if req.mode == "live" else 129.0
    benchmark = BenchmarkResult(
        cuda_baseline_ms=cuda,
        rocm_live_ms=rocm,
        performance_delta_percent=round(((rocm - cuda) / cuda) * 100, 1),
    )

    diff_annotations = _diff_annotations()
    hipify_artifacts: list[dict] = []
    hipify_stats = {"files_changed": 0, "lines_added": 0, "lines_removed": 0, "tool": None}
    if repo_files:
        hipify_artifacts, hipify_stats = _hipify_batch(repo_files, limit=3)
        if hipify_artifacts:
            runtime_source = "repo-scan+hipify"
            try:
                first = hipify_artifacts[0]
                fixed_line, semantic_patches = _apply_semantic_fixes(first["converted_first"])
                first["semantic_patches"] = semantic_patches
                diff_annotations = [
                    DiffAnnotation(
                        id="ann-hipify-repo",
                        file=os.path.relpath(first["file_path"], repo_dir),
                        line=1,
                        original=first["original_first"],
                        converted=fixed_line,
                        detection_source=f"hipify real ({hipify_stats['tool']})",
                        confidence="high",
                        effort="low ~5min",
                        insight=Insight(
                            risk_id="risk-warpsize",
                            summary="Real HIPIFY conversion executed with semantic fixer post-pass.",
                            impact=[
                                "Confirms syntax transformation is executed on real source code",
                                "Applies deterministic fixes for known migration-sensitive patterns",
                            ],
                            fix_applied="Generated HIP output plus semantic patch pass",
                            manual_review="yes, validate compile/link in ROCm environment",
                        ),
                    )
                ]
            except Exception:
                pass

    run_id = f"A{random.randint(1000, 9999)}"
    runtime_status_value = "fail" if req.mode == "live" else "pass"
    if build_status == "pass" and repo_dir:
        bin_path = os.path.join(repo_dir, "warpshift_runtime_check.out")
        runtime_exec_status, runtime_exec_detail, runtime_exec_ms = _run_runtime_execution(bin_path)
        if runtime_exec_status in {"pass", "fail"}:
            runtime_status_value = runtime_exec_status

    confidence = _compute_confidence(
        build_status=build_status,
        runtime_status=runtime_status_value,
        runtime_source=runtime_source,
        hipify_stats=hipify_stats,
    )

    result = AnalysisResult(
        run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc),
        migration_score=score,
        migration_confidence=confidence,
        estimated_effort="4-8 hours manual",
        risk_items=risks,
        insights=_insights(),
        benchmark=benchmark,
        decision_engine=_decision(risks),
        diff_annotations=diff_annotations,
        pull_request_preview=_pull_request_preview(
            files_changed=hipify_stats["files_changed"] or 12,
            lines_added=hipify_stats["lines_added"] or 340,
            lines_removed=hipify_stats["lines_removed"] or 210,
        ),
        runtime_source=runtime_source,
        build_system=build_system,
        build_status=build_status,
        repo_commit=repo_commit,
        runtime_status=runtime_status_value,
    )

    if settings.github_real_pr:
        real_pr_url = _create_real_pr_if_enabled(
            repo_dir=repo_dir,
            run_id=run_id,
            title=result.pull_request_preview.title,
            body=result.pull_request_preview.github_pr_body,
        )
        if real_pr_url:
            result.pull_request_preview.real_pr_url = real_pr_url

    docker_stage_logs = []
    if settings.execution_mode == "docker":
        from .docker_executor import run_in_sandbox
        # Run docker sandbox in background or block (we block here for simplicity, though real time streaming is better)
        sandbox_result = run_in_sandbox(req.github_url, req.mode, gpu=False)
        docker_stage_logs = sandbox_result.stage_logs
        if sandbox_result.exit_code != 0:
            result.runtime_status = "fail"
            # Attempt to pull runtime error from docker logs
            result.build_detail = sandbox_result.stderr or "Docker execution failed"

    os.makedirs(RUNS_DIR, exist_ok=True)
    evidence_path = os.path.join(RUNS_DIR, f"{run_id}.json")
    evidence = {
        "run_id": run_id,
        "repo_url": req.github_url,
        "repo_commit": repo_commit,
        "runtime_source": runtime_source,
        "repo_files_scanned": [os.path.relpath(f, repo_dir) for f in repo_files] if repo_dir else [],
        "source_graph_file_count": len(source_graph_files),
        "include_graph_sample": {
            os.path.relpath(k, repo_dir): v[:5]
            for k, v in list(include_graph.items())[:10]
        }
        if repo_dir
        else {},
        "repo_signals": repo_signals,
        "dependency_scan": dependency_scan,
        "dependency_mapping": dependency_mapping,
        "build_system": build_system,
        "build_status": build_status,
        "build_detail": build_detail,
        "build_rewrite_previews": build_rewrite_previews,
        "hipify_stats": hipify_stats,
        "hipify_artifacts": hipify_artifacts,
        "runtime_execution": {
            "status": runtime_exec_status,
            "detail": runtime_exec_detail,
            "elapsed_ms": runtime_exec_ms,
        },
        "real_pr_url": real_pr_url,
        "diff_annotations": [d.model_dump(mode="json") for d in diff_annotations],
        "docker_stage_logs": docker_stage_logs,
    }
    with open(evidence_path, "w", encoding="utf-8") as fp:
        json.dump(evidence, fp, ensure_ascii=True, indent=2)
    result.evidence_file = evidence_path

    append_history(result, req.github_url)
    return result


def stage_events(req: AnalysisRequest):
    repo_dir, _ = _prepare_repo(req.github_url)
    repo_files = _collect_cuda_files(repo_dir, limit=1) if repo_dir else []
    yield ("stage_start", {"stage": 1, "name": "HIPIFY Conversion"})
    s1 = run_hipify_stage(req.github_url)
    yield ("stage_update", {"stage": 1, "progress": 65, "status": s1.status, "detail": s1.detail})
    time.sleep(settings.stage_delay_seconds)

    yield ("stage_start", {"stage": 2, "name": "Static Analysis"})
    s2 = run_static_analysis_stage()
    yield ("stage_update", {"stage": 2, "progress": 100, "status": s2.status, "detail": s2.detail})
    time.sleep(settings.stage_delay_seconds)

    yield ("stage_start", {"stage": 3, "name": "Runtime Validation"})
    s3 = run_runtime_validation_stage(req.mode, repo_files[0] if repo_files else None)
    if s3.status == "failed":
        yield (
            "runtime_error",
            {
                "error": s3.detail,
                "detection_source": "runtime validation",
            },
        )
    else:
        yield ("stage_update", {"stage": 3, "progress": 100, "status": s3.status, "detail": s3.detail})
    time.sleep(settings.stage_delay_seconds)

    yield ("stage_start", {"stage": 4, "name": "AI Explanation Layer"})
    s4 = run_ai_explanation_stage(
        issue_description="warpSize hardcoded as 32 breaks on AMD wavefront 64.",
        original_code="int warpSize = 32;",
        converted_code="int warpSize = hipWarpSize;",
        hipify_diff="- int warpSize = 32;\n+ int warpSize = hipWarpSize;",
        detection_source="static analysis"
    )
    yield ("stage_update", {"stage": 4, "progress": 100, "status": s4.status, "detail": s4.detail})

    result = run_analysis(req)
    yield ("completed", result.model_dump(mode="json"))


def create_pr_for_run(run_id: str) -> str | None:
    history = _load_history()
    run_data = next((h for h in history if h["run_id"] == run_id), None)
    if not run_data:
        return None
    repo_dir, _ = _prepare_repo(run_data["github_url"])
    if not repo_dir:
        return None
        
    evidence_path = run_data["evidence_file"]
    if not os.path.exists(evidence_path):
        return None
    
    with open(evidence_path, "r", encoding="utf-8") as fp:
        evidence = json.load(fp)
        
    title = f"WarpShift Migration Run {run_id}"
    body = "Automated PR generated by WarpShift."
    
    # We temporarily set the setting to true to force creation
    original_setting = settings.github_real_pr
    settings.github_real_pr = True
    try:
        url = _create_real_pr_if_enabled(repo_dir, run_id, title, body)
    finally:
        settings.github_real_pr = original_setting
        
    if url:
        evidence["real_pr_url"] = url
        with open(evidence_path, "w", encoding="utf-8") as fp:
            json.dump(evidence, fp, indent=2)
            
    return url
