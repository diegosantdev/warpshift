"""
ssh_executor.py — Real GPU execution on AMD MI300X via SSH.

Uses the system `ssh`/`scp` binaries (no extra dependencies).
Falls back gracefully with structured errors if SSH is unavailable.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .settings import settings


# ── SAXPY benchmark source (pure HIP, used as the migration target) ──────────
_SAXPY_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N (1 << 24)
#define ALPHA 2.0f

__global__ void saxpy(float a, float* x, float* y, float* out, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out[tid] = a * x[tid] + y[tid];
}

int main() {
    float *hx, *hy, *hout, *dx, *dy, *dout;
    size_t bytes = N * sizeof(float);

    hx   = (float*)malloc(bytes);
    hy   = (float*)malloc(bytes);
    hout = (float*)malloc(bytes);
    for (size_t i = 0; i < N; i++) { hx[i] = 1.0f; hy[i] = 2.0f; }

    hipMalloc(&dx, bytes); hipMalloc(&dy, bytes); hipMalloc(&dout, bytes);
    hipMemcpy(dx, hx, bytes, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy, bytes, hipMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // Warmup
    hipLaunchKernelGGL(saxpy, dim3(blocks), dim3(threads), 0, 0,
                       ALPHA, dx, dy, dout, N);
    hipDeviceSynchronize();

    // Timed: 10 iterations
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);
    for (int i = 0; i < 10; i++)
        hipLaunchKernelGGL(saxpy, dim3(blocks), dim3(threads), 0, 0,
                           ALPHA, dx, dy, dout, N);
    hipEventRecord(stop);
    hipDeviceSynchronize();

    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);
    ms /= 10.0f;

    hipMemcpy(hout, dout, bytes, hipMemcpyDeviceToHost);

    int ok = 1;
    for (size_t i = 0; i < N; i++) {
        if (fabsf(hout[i] - (ALPHA * hx[i] + hy[i])) > 1e-4f) { ok = 0; break; }
    }

    if (ok)
        printf("[WARPSHIFT_VALIDATION] status=SUCCESS\n");
    else
        printf("[WARPSHIFT_VALIDATION] status=FAILED\n");

    printf("[WARPSHIFT_BENCHMARK] time_ms=%.4f n=%d alpha=%.1f hardware=MI300X\n",
           ms, N, ALPHA);
    printf("[WARPSHIFT_GPU] rocm_version=%s\n", HIP_VERSION_STRING);

    hipFree(dx); hipFree(dy); hipFree(dout);
    free(hx); free(hy); free(hout);
    return ok ? 0 : 1;
}
"""

# ── CUDA source that gets hipified on the remote (our benchmark_sample) ───────
_CUDA_SAXPY_SRC = r"""
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N (1 << 24)
#define ALPHA 2.0f

__global__ void saxpy_cuda(float a, float* x, float* y, float* out, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // warpSize hardcoded — will be flagged by WarpShift static analysis
    int warp = 32;
    if (tid < n) out[tid] = a * x[tid] + y[tid];
    (void)warp;
}

int main() {
    float *hx, *hy, *hout, *dx, *dy, *dout;
    size_t bytes = N * sizeof(float);

    hx   = (float*)malloc(bytes);
    hy   = (float*)malloc(bytes);
    hout = (float*)malloc(bytes);
    for (size_t i = 0; i < N; i++) { hx[i] = 1.0f; hy[i] = 2.0f; }

    cudaMalloc(&dx, bytes);
    cudaMalloc(&dy, bytes);
    cudaMalloc(&dout, bytes);
    cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // Warmup
    saxpy_cuda<<<blocks, threads>>>(ALPHA, dx, dy, dout, N);
    cudaDeviceSynchronize();

    // Timed
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        saxpy_cuda<<<blocks, threads>>>(ALPHA, dx, dy, dout, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f;

    cudaMemcpy(hout, dout, bytes, cudaMemcpyDeviceToHost);

    int ok = 1;
    for (size_t i = 0; i < N; i++) {
        if (fabsf(hout[i] - (ALPHA * hx[i] + hy[i])) > 1e-4f) { ok = 0; break; }
    }

    if (ok)
        printf("[WARPSHIFT_VALIDATION] status=SUCCESS\n");
    else
        printf("[WARPSHIFT_VALIDATION] status=FAILED\n");
    printf("[WARPSHIFT_BENCHMARK] time_ms=%.4f n=%d alpha=%.1f hardware=CUDA\n",
           ms, N, ALPHA);

    cudaFree(dx); cudaFree(dy); cudaFree(dout);
    free(hx); free(hy); free(hout);
    return ok ? 0 : 1;
}
"""


@dataclass
class GPUResult:
    """Structured result from a real GPU execution on MI300X."""
    success: bool
    bench_ms: Optional[float]
    validated: bool
    runtime_source: str        # e.g. "ssh+hipify+hipcc+gpu"
    hipify_output: str = ""
    hipify_diff: str = ""
    hipcc_version: str = ""
    hipcc_stderr: str = ""
    run_stdout: str = ""
    run_stderr: str = ""
    rocm_version: str = ""
    hardware: str = "AMD Instinct MI300X"
    error: str = ""
    remote_dir: str = ""
    ssh_host: str = ""
    ssh_port: int = 22


def _ssh_cmd(
    remote_cmd: str,
    host: str = "",
    user: str = "",
    key: str = "",
    port: int = 22,
    timeout: int = 60,
) -> tuple[int, str, str]:
    """Run a command on remote via SSH. Returns (exit_code, stdout, stderr)."""
    h = host or settings.ssh_host
    u = user or settings.ssh_user
    k = key or settings.ssh_key_path
    p = port or settings.ssh_port

    if not h:
        return -1, "", "SSH host not configured (MIGRATEAI_SSH_HOST)"

    cmd = [
        "ssh",
        "-i", k,
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={min(timeout, 20)}",
        "-o", "ServerAliveInterval=10",
        "-p", str(p),
        f"{u}@{h}",
        remote_cmd,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"SSH command timed out after {timeout}s"
    except FileNotFoundError:
        return -1, "", "ssh binary not found in PATH"
    except Exception as exc:
        return -1, "", str(exc)


def _scp_to_remote(
    local_path: str,
    remote_path: str,
    host: str = "",
    user: str = "",
    key: str = "",
    port: int = 22,
    timeout: int = 30,
) -> tuple[bool, str]:
    """SCP a file to remote. Returns (success, error_msg)."""
    h = host or settings.ssh_host
    u = user or settings.ssh_user
    k = key or settings.ssh_key_path
    p = port or settings.ssh_port

    cmd = [
        "scp",
        "-i", k,
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-P", str(p),
        local_path,
        f"{u}@{h}:{remote_path}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            return True, ""
        return False, (proc.stderr or proc.stdout or "scp failed").strip()
    except subprocess.TimeoutExpired:
        return False, f"scp timed out after {timeout}s"
    except Exception as exc:
        return False, str(exc)


def probe_ssh() -> tuple[bool, str, int]:
    """
    Detect working SSH connection: tries configured port, then common alternatives.
    Returns (connected, username, port).
    """
    host = settings.ssh_host
    key  = settings.ssh_key_path
    users = [settings.ssh_user] + [u for u in ("ubuntu", "user", "rocm", "amd", "root") if u != settings.ssh_user]
    ports = [settings.ssh_port] + [p for p in (22, 2222, 2200, 2020, 22222, 8022, 3022) if p != settings.ssh_port]

    for port in ports:
        for user in users:
            rc, out, err = _ssh_cmd(
                "echo WARPSHIFT_PROBE_OK",
                host=host, user=user, key=key, port=port, timeout=10,
            )
            if rc == 0 and "WARPSHIFT_PROBE_OK" in out:
                return True, user, port
    return False, "", 0


def check_rocm_environment() -> dict:
    """Query ROCm/hipcc environment on the remote machine."""
    host = settings.ssh_host
    if not host:
        return {"available": False, "error": "SSH host not configured"}

    script = (
        "set -e; "
        "echo HIPCC=$(which hipcc 2>/dev/null || echo MISSING); "
        "echo HIPIFY=$(which hipify-perl 2>/dev/null || echo MISSING); "
        "echo ROCMVER=$(cat /opt/rocm/.info/version 2>/dev/null || hipcc --version 2>&1 | head -1 || echo UNKNOWN); "
        "echo GPUCOUNT=$(ls /dev/kfd 2>/dev/null | wc -l || echo 0); "
        "echo GPU_INFO=$(rocm-smi --showproductname 2>/dev/null | grep 'AMD Instinct' | head -1 | tr -d '\n' || echo UNKNOWN)"
    )
    rc, out, err = _ssh_cmd(script, timeout=20)
    if rc != 0:
        return {"available": False, "error": err or out}

    info: dict = {"available": True, "raw": out}
    for line in out.splitlines():
        if line.startswith("HIPCC="):
            info["hipcc"] = line.split("=", 1)[1].strip()
        elif line.startswith("HIPIFY="):
            info["hipify"] = line.split("=", 1)[1].strip()
        elif line.startswith("ROCMVER="):
            info["rocm_version"] = line.split("=", 1)[1].strip()
        elif line.startswith("GPU_INFO="):
            info["gpu_info"] = line.split("=", 1)[1].strip()
    return info


def run_real_gpu_validation(mode: str = "saxpy_hip") -> GPUResult:
    """
    Full pipeline on the remote MI300X:
      1. Write CUDA source locally → SCP to remote
      2. Run hipify-perl on remote  → get HIP source + diff
      3. Compile with hipcc on remote
      4. Execute binary on GPU
      5. Parse [WARPSHIFT_BENCHMARK] and [WARPSHIFT_VALIDATION] markers
      6. Return GPUResult

    mode="saxpy_hip"  — skip hipify, use pure HIP SAXPY directly
    mode="saxpy_cuda" — upload CUDA → hipify → compile → run (full migration demo)
    """
    host = settings.ssh_host
    if not host:
        return GPUResult(
            success=False, bench_ms=None, validated=False,
            runtime_source="mock",
            error="MIGRATEAI_SSH_HOST not set",
        )

    run_id = uuid.uuid4().hex[:8]
    remote_dir = f"/tmp/warpshift_{run_id}"
    result = GPUResult(
        success=False, bench_ms=None, validated=False,
        runtime_source="ssh+hipcc+gpu",
        ssh_host=host,
        ssh_port=settings.ssh_port,
        remote_dir=remote_dir,
    )

    # ── Step 0: create remote working dir ───────────────────────────────────
    rc, out, err = _ssh_cmd(f"mkdir -p {remote_dir}", timeout=10)
    if rc != 0:
        result.error = f"Could not create remote dir: {err or out}"
        return result

    # ── Step 1: write source locally, SCP to remote ─────────────────────────
    with tempfile.TemporaryDirectory(prefix="warpshift_local_") as tmpdir:
        if mode == "saxpy_cuda":
            local_src = os.path.join(tmpdir, "saxpy.cu")
            with open(local_src, "w", encoding="utf-8") as fp:
                fp.write(_CUDA_SAXPY_SRC)
            remote_src = f"{remote_dir}/saxpy.cu"
            ok, err_msg = _scp_to_remote(local_src, remote_src)
            if not ok:
                result.error = f"SCP failed: {err_msg}"
                return result

            # ── Step 2: hipify-perl on remote ──────────────────────────────
            remote_hip = f"{remote_dir}/saxpy.hip"
            hipify_bin = f"{settings.rocm_path}/bin/hipify-perl"
            # fallback if not in rocm_path
            hipify_cmd = (
                f"if command -v hipify-perl >/dev/null 2>&1; then "
                f"  hipify-perl {remote_src} > {remote_hip}; "
                f"elif [ -x '{hipify_bin}' ]; then "
                f"  {hipify_bin} {remote_src} > {remote_hip}; "
                f"else "
                f"  echo HIPIFY_MISSING; exit 1; "
                f"fi && cat {remote_hip}"
            )
            rc, hip_out, hip_err = _ssh_cmd(hipify_cmd, timeout=30)
            if rc != 0:
                result.error = f"hipify-perl failed: {hip_err or hip_out}"
                return result

            result.hipify_output = hip_out[:2000]
            result.runtime_source = "ssh+hipify+hipcc+gpu"

            # Generate diff on remote
            rc2, diff_out, _ = _ssh_cmd(
                f"diff -u {remote_src} {remote_hip} || true",
                timeout=10,
            )
            result.hipify_diff = diff_out[:3000]
            compile_target = remote_hip

        else:
            # Pure HIP SAXPY — write and SCP
            local_hip = os.path.join(tmpdir, "saxpy.hip")
            with open(local_hip, "w", encoding="utf-8") as fp:
                fp.write(_SAXPY_HIP_SRC)
            remote_hip = f"{remote_dir}/saxpy.hip"
            ok, err_msg = _scp_to_remote(local_hip, remote_hip)
            if not ok:
                result.error = f"SCP failed: {err_msg}"
                return result
            compile_target = remote_hip
            result.runtime_source = "ssh+hipcc+gpu"

    # ── Step 3: get hipcc version ────────────────────────────────────────────
    rc, ver_out, _ = _ssh_cmd(
        "hipcc --version 2>&1 | head -3 || echo UNKNOWN",
        timeout=15,
    )
    result.hipcc_version = ver_out.strip()

    # ── Step 4: compile with hipcc ───────────────────────────────────────────
    remote_bin = f"{remote_dir}/saxpy_bench"
    compile_cmd = (
        f"hipcc {compile_target} -o {remote_bin} -lm "
        f"--offload-arch=gfx942 2>&1"  # MI300X = gfx942
    )
    rc, compile_out, compile_err = _ssh_cmd(compile_cmd, timeout=120)
    result.hipcc_stderr = (compile_out + compile_err).strip()[:1200]
    if rc != 0:
        result.error = f"hipcc compile failed (exit {rc}): {result.hipcc_stderr[:400]}"
        return result

    # ── Step 5: execute on GPU ───────────────────────────────────────────────
    t0 = time.perf_counter()
    rc, run_out, run_err = _ssh_cmd(
        f"chmod +x {remote_bin} && {remote_bin}",
        timeout=90,
    )
    wall_ms = round((time.perf_counter() - t0) * 1000, 2)

    result.run_stdout = run_out.strip()[:1200]
    result.run_stderr = run_err.strip()[:400]
    combined = run_out + "\n" + run_err

    # ── Step 6: parse output markers ────────────────────────────────────────
    result.validated = "[WARPSHIFT_VALIDATION] status=SUCCESS" in combined
    failed_val = "[WARPSHIFT_VALIDATION] status=FAILED" in combined

    bench_ms: Optional[float] = None
    if "[WARPSHIFT_BENCHMARK] time_ms=" in combined:
        try:
            bench_ms = float(combined.split("[WARPSHIFT_BENCHMARK] time_ms=")[1].split()[0])
        except Exception:
            pass

    if "[WARPSHIFT_GPU] rocm_version=" in combined:
        try:
            result.rocm_version = combined.split("[WARPSHIFT_GPU] rocm_version=")[1].split()[0]
        except Exception:
            pass

    result.bench_ms = bench_ms if bench_ms is not None else wall_ms
    result.success = (rc == 0) and result.validated and not failed_val

    if not result.success and not result.error:
        if failed_val:
            result.error = "Numerical validation FAILED — results mismatch on GPU"
        elif rc != 0:
            result.error = f"Binary exited {rc}: {result.run_stderr[:200] or result.run_stdout[:200]}"

    # ── Step 7: cleanup remote ───────────────────────────────────────────────
    _ssh_cmd(f"rm -rf {remote_dir}", timeout=10)

    return result
