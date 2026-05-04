# -*- coding: utf-8 -*-
"""
deploy_to_gpu.py -- Hot-patch the MI300X backend via HTTP file upload.

Run from your LOCAL dev machine:
    cd "d:/Projetos 2.0/Migrate AI"
    python backend/scripts/deploy_to_gpu.py

Steps:
1. Health-check MI300X
2. Try to upload each .py via /admin/patch-file (POST multipart)
   - If endpoint missing (old code): falls back to /admin/git-pull
   - If that also missing: prints manual instructions
3. Verify ROCm toolchain via /admin/rocm-info
4. Run a live /analyze call and confirm runtime_source != "mock"
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

MI300X_URL  = os.getenv("WARPSHIFT_GPU_URL", "http://134.199.200.190:8000")

# Script is at backend/scripts/deploy_to_gpu.py
# BACKEND_DIR is backend/ (one level up from scripts/)
_SCRIPT_DIR  = Path(__file__).resolve().parent
BACKEND_DIR  = _SCRIPT_DIR.parent   # .../backend

FILES_TO_PATCH = [
    "app/settings.py",
    "app/stages.py",
    "app/pipeline.py",
    "app/main.py",
    "app/ssh_executor.py",
    "app/schemas.py",
    "app/real_anchor.py",
]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(path):
    try:
        with urllib.request.urlopen(f"{MI300X_URL}{path}", timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def _post_json(path, body):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{MI300X_URL}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode(errors="replace")[:300]
        return {"error": f"HTTP {e.code}: {body_txt}"}
    except Exception as e:
        return {"error": str(e)}


def _upload_file(rel_path, content):
    """Upload via multipart/form-data to POST /admin/patch-file?path=<rel>"""
    boundary = b"----WarpShiftBoundary7MA4YWxkTrZu0gW"
    body = (
        b"--" + boundary + b"\r\n"
        + b'Content-Disposition: form-data; name="file"; filename="'
        + rel_path.encode() + b'"\r\n'
        + b"Content-Type: text/x-python\r\n\r\n"
        + content
        + b"\r\n--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(
        f"{MI300X_URL}/admin/patch-file?path={rel_path}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"}
    except Exception as e:
        return {"error": str(e)}


# ── Main deploy ───────────────────────────────────────────────────────────────

def main():
    print(f"\n[DEPLOY] WarpShift GPU Deploy -> {MI300X_URL}")
    print("-" * 60)

    # ── 1. Health check ────────────────────────────────────────────────────
    print("1. Health check...")
    h = _get("/health")
    if "error" in h:
        print(f"   [ERR] MI300X unreachable: {h['error']}")
        sys.exit(1)
    print(f"   [OK] Online | mode={h.get('backend_mode','?')} hipcc={h.get('hipcc','?')}")

    # ── 2. ROCm info (new endpoint — exists only on new code) ──────────────
    info = _get("/admin/rocm-info")
    new_code_already = "error" not in info
    if new_code_already:
        print(f"   [OK] New code detected on MI300X")
        print(f"        hipcc  : {info.get('hipcc_available')} - {info.get('hipcc_version','')[:60]}")
        print(f"        hipify : {info.get('hipify_available')} - {info.get('hipify_version','')[:60]}")
        print(f"        MI300X : {info.get('mi300x_detected')}")
    else:
        print("   [INFO] Old code on MI300X (/admin/rocm-info not found) -- uploading patch...")

    # ── 3. Upload files ────────────────────────────────────────────────────
    if not new_code_already:
        print("\n2. Uploading patched Python files...")
        patch_failed = False
        for rel in FILES_TO_PATCH:
            local = BACKEND_DIR / rel
            if not local.exists():
                print(f"   [SKIP] {rel} (not found locally at {local})")
                continue
            content = local.read_bytes()
            result  = _upload_file(rel, content)
            if "error" in result:
                err = str(result["error"])
                print(f"   [ERR]  {rel}: {err[:120]}")
                if "404" in err or "405" in err or "Not Found" in err:
                    patch_failed = True
                    break
            else:
                print(f"   [OK]   {rel} ({result.get('bytes',0)} bytes)")

        if patch_failed:
            # Try git pull fallback
            print("\n   [INFO] /admin/patch-file not available. Trying /admin/git-pull ...")
            pull = _post_json("/admin/git-pull", {})
            if pull.get("success"):
                print(f"   [OK]  git pull: {pull.get('stdout','').strip()[:200]}")
            else:
                err = pull.get("error") or pull.get("stderr") or str(pull)
                print(f"   [ERR] git pull failed: {err[:200]}")
                print_manual_instructions()
                return

        time.sleep(3)  # wait for uvicorn --reload

    # ── 4. Verify ROCm after patch ─────────────────────────────────────────
    print("\n3. Verifying ROCm environment on MI300X...")
    info2 = _get("/admin/rocm-info")
    if "error" not in info2:
        print(f"   hipcc    : {info2.get('hipcc_available')} -- {info2.get('hipcc_version','')[:70]}")
        print(f"   hipify   : {info2.get('hipify_available')} -- {info2.get('hipify_version','')[:70]}")
        print(f"   MI300X   : {info2.get('mi300x_detected')}")
        smi = info2.get("rocm_smi", "")
        if smi:
            print(f"   rocm-smi : {smi[:100]}")
    else:
        print(f"   [WARN] /admin/rocm-info still not available: {info2['error'][:100]}")

    # ── 5. Live analysis test ──────────────────────────────────────────────
    print("\n4. Running live analysis (SAXPY on GPU)...")
    result = _post_json("/analyze", {
        "github_url": "https://github.com/NVIDIA/cuda-samples",
        "mode": "live",
    })
    if "error" in result:
        print(f"   [ERR] {result['error'][:200]}")
    else:
        rs    = result.get("runtime_source", "?")
        score = result.get("migration_score", "?")
        conf  = result.get("migration_confidence", "?")
        bm    = result.get("benchmark", {})
        print(f"   runtime_source : {rs}")
        print(f"   score          : {score}/100")
        print(f"   confidence     : {conf}%")
        print(f"   cuda_ms        : {bm.get('cuda_baseline_ms','?')}")
        print(f"   rocm_ms        : {bm.get('rocm_live_ms','?')}")
        print(f"   delta_%        : {bm.get('performance_delta_percent','?')}")
        print()
        if rs == "mock":
            print("   [WARN] runtime_source is still 'mock'")
            print("         hipcc may not be in PATH or the MI300X has old code.")
            print("         Run: POST http://134.199.200.190:8000/admin/rocm-info to debug.")
        else:
            print(f"   [SUCCESS] REAL GPU EXECUTION: runtime_source={rs}")

    print(f"\n{'-'*60}\nDeploy complete.\n")


def print_manual_instructions():
    print("""
+--------------------------------------------------------------+
|  MANUAL DEPLOY: paste this in the AMD Developer Cloud        |
|  web terminal for your MI300X instance:                      |
+--------------------------------------------------------------+
|                                                              |
|  cd /opt/warpshift && git pull origin master                 |
|  pkill -f "uvicorn app.main" || true                         |
|  sleep 2                                                     |
|  cd backend                                                  |
|  nohup uvicorn app.main:app                                  |
|    --host 0.0.0.0 --port 8000 --reload                       |
|    > /tmp/warpshift.log 2>&1 &                               |
|  echo "Started PID $!"                                       |
|                                                              |
+--------------------------------------------------------------+
""")


if __name__ == "__main__":
    main()
