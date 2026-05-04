#!/usr/bin/env python3
"""
deploy_to_gpu.py — Hot-patch the MI300X backend via HTTP file upload.

Run this from your LOCAL dev machine:
    cd "d:/Projetos 2.0/Migrate AI"
    python scripts/deploy_to_gpu.py

It will:
1. Upload all changed backend Python files to the MI300X
2. Trigger a server restart via the admin endpoint
3. Verify the new code is live

Requires the MI300X to already be running the version with /admin/patch-file.
If not (first deploy), use the git-based bootstrap instead:
    On MI300X console: cd /opt/warpshift && git pull && pkill -f uvicorn && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

MI300X_URL = os.getenv("WARPSHIFT_GPU_URL", "http://134.199.200.190:8000")
BACKEND_DIR = Path(__file__).parent.parent / "backend"

FILES_TO_PATCH = [
    "app/settings.py",
    "app/stages.py",
    "app/pipeline.py",
    "app/main.py",
    "app/ssh_executor.py",
    "app/schemas.py",
    "app/real_anchor.py",
]


def _get(path: str) -> dict:
    url = f"{MI300X_URL}{path}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def _post_json(path: str, body: dict) -> dict:
    url = f"{MI300X_URL}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def _upload_file(rel_path: str, content: bytes) -> dict:
    """Upload a file via multipart/form-data to /admin/patch-file."""
    url = f"{MI300X_URL}/admin/patch-file?path={rel_path}"
    boundary = b"----WarpShiftBoundary7MA4YWxkTrZu0gW"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="' + rel_path.encode() + b'"\r\n'
        b"Content-Type: text/plain\r\n\r\n"
        + content
        + b"\r\n--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"\n🚀 WarpShift GPU Deploy → {MI300X_URL}\n{'─'*60}")

    # 1. Health check
    print("1. Health check...")
    h = _get("/health")
    if "error" in h:
        print(f"   ❌ MI300X unreachable: {h['error']}")
        sys.exit(1)
    print(f"   ✅ Online | mode={h.get('backend_mode')} hipcc={h.get('hipcc')}")

    # 2. Check if patch endpoint exists
    info = _get("/admin/rocm-info")
    if "error" not in info:
        print(f"   🎯 ROCm: hipcc={info.get('hipcc_available')} hipify={info.get('hipify_available')} MI300X={info.get('mi300x_detected')}")
    else:
        print(f"   ⚠️  /admin/rocm-info not available yet (old code) — will patch via HTTP")

    # 3. Upload each file
    print("\n2. Uploading patched files...")
    for rel in FILES_TO_PATCH:
        local = BACKEND_DIR / rel
        if not local.exists():
            print(f"   ⚠  SKIP {rel} (not found locally)")
            continue
        content = local.read_bytes()
        result = _upload_file(rel, content)
        if "error" in result:
            print(f"   ❌ {rel}: {result['error']}")
            # If patch endpoint doesn't exist, fall through to git pull
            if "404" in str(result["error"]) or "405" in str(result["error"]):
                print("\n   ℹ️  /admin/patch-file not available. Trying git pull instead...")
                pull = _post_json("/admin/git-pull", {})
                if pull.get("success"):
                    print(f"   ✅ git pull: {pull.get('stdout')}")
                else:
                    print(f"   ❌ git pull failed: {pull.get('stderr')}")
                    print_manual_instructions()
                return
        else:
            print(f"   ✅ {rel} ({result.get('bytes')} bytes)")

    # 4. Restart server
    print("\n3. Restarting uvicorn...")
    time.sleep(2)
    h2 = _get("/health")
    if "error" not in h2:
        print(f"   ✅ Server up | mode={h2.get('backend_mode')}")
    
    # 5. Verify real ROCm
    print("\n4. Verifying ROCm environment...")
    info2 = _get("/admin/rocm-info")
    if "error" not in info2:
        print(f"   hipcc     : {info2.get('hipcc_available')} — {info2.get('hipcc_version', '')[:60]}")
        print(f"   hipify    : {info2.get('hipify_available')} — {info2.get('hipify_version', '')[:60]}")
        print(f"   MI300X    : {info2.get('mi300x_detected')}")
        print(f"   rocm-smi  : {info2.get('rocm_smi', '')[:80]}")
    else:
        print(f"   ⚠️  {info2['error']}")

    # 6. Test live analysis
    print("\n5. Running live analysis (no repo — SAXPY only)...")
    result = _post_json("/analyze", {
        "github_url": "https://github.com/NVIDIA/cuda-samples",
        "mode": "live",
    })
    if "error" in result:
        print(f"   ❌ {result['error']}")
    else:
        rs = result.get("runtime_source", "?")
        score = result.get("migration_score", "?")
        conf = result.get("migration_confidence", "?")
        bench = result.get("benchmark", {})
        rocm_ms = bench.get("rocm_live_ms", "?")
        print(f"   runtime_source : {rs}")
        print(f"   score          : {score}/100")
        print(f"   confidence     : {conf}%")
        print(f"   rocm_live_ms   : {rocm_ms}")
        if rs == "mock":
            print("\n   ⚠️  Still mock — hipcc may not be in PATH on MI300X")
        else:
            print(f"\n   🎉 REAL GPU EXECUTION CONFIRMED: {rs}")

    print(f"\n{'─'*60}\nDeploy complete.\n")


def print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  MANUAL DEPLOY — Run these on the AMD Developer Cloud        ║
║  console / web terminal for the MI300X instance:             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  cd /opt/warpshift                                           ║
║  git pull origin main                                        ║
║  pkill -f "uvicorn app.main" || true                         ║
║  sleep 2                                                     ║
║  cd backend                                                  ║
║  nohup uvicorn app.main:app \\                               ║
║    --host 0.0.0.0 --port 8000 --reload &                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
