from __future__ import annotations

import os


class Settings:
    # backend mode: "mock" is safe for local demo, "real" tries tools on host.
    backend_mode: str = os.getenv("MIGRATEAI_BACKEND_MODE", "mock")
    hipify_bin: str = os.getenv("MIGRATEAI_HIPIFY_BIN", "hipify-clang")
    hipcc_bin: str = os.getenv("MIGRATEAI_HIPCC_BIN", "hipcc")
    vllm_url: str = os.getenv("MIGRATEAI_VLLM_URL", "https://api.groq.com/openai/v1/chat/completions")
    vllm_model: str = os.getenv("MIGRATEAI_VLLM_MODEL", "llama3-8b-8192")
    llm_api_key: str = os.getenv("MIGRATEAI_LLM_API_KEY", "")
    vllm_timeout_seconds: int = int(os.getenv("MIGRATEAI_VLLM_TIMEOUT_SECONDS", "15"))
    stage_delay_seconds: float = float(os.getenv("MIGRATEAI_STAGE_DELAY_SECONDS", "0.7"))
    anchor_repo_url: str = os.getenv("MIGRATEAI_ANCHOR_REPO_URL", "https://github.com/NVIDIA/cuda-samples")
    anchor_repo_ref: str = os.getenv("MIGRATEAI_ANCHOR_REPO_REF", "master")
    anchor_relative_path: str = os.getenv(
        "MIGRATEAI_ANCHOR_RELATIVE_PATH",
        "Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu",
    )
    anchor_cache_dir: str = os.getenv(
        "MIGRATEAI_ANCHOR_CACHE_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "anchor-repo")),
    )
    anchor_artifact_file: str = os.getenv(
        "MIGRATEAI_ANCHOR_ARTIFACT_FILE",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "real_anchor_artifact.json")),
    )
    runtime_build_timeout_seconds: int = int(os.getenv("MIGRATEAI_RUNTIME_BUILD_TIMEOUT_SECONDS", "120"))
    github_real_pr: bool = os.getenv("GITHUB_REAL_PR", "false").lower() == "true"
    github_default_base_branch: str = os.getenv("GITHUB_DEFAULT_BASE_BRANCH", "main")

    # Docker sandbox executor settings
    execution_mode: str = os.getenv("WARPSHIFT_EXECUTION_MODE", "host")  # "host" | "docker"
    docker_image: str = os.getenv("WARPSHIFT_DOCKER_IMAGE", "warpshift-runner:latest")
    docker_memory_limit: str = os.getenv("WARPSHIFT_DOCKER_MEMORY", "2g")
    docker_cpu_limit: str = os.getenv("WARPSHIFT_DOCKER_CPUS", "1")
    docker_timeout_seconds: int = int(os.getenv("WARPSHIFT_DOCKER_TIMEOUT", "120"))


settings = Settings()
