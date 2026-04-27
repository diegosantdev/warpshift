# WarpShift Runner — Isolated sandbox for CUDA-to-ROCm migration pipeline
# Build: docker build -t warpshift-runner:latest .
# Run:   docker run --rm -v /tmp/run_X:/workspace warpshift-runner:latest

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Core toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang \
    perl \
    python3 \
    python3-pip \
    git \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create mock hipcc wrapper for syntax-only validation without GPU
RUN mkdir -p /usr/local/bin && \
    printf '#!/bin/bash\n\
# WarpShift mock hipcc — validates syntax without GPU hardware\n\
if [ "$1" = "--version" ]; then\n\
    echo "HIP version: 6.0.0 (warpshift-mock)"\n\
    echo "AMD clang version 17.0.0"\n\
    exit 0\n\
fi\n\
# Compile using clang as a syntax checker\n\
exec clang++ -std=c++17 -fsyntax-only -D__HIP_PLATFORM_AMD__ "$@"\n' > /usr/local/bin/hipcc && \
    chmod +x /usr/local/bin/hipcc

# Create mock hipify-perl wrapper
RUN printf '#!/bin/bash\n\
if [ "$1" = "--version" ]; then\n\
    echo "hipify-perl v1.0 (warpshift-mock)"\n\
    exit 0\n\
fi\n\
# Basic CUDA-to-HIP text substitution for demo\n\
sed -e "s/cudaMalloc/hipMalloc/g" \
    -e "s/cudaFree/hipFree/g" \
    -e "s/cudaMemcpy/hipMemcpy/g" \
    -e "s/cudaStream_t/hipStream_t/g" \
    -e "s/cudaEvent_t/hipEvent_t/g" \
    -e "s/cuda_runtime.h/hip\\/hip_runtime.h/g" \
    "$@"\n' > /usr/local/bin/hipify-perl && \
    chmod +x /usr/local/bin/hipify-perl

# Install Python dependencies
COPY backend/requirements.txt /opt/warpshift/requirements.txt
RUN pip3 install --no-cache-dir -r /opt/warpshift/requirements.txt

# Copy backend source
COPY backend/ /opt/warpshift/backend/

# Workspace layout
RUN mkdir -p /workspace/input_repo /workspace/converted /workspace/build /workspace/logs

ENV MIGRATEAI_BACKEND_MODE=real
ENV MIGRATEAI_HIPIFY_BIN=hipify-perl
ENV MIGRATEAI_HIPCC_BIN=hipcc

WORKDIR /opt/warpshift

# Entrypoint: run pipeline on the repo specified via env vars
COPY backend/scripts/docker_entrypoint.py /opt/warpshift/entrypoint.py
ENTRYPOINT ["python3", "/opt/warpshift/entrypoint.py"]
