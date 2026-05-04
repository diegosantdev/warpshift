# Build In Public Posts

## Post 1
warpSize hardcoded as 32 is the #1 silent CUDA -> ROCm bug. AMD CDNA uses wavefront 64.  
MigrateAI detects and annotates this automatically before migration starts.  
@AIatAMD @lablab #AMDDevHackathon #ROCm #CUDA

## Post 2
Live result on migration benchmark: CUDA 120ms -> ROCm 135ms on MI300X (+12.5%).  
Expected for first pass with zero kernel tuning.  
Point is reliability + explainability, then optimization.  
@AIatAMD @lablab #AMDDevHackathon

## Post 3
Top 3 things HIPIFY alone will not solve:
1) cuDNN custom ops parity
2) warpSize assumptions
3) dynamic launch incompatibilities

MigrateAI detects all 3 with line-level annotations and next-step guidance.
