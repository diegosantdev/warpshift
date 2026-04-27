from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["high", "medium", "low"]
Decision = Literal["proceed_with_caution", "do_not_migrate_yet"]


class AnalysisRequest(BaseModel):
    github_url: str = Field(..., description="Repository URL to analyze")
    mode: Literal["live", "full"] = "live"


class RiskItem(BaseModel):
    id: str
    level: RiskLevel
    title: str
    description: str
    detection_source: str
    line: int | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    effort: str = "medium ~30min"
    fix: str | None = None
    blocking: bool = False


class Insight(BaseModel):
    risk_id: str
    summary: str = ""
    impact: list[str]
    fix_applied: str
    manual_review: str


class DiffAnnotation(BaseModel):
    id: str
    file: str
    line: int
    original: str
    converted: str
    detection_source: str
    confidence: Literal["high", "medium", "low"]
    effort: str
    insight: Insight


class BenchmarkResult(BaseModel):
    cuda_baseline_ms: float
    rocm_live_ms: float
    performance_delta_percent: float
    hardware: str = "AMD Instinct MI300X"
    rocm_version: str = "ROCm 7.x"


class DecisionEngineResult(BaseModel):
    decision: Decision
    why: list[str]
    unresolved_consequences: list[str]
    next_step: str


class PullRequestPreview(BaseModel):
    pr_number: int
    title: str
    files_changed: int
    lines_added: int
    lines_removed: int
    auto_converted: list[str]
    flagged_for_review: list[str]
    manual_fix_required: list[str]
    github_pr_body: str
    real_pr_url: str | None = None


class AnalysisResult(BaseModel):
    run_id: str
    timestamp_utc: datetime
    migration_score: int
    migration_confidence: int
    estimated_effort: str
    risk_items: list[RiskItem]
    insights: list[Insight]
    benchmark: BenchmarkResult
    decision_engine: DecisionEngineResult
    diff_annotations: list[DiffAnnotation]
    pull_request_preview: PullRequestPreview
    runtime_source: Literal["mock", "repo-scan", "repo-scan+hipify"]
    build_system: str | None = None
    build_status: Literal["not_run", "pass", "fail"] = "not_run"
    evidence_file: str | None = None
    repo_commit: str | None = None
    hipify_coverage_percent: int = 65
    runtime_status: Literal["pass", "fail"] = "fail"
