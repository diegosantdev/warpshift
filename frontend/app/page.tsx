"use client";

import Image from "next/image";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Label,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  AreaChart,
  Area,
} from "recharts";

function ChevronRightIcon({ className = "h-3 w-3" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <path
        d="M7 5L12 10L7 15"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ArrowRightIcon({ className = "h-3.5 w-3.5" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <path
        d="M3 10H16M16 10L11 5M16 10L11 15"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function CheckIcon({ className = "h-3.5 w-3.5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" className={className} aria-hidden="true">
      <path d="M4 10.5L8 14.5L16 6.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function AlertIcon({ className = "h-3.5 w-3.5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" className={className} aria-hidden="true">
      <path d="M10 3L17 16H3L10 3Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M10 7.5V11" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="10" cy="13.5" r="0.9" fill="currentColor" />
    </svg>
  );
}

function XIcon({ className = "h-3.5 w-3.5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" className={className} aria-hidden="true">
      <path d="M5 5L15 15M15 5L5 15" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

type Stage = {
  stage: number;
  name: string;
  status: "idle" | "running" | "done" | "failed";
  duration_s?: number;
  gpu_ms?: number;
  detail?: string;
  log?: any;
};

type RiskItem = {
  id: string;
  level: "high" | "medium" | "low";
  title: string;
  description: string;
  detection_source: string;
  line?: number;
};

type HistoryItem = {
  run_id: string;
  timestamp_utc: string;
  github_url: string;
  migration_score: number;
  migration_confidence: number;
  decision: "proceed_with_caution" | "do_not_migrate_yet";
};

type AnalysisResult = {
  run_id: string;
  timestamp_utc: string;
  migration_score: number;
  migration_confidence: number;
  decision_engine: { decision: "proceed_with_caution" | "do_not_migrate_yet" };
  benchmark: {
    cuda_baseline_ms: number;
    rocm_live_ms: number;
    performance_delta_percent: number;
    hardware: string;
    rocm_version: string;
  };
  risk_items: RiskItem[];
  diff_annotations: Array<{
    id: string;
    file: string;
    line: number;
    original: string;
    converted: string;
    detection_source: string;
    confidence: "high" | "medium" | "low";
    effort: string;
    insight: {
      summary: string;
      impact: string[];
      fix_applied: string;
      manual_review: string;
    };
  }>;
  pull_request_preview: {
    pr_number: number;
    title: string;
    files_changed: number;
    lines_added: number;
    lines_removed: number;
    auto_converted: string[];
    flagged_for_review: string[];
    manual_fix_required: string[];
    github_pr_body: string;
    real_pr_url?: string | null;
  };
  runtime_source: string;
  hipify_coverage_percent: number;
  has_converted_code?: boolean;
  runtime_status: "pass" | "fail";
  build_system?: string | null;
  build_status: "not_run" | "pass" | "fail";
  evidence_file?: string | null;
  repo_commit?: string | null;
};

type AnchorStatus = {
  available: boolean;
  mode: string;
  artifact: null | {
    repo_url: string;
    repo_ref: string;
    repo_commit?: string;
    hipify_executed: boolean;
    source_relative_path: string;
    diff_preview: string;
    warp_detection?: {
      found: boolean;
      line?: number;
      content?: string;
    };
  };
};

const defaultStages: Stage[] = [
  { stage: 1, name: "HIPIFY Conversion", status: "idle" },
  { stage: 2, name: "Static Analysis", status: "idle" },
  { stage: 3, name: "Runtime Validation", status: "idle" },
  { stage: 4, name: "AI Explanation Layer", status: "idle" },
];

export default function Home() {
  const [githubUrl, setGithubUrl] = useState("");
  const [mode, setMode] = useState<"live" | "full">("live");
  const [stages, setStages] = useState<Stage[]>(defaultStages);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [expandedInsight, setExpandedInsight] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"risk" | "benchmark" | "pr">("risk");
  const [demoRepos, setDemoRepos] = useState<string[]>([]);
  const [exportedReport, setExportedReport] = useState<string>("");
  const [anchorStatus, setAnchorStatus] = useState<AnchorStatus | null>(null);
  const [runEvidence, setRunEvidence] = useState<string>("");
  const [showRawJson, setShowRawJson] = useState(false);
  const [running, setRunning] = useState(false);
  const [publishingPR, setPublishingPR] = useState(false);
  const [downloadingPatch, setDownloadingPatch] = useState(false);

  const publishRealPR = async () => {
    if (!result) return;
    setPublishingPR(true);
    try {
      const response = await fetch(`/api/runs/${result.run_id}/create-pr`, { method: "POST" });
      const data = await response.json();
      if (data.status === "ok") {
        setResult((prev) => prev ? {
          ...prev, 
          pull_request_preview: { ...prev.pull_request_preview, real_pr_url: data.pr_url }
        } : null);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setPublishingPR(false);
    }
  };

  const downloadFile = (content: string, filename: string, mime: string) => {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const loadHistory = async () => {
    const response = await fetch("/api/history");
    const payload = (await response.json()) as { items: HistoryItem[] };
    if (payload.items?.length) {
      setHistory(payload.items);
    }
  };

  useEffect(() => {
    loadHistory().catch(() => null);
    fetch("/api/demo-repos")
      .then((r) => r.json())
      .then((d: { items: string[] }) => setDemoRepos(d.items || []))
      .catch(() => null);
  }, []);

  const riskCounts = useMemo(() => {
    const items = result?.risk_items ?? [];
    const flaggedCount = result?.pull_request_preview?.flagged_for_review?.length || 0;
    return {
      high: items.filter((r) => r.level === "high").length,
      medium: items.filter((r) => r.level === "medium").length + flaggedCount,
      low: items.filter((r) => r.level === "low").length,
    };
  }, [result]);

  const benchmarkData = useMemo(() => {
    if (!result) return [];
    return [
      { name: "CUDA V100", value: result.benchmark.cuda_baseline_ms },
      { name: "ROCm MI300X", value: result.benchmark.rocm_live_ms },
    ];
  }, [result]);

  const previousScore = useMemo(() => {
    if (!result) return null;
    const priorRuns = history.filter(h => h.run_id !== result.run_id);
    if (priorRuns.length > 0) return priorRuns[0].migration_score;
    return null;
  }, [result, history]);

  const deltaChartData = useMemo(() => {
    const sorted = [...history].sort((a, b) => new Date(a.timestamp_utc).getTime() - new Date(b.timestamp_utc).getTime());
    const hasCurrent = result ? sorted.some(h => h.run_id === result.run_id) : true;
    
    let combined = sorted;
    if (result && !hasCurrent) {
       combined = [...sorted, {
          run_id: result.run_id,
          timestamp_utc: result.timestamp_utc,
          migration_score: result.migration_score
       } as any];
    }
    
    let data = combined.slice(-6).map((item, i) => ({
      name: `R${i + 1}`,
      score: item.migration_score,
    }));
    
    if (data.length === 1) {
      data = [{ name: 'Start', score: 0 }, data[0]];
    }
    
    return data;
  }, [history, result]);

  const runMigration = () => {
    setRunning(true);
    setRuntimeError(null);
    setResult(null);
    setExpandedInsight(null);
    setStages(defaultStages);

    const fallbackRepo = demoRepos[0] ?? "https://github.com/NVIDIA/cuda-samples";
    const effectiveUrl =
      githubUrl.startsWith("http://") || githubUrl.startsWith("https://")
        ? githubUrl
        : fallbackRepo;
    const url = new URL("/api/analyze/stream", window.location.origin);
    url.searchParams.set("github_url", effectiveUrl);
    url.searchParams.set("mode", mode);

    const stream = new EventSource(url.toString());

    stream.addEventListener("stage_start", (evt) => {
      const payload = JSON.parse((evt as MessageEvent).data) as { stage: number };
      setStages((prev) =>
        prev.map((s) =>
          s.stage === payload.stage ? { ...s, status: "running" } : s,
        ),
      );
    });

    stream.addEventListener("stage_update", (evt) => {
      const payload = JSON.parse((evt as MessageEvent).data) as { stage: number, status?: string, duration_s?: number, gpu_ms?: number, detail?: string, log?: any };
      setStages((prev) =>
        prev.map((s) =>
          s.stage === payload.stage ? { ...s, status: (payload.status as any) || "done", duration_s: payload.duration_s, gpu_ms: payload.gpu_ms, detail: payload.detail, log: payload.log } : s,
        ),
      );
    });

    stream.addEventListener("runtime_error", (evt) => {
      const payload = JSON.parse((evt as MessageEvent).data) as { error: string };
      setRuntimeError(payload.error);
      setStages((prev) =>
        prev.map((s) => (s.stage === 3 ? { ...s, status: "failed" } : s)),
      );
    });

    stream.addEventListener("completed", (evt) => {
      const payload = JSON.parse((evt as MessageEvent).data) as AnalysisResult;
      setResult(payload);
      setRunning(false);
      loadHistory().catch(() => null);
      fetch(`/api/runs/${payload.run_id}`)
        .then((r) => r.json())
        .then((d: { evidence: unknown }) =>
          setRunEvidence(JSON.stringify(d.evidence, null, 2)),
        )
        .catch(() => setRunEvidence(""));
      stream.close();
    });

    stream.onerror = () => {
      stream.close();
      setRunning(false);
    };
  };

  const runDemoSequence = () => {
    setMode("live");
    runMigration();
  };

  const exportRiskReport = (format: "json" | "markdown") => {
    if (!result) return;
    if (format === "markdown") {
      const lines = [
        "# WarpShift Pre-Migration Risk Report",
        `- Run: ${result.run_id}`,
        `- Repo: ${githubUrl}`,
        `- Score: ${result.migration_score}/100`,
        `- Confidence: ${result.migration_confidence}%`,
        `- Hardware: ${result.benchmark.hardware}`,
        `- rocm_live_ms: ${result.benchmark.rocm_live_ms}ms`,
        `- Speedup vs V100: ${(result.benchmark.cuda_baseline_ms / result.benchmark.rocm_live_ms).toFixed(1)}x`,
        "",
        "## Detected Risks",
      ];
      for (const risk of result.risk_items || []) {
        lines.push(`- [${risk.level.toUpperCase()}] ${risk.title}`);
        lines.push(`  - Source: ${risk.detection_source}`);
        lines.push(`  - Line: ${risk.line ?? "N/A"}`);
        lines.push(`  - Description: ${risk.description}`);
      }
      lines.push("");
      lines.push("## Decision");
      lines.push(`- ${result.decision_engine?.decision ?? "N/A"}`);
      downloadFile(lines.join("\n"), `warpshift-risk-${result.run_id}.md`, "text/markdown");
    } else {
      downloadFile(JSON.stringify(result, null, 2), `warpshift-run-${result.run_id}.json`, "application/json");
    }
  };

  const downloadPatch = () => {
    if (!result || !result.diff_annotations?.length) return;
    setDownloadingPatch(true);
    const lines = [
      `--- CUDA Source`,
      `+++ HIP/ROCm Converted`,
      `# WarpShift Migration Patch`,
      `# Run: ${result.run_id} | Score: ${result.migration_score}/100`,
      `# Hardware: ${result.benchmark.hardware}`,
      "",
    ];
    for (const ann of result.diff_annotations) {
      lines.push(`@@ ${ann.file}:${ann.line} [${ann.confidence.toUpperCase()}] @@`);
      lines.push(`-${ann.original}`);
      lines.push(`+${ann.converted}`);
      lines.push("");
    }
    downloadFile(lines.join("\n"), `warpshift-patch-${result.run_id}.patch`, "text/plain");
    setDownloadingPatch(false);
  };

  return (
    <main className="min-h-screen p-4 text-sm text-zinc-100 selection:bg-[#8d59fe]/30">
      <div className="glass-panel animate-fade-in mb-4 rounded-xl p-4 relative overflow-hidden">
        <div className="flex items-center justify-between relative z-10">
          <div className="flex items-center gap-2">
            <Image
              src="/warpshift.png"
              alt="WarpShift logo"
              width={26}
              height={26}
              className="rounded"
            />
            <h1 className="text-lg font-semibold">WarpShift AI Agent</h1>
          </div>
          <div className="flex items-center gap-3">
            <span className="rounded bg-emerald-500/20 border border-emerald-500/30 px-3 py-1.5 text-xs font-bold text-emerald-400 flex items-center gap-2 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
              MI300X Live · hipcc+gpu
            </span>
            <span className="rounded bg-[#8d59fe]/25 px-2 py-1.5 text-xs font-medium text-[#cfbcff]">
              ROCm 7.x
            </span>
          </div>
        </div>
      </div>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-4">
        <article className="glass-panel animate-slide-up rounded-xl transition-all duration-500 p-4 relative overflow-hidden">
          <h2 className="mb-3 text-xs font-semibold tracking-wide text-zinc-400">
            RECENT ANALYSES
          </h2>
          <div className="space-y-2">
            {history.slice(0, 3).map((analysis, idx) => (
              <button
                key={`${analysis.run_id}-${analysis.timestamp_utc}-${idx}`}
                onClick={() => setGithubUrl(analysis.github_url)}
                className="w-full rounded bg-zinc-900 p-2 text-left"
              >
                <p className="font-medium">{analysis.github_url.split("/").pop()}</p>
                <p className="text-zinc-400">Score: {analysis.migration_score}</p>
                <p className="text-xs text-zinc-500">{analysis.run_id}</p>
              </button>
            ))}
            {history.length === 0 ? (
              <div className="rounded border border-zinc-800 bg-zinc-900/50 p-2 text-xs text-zinc-500">
                No real runs yet. Execute MIGRATE to populate history.
              </div>
            ) : null}
          </div>
        </article>

        <article className="glass-panel animate-slide-up rounded-xl transition-all duration-500 p-4 relative overflow-hidden">
          <h2 className="mb-3 text-xs font-semibold tracking-wide text-zinc-400">
            PIPELINE
          </h2>
          <p className="mb-1 text-zinc-400">GitHub Repository URL</p>
          <input
            className="mb-2 w-full rounded border border-zinc-700 bg-zinc-900 px-3 py-2 placeholder:text-zinc-600"
            value={githubUrl}
            onChange={(e) => setGithubUrl(e.target.value)}
            placeholder="https://github.com/org/cuda-repo"
          />
          <div className="mb-2 flex gap-2">
            <button
              onClick={() => setMode("live")}
              className={`rounded px-2 py-1 ${mode === "live" ? "bg-[#8d59fe] text-white" : "bg-zinc-800"}`}
            >
              Live (&lt;90s)
            </button>
            <button
              onClick={() => setMode("full")}
              className={`rounded px-2 py-1 ${mode === "full" ? "bg-[#8d59fe] text-white" : "bg-zinc-800"}`}
            >
              Full (preprocessed)
            </button>
            <button
              onClick={runMigration}
              disabled={running}
              className="inline-flex items-center gap-1 rounded bg-[#8d59fe] px-3 py-1 font-semibold text-white disabled:opacity-50 transition-all whitespace-nowrap"
            >
              {running ? "ANALYZING..." : "MIGRATE"}
              {!running ? <ArrowRightIcon /> : null}
            </button>
          </div>
          <div className="space-y-2">
            {stages.map((stage) => (
              <div key={stage.stage} className="glass-card rounded p-3 flex flex-col">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-zinc-200 flex flex-wrap items-center gap-2">
                      Stage {stage.stage}: {stage.name}
                      {stage.status === "done" && stage.duration_s !== undefined && (
                        <span className="text-[10px] font-mono text-zinc-500 font-normal">
                          ({stage.duration_s < 0.1 ? "<0.1" : stage.duration_s.toFixed(1)}s)
                        </span>
                      )}
                      {stage.status === "done" && stage.gpu_ms !== undefined && (
                        <span className="text-[10px] font-mono text-emerald-500 font-bold bg-emerald-500/10 px-1.5 py-0.5 rounded flex items-center gap-1">
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                          {stage.gpu_ms}ms ← MI300X
                        </span>
                      )}
                    </p>
                    <p className={`text-xs mt-1 font-semibold uppercase tracking-wider ${
                      stage.status === "running" ? "text-[#cfbcff]" :
                      stage.status === "done" ? "text-emerald-400" :
                      stage.status === "failed" ? "text-red-400" : "text-zinc-500"
                    }`}>
                      {stage.status}
                    </p>
                  </div>
                  <div className="flex items-center justify-center">
                    {stage.status === "idle" && <div className="h-2 w-2 rounded-full bg-zinc-600"></div>}
                    {stage.status === "running" && (
                      <svg className="animate-spin h-4 w-4 text-[#8d59fe]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    )}
                    {stage.status === "done" && <CheckIcon className="h-4 w-4 text-emerald-400" />}
                    {stage.status === "failed" && <div className="h-4 w-4 text-red-500 font-bold flex items-center justify-center">×</div>}
                  </div>
                </div>
                
                {stage.status !== "idle" && (stage.detail || stage.log?.stdout) && (
                  <div className="mt-3 bg-zinc-950/80 rounded p-2 text-[10px] font-mono text-zinc-400 border border-zinc-800/50 shadow-inner">
                    {stage.detail && <div className="text-[#cfbcff] mb-1 flex items-start gap-1">
                      {stage.detail}
                    </div>}
                    {stage.log?.stdout && (
                      <div className="opacity-70 whitespace-pre-wrap break-words min-w-0 max-w-full leading-relaxed pl-3 border-l border-zinc-800 overflow-hidden">
                        {stage.log.stdout.trim().split('\n').filter(Boolean).slice(-3).join('\n')}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="mt-3 space-y-2">
            {(result?.diff_annotations ?? []).map((annotation) => (
              <div key={annotation.id} className="glass-card rounded p-3">
                <p className="text-xs font-mono text-zinc-400 mb-2 font-medium break-all">
                  {annotation.file}:{annotation.line}
                </p>
                <div className="rounded bg-zinc-950/80 p-2 text-[13px] font-mono overflow-x-auto border border-zinc-800/50">
                  <div className="diff-line diff-remove">- {annotation.original}</div>
                  <div className="diff-line diff-add">+ {annotation.converted}</div>
                </div>
                <button
                  className="mt-3 inline-flex items-center gap-1 text-[11px] font-semibold tracking-wide text-[#cfbcff] uppercase hover:text-white transition-colors"
                  onClick={() =>
                    setExpandedInsight((prev) =>
                      prev === annotation.id ? null : annotation.id,
                    )
                  }
                >
                  <ChevronRightIcon className={`h-3 w-3 transition-transform duration-200 ${expandedInsight === annotation.id ? "rotate-90" : ""}`} />
                  WarpShift Insight
                </button>
                {expandedInsight === annotation.id ? (
                  <div className="mt-2 rounded bg-[#8d59fe]/10 border border-[#8d59fe]/20 p-3 text-xs animate-slide-down">
                    <p className="font-semibold text-white">{annotation.insight.summary}</p>
                    <p className="mt-2 text-[#cfbcff] font-medium text-[11px] uppercase tracking-wider">Impact</p>
                    <ul className="mt-1 space-y-1">
                      {annotation.insight.impact.map((item) => (
                        <li key={item} className="text-zinc-300 flex items-start gap-1.5">
                           <span className="text-[#8d59fe] mt-0.5">•</span>
                           <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                    <p className="mt-3 font-medium text-white">
                      <span className="text-emerald-400 mr-1 font-semibold">Fix applied:</span> {annotation.insight.fix_applied}
                    </p>
                    <div className="mt-3 pt-2 border-t border-[#8d59fe]/20 flex flex-wrap gap-x-4 gap-y-2 text-[11px] text-zinc-400">
                      <p><span className="text-zinc-500">Source:</span> {annotation.detection_source}</p>
                      <p><span className="text-zinc-500">Confidence:</span> <span className={annotation.confidence === 'high' ? 'text-emerald-400' : 'text-amber-400'}>{annotation.confidence.toUpperCase()}</span></p>
                      <p><span className="text-zinc-500">Effort:</span> {annotation.effort}</p>
                    </div>
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </article>

        <article className="glass-panel animate-slide-up rounded-xl transition-all duration-500 p-4 relative overflow-hidden">
          <h2 className="mb-3 text-xs font-semibold tracking-wide text-zinc-400">
            OUTPUT
          </h2>
          <p>Score: {result?.migration_score ?? "-"}/100</p>
          <p>Conf: {result?.migration_confidence ?? "-"}%</p>
          <p>HIGH: {riskCounts.high}</p>
          <p>MED: {riskCounts.medium}</p>
          <p>LOW: {riskCounts.low}</p>
          {result ? (
            <div className={`mt-2 rounded-lg p-3 font-semibold text-center border shadow-lg ${
              result.decision_engine.decision === "do_not_migrate_yet"
                ? "bg-red-500/10 border-red-500/30 text-red-400"
                : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
            }`}>
              {result.decision_engine.decision === "do_not_migrate_yet" ? (
                <span className="flex items-center justify-center gap-1.5"><XIcon className="h-4 w-4" /> AGENT DECISION: DO NOT MIGRATE</span>
              ) : (
                <span className="flex items-center justify-center gap-1.5"><CheckIcon className="h-4 w-4" /> AGENT DECISION: PROCEED WITH MIGRATION</span>
              )}
            </div>
          ) : (
            <div className="mt-2 rounded-lg p-3 font-semibold text-center border border-dashed border-zinc-800 text-zinc-600">
              Awaiting Analysis
            </div>
          )}
          {runtimeError ? (
            <p className="mt-2 rounded bg-[#8d59fe]/20 p-2 text-[#cfbcff]">
              Build failed: {runtimeError}
            </p>
          ) : null}
          {result ? (
            <p className="mt-2 text-zinc-400">
              Run #{result.run_id} ·{" "}
              {new Date(result.timestamp_utc).toLocaleString("en-GB", {
                hour12: false,
              })}
            </p>
          ) : null}
          {result ? (
            <p className="mt-1 text-xs text-zinc-500">
              Source: <span className={result.runtime_source.includes("gpu") ? "text-emerald-400 font-semibold" : "text-zinc-400"}>{result.runtime_source}</span> | Commit:{" "}
              {(result.repo_commit || "n/a").slice(0, 12)}
            </p>
          ) : null}
          {result ? (
            <p className="mt-1 text-xs text-zinc-500">
              HIPIFY coverage: {result.hipify_coverage_percent}% | Runtime:{" "}
              <span className={result.runtime_status === "pass" ? "text-emerald-400 font-semibold" : "text-zinc-400"}>{result.runtime_status.toUpperCase()}</span>
            </p>
          ) : null}
          {result ? (
            <p className="mt-1 text-xs text-zinc-500">
              Build: {(result.build_system || "unknown").toUpperCase()} | Status:{" "}
              {result.build_status.toUpperCase()}
            </p>
          ) : null}

          {/* ── Download Converted Code ── */}
          {result?.has_converted_code && (
            <a
              href={`/api/runs/${result.run_id}/download`}
              target="_blank"
              rel="noreferrer"
              className="mt-3 flex items-center justify-center gap-2 rounded-lg bg-[#8d59fe] hover:bg-[#7a48ef] transition-colors px-4 py-2.5 text-sm font-bold text-white"
            >
              ↓ Download Converted Code (.zip)
            </a>
          )}

          <div className="mt-3 flex gap-2">
            <button
              onClick={() => setActiveTab("risk")}
              className={`rounded px-2 py-1 text-xs ${activeTab === "risk" ? "bg-[#8d59fe] text-white" : "bg-zinc-800"}`}
            >
              Risk Report
            </button>
            <button
              onClick={() => setActiveTab("benchmark")}
              className={`rounded px-2 py-1 text-xs ${activeTab === "benchmark" ? "bg-[#8d59fe] text-white" : "bg-zinc-800"}`}
            >
              GPU Benchmark
            </button>
            <button
              onClick={() => setActiveTab("pr")}
              className={`rounded px-2 py-1 text-xs ${activeTab === "pr" ? "bg-[#8d59fe] text-white" : "bg-zinc-800"}`}
            >
              PR Preview
            </button>
          </div>
          {result && (
            <div className="mt-2 flex flex-wrap gap-2">
              <button
                onClick={() => exportRiskReport("markdown")}
                className="rounded bg-zinc-800 hover:bg-zinc-700 transition-colors px-2 py-1 text-xs flex items-center gap-1"
              >
                ↓ Risk Report .md
              </button>
              <button
                onClick={() => exportRiskReport("json")}
                className="rounded bg-zinc-800 hover:bg-zinc-700 transition-colors px-2 py-1 text-xs flex items-center gap-1"
              >
                ↓ Full Report .json
              </button>
              {result.diff_annotations?.length > 0 && (
                <button
                  onClick={downloadPatch}
                  disabled={downloadingPatch}
                  className="rounded bg-[#8d59fe]/20 hover:bg-[#8d59fe]/30 transition-colors border border-[#8d59fe]/30 px-2 py-1 text-xs text-[#cfbcff] flex items-center gap-1 disabled:opacity-50"
                >
                  ↓ Download Patch .patch
                </button>
              )}
            </div>
          )}

          {activeTab === "risk" ? (
            <div className="mt-3 space-y-2 text-xs">
              {(result?.risk_items ?? []).map((risk) => (
                <div key={risk.id} className="rounded bg-zinc-900 p-2">
                  <p className="font-medium">{risk.title}</p>
                  <p className="text-zinc-400">{risk.description}</p>
                  <p className="text-zinc-500">Source: {risk.detection_source}</p>
                </div>
              ))}
            </div>
          ) : null}

          {activeTab === "benchmark" ? (
            <div className="mt-3">
              {result && result.benchmark.performance_delta_percent > 0 && (
                <div className="flex items-center justify-end mb-2">
                  <div className="bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-lg text-emerald-400 flex items-center gap-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                    <span className="font-bold text-xl">{(result.benchmark.performance_delta_percent / 100 + 1).toFixed(1)}x</span>
                    <span className="text-xs font-semibold tracking-wide uppercase opacity-80">Faster on MI300X</span>
                  </div>
                </div>
              )}
              <div className="h-48 w-full">
                <ResponsiveContainer width="100%" height="100%" minWidth={10} minHeight={10}>
                  <BarChart data={benchmarkData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
                    <XAxis dataKey="name" stroke="#a1a1aa" />
                    <YAxis stroke="#a1a1aa">
                      <Label value="ms/iter" angle={-90} position="insideLeft" fill="#a1a1aa" />
                    </YAxis>
                    <Tooltip />
                    <Bar dataKey="value" fill="#8d59fe" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              {result ? (
                <p className="mt-2 text-xs text-zinc-400">
                  Delta: {result.benchmark.performance_delta_percent}% |{" "}
                  Speedup: {(result.benchmark.cuda_baseline_ms / result.benchmark.rocm_live_ms).toFixed(1)}x vs V100 |{" "}
                  {result.benchmark.hardware} | {result.benchmark.rocm_version}
                </p>
              ) : null}
            </div>
          ) : null}

          {activeTab === "pr" ? (
            <div className="mt-3 glass-card rounded p-3 text-xs border-l-2 border-l-[#8d59fe]">
              {result ? (
                <>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="bg-[#8d59fe]/20 text-[#cfbcff] px-2 py-0.5 rounded font-mono text-[10px]">Open</span>
                    <p className="font-semibold text-base text-zinc-100">
                      {result.pull_request_preview.title} <span className="font-normal text-zinc-500">#{result.pull_request_preview.pr_number}</span>
                    </p>
                  </div>
                  <div className="flex items-center gap-4 text-zinc-400 mb-3 border-b border-zinc-800/50 pb-2">
                    <p>Files changed: <span className="font-medium text-zinc-200">{result.pull_request_preview.files_changed}</span></p>
                    <p>
                      <span className="text-emerald-400">+{result.pull_request_preview.lines_added}</span>{" "}
                      <span className="text-red-400">-{result.pull_request_preview.lines_removed}</span>
                    </p>
                  </div>
                  
                  {result.pull_request_preview.flagged_for_review?.length > 0 && (
                    <div className="mt-4 rounded bg-red-500/10 border border-red-500/20 p-3">
                      <h3 className="text-xs font-bold text-red-400 mb-2 uppercase tracking-wide flex items-center gap-2">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                        Flagged For Review
                      </h3>
                      <ul className="list-disc list-inside text-xs text-red-300/80 space-y-1">
                        {result.pull_request_preview.flagged_for_review.map((flag, idx) => (
                          <li key={idx} className="font-mono">{flag}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <p className="mt-4 text-zinc-300 font-medium">GitHub PR body:</p>
                  <pre className="mt-2 rounded bg-zinc-950/80 p-3 text-[11px] whitespace-pre-wrap break-words border border-zinc-800/50 text-zinc-300 leading-relaxed">
                    {result.pull_request_preview.github_pr_body}
                  </pre>
                </>
              ) : (
                <div className="p-4 text-center text-zinc-500">
                  <p>Run migration to generate Pull Request preview.</p>
                </div>
              )}
            </div>
          ) : null}

          {runEvidence ? (
            <div className="mt-3">
              <button
                onClick={() => setShowRawJson((prev) => !prev)}
                className="inline-flex items-center gap-1 text-xs text-zinc-300"
              >
                <ChevronRightIcon className={`h-3 w-3 transition-transform ${showRawJson ? "rotate-90" : ""}`} />
                View raw JSON
              </button>
              {showRawJson ? (
                <pre className="mt-2 max-h-40 overflow-auto rounded bg-zinc-950 p-2 text-[11px] text-zinc-300 whitespace-pre-wrap break-words">
                  {runEvidence}
                </pre>
              ) : null}
            </div>
          ) : null}
        </article>

        <article className="glass-panel animate-slide-up rounded-xl transition-all duration-500 p-4 relative overflow-hidden flex flex-col justify-between">
          <div>
            <h2 className="mb-3 text-xs font-semibold tracking-wide text-zinc-400">
              DELTA TRACKING
            </h2>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-3xl font-light text-zinc-500">{previousScore ?? "-"}</span>
              <ArrowRightIcon className="h-5 w-5 text-zinc-600" />
              <span className="text-3xl font-semibold text-emerald-400">{result?.migration_score ?? "-"}</span>
              {result && previousScore && (
                <span className={`ml-2 px-2 py-0.5 rounded-full text-xs font-bold ${result.migration_score >= previousScore ? 'bg-emerald-400/10 text-emerald-400' : 'bg-red-400/10 text-red-400'}`}>
                  {result.migration_score >= previousScore ? "+" : ""}{result.migration_score - previousScore} pts
                </span>
              )}
            </div>
            
            <div className="mt-5 h-32 w-full">
              {deltaChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%" minWidth={10} minHeight={10}>
                  <AreaChart data={deltaChartData} margin={{ top: 5, right: 0, left: -25, bottom: 0 }}>
                    <defs>
                      <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis dataKey="name" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                    <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} domain={['dataMin - 5', 100]} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '8px', fontSize: '11px', color: '#a1a1aa' }}
                      itemStyle={{ color: '#10b981', fontWeight: 600 }}
                      labelStyle={{ color: '#d4d4d8', marginBottom: '4px' }}
                    />
                    <Area type="monotone" dataKey="score" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#scoreGradient)" activeDot={{ r: 4, fill: '#10b981', stroke: '#18181b', strokeWidth: 2 }} />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-zinc-800 bg-zinc-900/30 text-xs text-zinc-600">
                   Run migrations to build history
                </div>
              )}
            </div>
          </div>
          
          <div className="mt-auto pt-4">
            {result && (
              <button
                onClick={runMigration}
                className="w-full rounded bg-zinc-800/80 hover:bg-zinc-700 transition-colors px-3 py-2 text-xs font-medium text-zinc-300 border border-zinc-700/50"
              >
                Rerun after fixes
              </button>
            )}
            {result ? (
              <p className="mt-3 text-[10px] text-zinc-500 text-center uppercase tracking-wider">
                Run #{result.run_id.slice(0, 8)}
              </p>
            ) : (
              <p className="mt-3 text-[10px] text-zinc-500 text-center uppercase tracking-wider">
                Awaiting executions
              </p>
            )}
          </div>
        </article>
      </section>
    </main>
  );
}
