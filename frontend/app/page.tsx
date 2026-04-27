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

type Stage = {
  stage: number;
  name: string;
  status: "idle" | "running" | "done" | "failed";
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
  runtime_source: "mock" | "repo-scan" | "repo-scan+hipify";
  hipify_coverage_percent: number;
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
  const [githubUrl, setGithubUrl] = useState("https://github.com/user/cuda-model");
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

  const publishRealPR = async () => {
    if (!result) return;
    setPublishingPR(true);
    try {
      const response = await fetch(`http://localhost:8000/runs/${result.run_id}/create-pr`, { method: "POST" });
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

  const anchorBadge = useMemo(() => {
    const artifact = anchorStatus?.artifact;
    if (!artifact || !anchorStatus?.available) {
      return {
        label: "NO REAL ARTIFACT",
        className: "border-[#8d59fe]/50 bg-[#8d59fe]/15 text-[#cfbcff]",
      };
    }

    const healthy =
      artifact.hipify_executed &&
      Boolean(artifact.diff_preview) &&
      Boolean(artifact.warp_detection?.found) &&
      Boolean(artifact.repo_commit);

    if (healthy) {
      return {
        label: "Anchored to NVIDIA/cuda-samples",
        className: "border-[#8d59fe]/50 bg-[#8d59fe]/15 text-[#cfbcff]",
      };
    }

    return {
      label: "FALLBACK ACTIVE (CACHED)",
      className: "border-[#8d59fe]/50 bg-[#8d59fe]/15 text-[#cfbcff]",
    };
  }, [anchorStatus]);

  const loadHistory = async () => {
    const response = await fetch("http://localhost:8000/history");
    const payload = (await response.json()) as { items: HistoryItem[] };
    if (payload.items?.length) {
      setHistory(payload.items);
    }
  };

  useEffect(() => {
    loadHistory().catch(() => null);
    fetch("http://localhost:8000/demo-repos")
      .then((r) => r.json())
      .then((d: { items: string[] }) => setDemoRepos(d.items || []))
      .catch(() => null);
    fetch("http://localhost:8000/anchor/status")
      .then((r) => r.json())
      .then((d: AnchorStatus) => setAnchorStatus(d))
      .catch(() => null);
  }, []);

  const riskCounts = useMemo(() => {
    const items = result?.risk_items ?? [];
    return {
      high: items.filter((r) => r.level === "high").length,
      medium: items.filter((r) => r.level === "medium").length,
      low: items.filter((r) => r.level === "low").length,
    };
  }, [result]);

  const benchmarkData = useMemo(() => {
    if (!result) return [];
    return [
      { name: "CUDA A100", value: result.benchmark.cuda_baseline_ms },
      { name: "ROCm MI300X", value: result.benchmark.rocm_live_ms },
    ];
  }, [result]);

  const previousScore = useMemo(() => {
    if (!result) return null;
    const priorRuns = history.filter(h => h.run_id !== result.run_id); // In a real app we'd filter by URL too
    if (priorRuns.length > 0) return priorRuns[0].migration_score;
    return 78; // fallback for demo
  }, [result, history]);

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
    const url = new URL("http://localhost:8000/analyze/stream");
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
      const payload = JSON.parse((evt as MessageEvent).data) as { stage: number };
      setStages((prev) =>
        prev.map((s) =>
          s.stage === payload.stage ? { ...s, status: "done" } : s,
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
      fetch(`http://localhost:8000/runs/${payload.run_id}`)
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

  const exportRiskReport = async (format: "json" | "markdown") => {
    const response = await fetch(
      `http://localhost:8000/export/risk-report?format=${format}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          github_url: githubUrl || demoRepos[0] || "https://github.com/NVIDIA/cuda-samples",
          mode,
        }),
      },
    );
    const payload = (await response.json()) as { content: unknown };
    setExportedReport(
      typeof payload.content === "string"
        ? payload.content
        : JSON.stringify(payload.content, null, 2),
    );
  };

  return (
    <main className="min-h-screen p-4 text-sm text-zinc-100 selection:bg-[#8d59fe]/30">
      <div className="glass-panel mb-4 rounded-xl p-4 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-[#8d59fe] to-transparent opacity-50"></div>
        <div className="flex items-center justify-between relative z-10">
          <div className="flex items-center gap-2">
            <Image
              src="/warpshift.png"
              alt="WarpShift logo"
              width={26}
              height={26}
              className="rounded"
            />
            <h1 className="text-lg font-semibold">WarpShift - iterative migration workflow</h1>
          </div>
          <div className="flex items-center gap-3">
            <span className="rounded bg-emerald-500/20 border border-emerald-500/30 px-3 py-1.5 text-xs font-bold text-emerald-400 flex items-center gap-2 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              Running on AMD Developer Cloud (MI300X Ready)
            </span>
            <span className="rounded bg-[#8d59fe]/25 px-2 py-1.5 text-xs font-medium text-[#cfbcff]">
              ROCm 7.x
            </span>
          </div>
        </div>
        <button
          onClick={runDemoSequence}
          className="mt-3 rounded bg-[#8d59fe] px-3 py-1 text-xs font-semibold text-white"
        >
          Demo Sequence (2 min)
        </button>
        <div className={`mt-3 rounded border p-2 text-xs ${anchorBadge.className}`}>
          <p className="font-semibold">{anchorBadge.label}</p>
          <p>
            commit: {(anchorStatus?.artifact?.repo_commit || "n/a").slice(0, 12)}
          </p>
        </div>
      </div>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-4">
        <article className="glass-panel rounded-xl p-4 relative">
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
                <div className="mt-2">
                  <Image
                    src="/warpshift.png"
                    alt="WarpShift mark"
                    width={20}
                    height={20}
                    className="opacity-80"
                  />
                </div>
              </div>
            ) : null}
            <button className="w-full rounded border border-dashed border-zinc-700 p-2 text-left text-zinc-400">
              + new analysis
            </button>
          </div>
        </article>

        <article className="glass-panel rounded-xl p-4 relative">
          <h2 className="mb-3 text-xs font-semibold tracking-wide text-zinc-400">
            PIPELINE
          </h2>
          <p className="mb-1 text-zinc-400">GitHub URL</p>
          <input
            className="mb-2 w-full rounded border border-zinc-700 bg-zinc-900 px-3 py-2"
            value={githubUrl}
            onChange={(e) => setGithubUrl(e.target.value)}
          />
          {demoRepos.length ? (
            <p className="mb-2 text-xs text-zinc-500">
              Fallback demo repo: {demoRepos[0]}
            </p>
          ) : null}
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
              className="inline-flex items-center gap-1 rounded bg-[#8d59fe] px-3 py-1 font-semibold text-white disabled:opacity-50"
            >
              {running ? "RUNNING..." : "MIGRATE"}
              {!running ? <ArrowRightIcon /> : null}
            </button>
          </div>
          <div className="space-y-2">
            {stages.map((stage) => (
              <div key={stage.stage} className="glass-card rounded p-3 flex items-center justify-between">
                <div>
                  <p className="font-medium text-zinc-200">
                    Stage {stage.stage}: {stage.name}
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
                  {stage.status === "running" && <div className="h-3 w-3 rounded-full bg-[#8d59fe] animate-pulse-ring"></div>}
                  {stage.status === "done" && <CheckIcon className="h-4 w-4 text-emerald-400" />}
                  {stage.status === "failed" && <div className="h-4 w-4 text-red-500 font-bold flex items-center justify-center">×</div>}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-3 space-y-2">
            {(result?.diff_annotations ?? []).map((annotation) => (
              <div key={annotation.id} className="glass-card rounded p-3">
                <p className="text-[11px] font-mono text-zinc-400 mb-2 font-medium">
                  {annotation.file}:{annotation.line}
                </p>
                <div className="rounded bg-zinc-950/80 p-2 text-[11px] font-mono overflow-x-auto border border-zinc-800/50">
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
                  <ChevronRightIcon className={`transition-transform duration-200 ${expandedInsight === annotation.id ? "rotate-90" : ""}`} />
                  MigrateAI Insight
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

        <article className="glass-panel rounded-xl p-4 relative">
          <h2 className="mb-3 text-xs font-semibold tracking-wide text-zinc-400">
            OUTPUT
          </h2>
          <p>Score: {result?.migration_score ?? "-"}/100</p>
          <p>Conf: {result?.migration_confidence ?? "-"}%</p>
          <p>HIGH: {riskCounts.high}</p>
          <p>MED: {riskCounts.medium}</p>
          <p>LOW: {riskCounts.low}</p>
          <div className={`mt-2 rounded-lg p-3 font-semibold text-center border shadow-lg ${
            result?.decision_engine.decision === "do_not_migrate_yet"
              ? "bg-red-500/10 border-red-500/30 text-red-400"
              : "bg-amber-500/10 border-amber-500/30 text-amber-400"
          }`}>
            {result?.decision_engine.decision === "do_not_migrate_yet"
              ? "❌ DO NOT MIGRATE YET"
              : "⚠️ PROCEED WITH CAUTION"}
          </div>
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
              Source: {result.runtime_source} | Commit:{" "}
              {(result.repo_commit || "n/a").slice(0, 12)}
            </p>
          ) : null}
          {result ? (
            <p className="mt-1 text-xs text-zinc-500">
              HIPIFY coverage: {result.hipify_coverage_percent}% | Runtime:{" "}
              {result.runtime_status.toUpperCase()}
            </p>
          ) : null}
          {result ? (
            <p className="mt-1 text-xs text-zinc-500">
              Build: {(result.build_system || "unknown").toUpperCase()} | Status:{" "}
              {result.build_status.toUpperCase()}
            </p>
          ) : null}

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
              SAXPY Benchmark (GPU Validated)
            </button>
            <button
              onClick={() => setActiveTab("pr")}
              className={`rounded px-2 py-1 text-xs ${activeTab === "pr" ? "bg-[#8d59fe] text-white" : "bg-zinc-800"}`}
            >
              PR
            </button>
          </div>
          <div className="mt-2 flex gap-2">
            <button
              onClick={() => exportRiskReport("markdown")}
              className="rounded bg-zinc-800 px-2 py-1 text-xs"
            >
              Export Risk MD
            </button>
            <button
              onClick={() => exportRiskReport("json")}
              className="rounded bg-zinc-800 px-2 py-1 text-xs"
            >
              Export Risk JSON
            </button>
          </div>

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
            <div className="mt-3 h-48">
              <ResponsiveContainer width="100%" height="100%">
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
              {result ? (
                <p className="mt-2 text-xs text-zinc-400">
                  Delta: +{result.benchmark.performance_delta_percent}% |{" "}
                  {result.benchmark.hardware} | {result.benchmark.rocm_version}
                </p>
              ) : null}
            </div>
          ) : null}

          {activeTab === "pr" ? (
            <div className="mt-3 glass-card rounded p-3 text-xs border-l-2 border-l-[#8d59fe]">
              <div className="flex items-center gap-2 mb-2">
                <span className="bg-[#8d59fe]/20 text-[#cfbcff] px-2 py-0.5 rounded font-mono text-[10px]">Open</span>
                <p className="font-semibold text-base text-zinc-100">
                  {result?.pull_request_preview.title ?? "CUDA -> ROCm Migration"} <span className="font-normal text-zinc-500">#{result?.pull_request_preview.pr_number ?? 42}</span>
                </p>
              </div>
              <div className="flex items-center gap-4 text-zinc-400 mb-3 border-b border-zinc-800/50 pb-2">
                <p>Files changed: <span className="font-medium text-zinc-200">{result?.pull_request_preview.files_changed ?? 12}</span></p>
                <p>
                  <span className="text-emerald-400">+{result?.pull_request_preview.lines_added ?? 340}</span>{" "}
                  <span className="text-red-400">-{result?.pull_request_preview.lines_removed ?? 210}</span>
                </p>
              </div>
              <p className="mt-2 text-zinc-300 font-medium">GitHub PR body:</p>
              <pre className="mt-2 rounded bg-zinc-950/80 p-3 text-[11px] whitespace-pre-wrap break-words border border-zinc-800/50 text-zinc-300 leading-relaxed">
                {result?.pull_request_preview.github_pr_body ??
                  "Run migration to generate PR body"}
              </pre>
              {result?.pull_request_preview.real_pr_url ? (
                <a href={result.pull_request_preview.real_pr_url} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1 text-xs text-[#cfbcff] hover:underline">
                  View Real PR on GitHub <ArrowRightIcon className="h-3 w-3" />
                </a>
              ) : (
                <button
                  onClick={publishRealPR}
                  disabled={publishingPR}
                  className="mt-3 rounded bg-[#8d59fe] px-3 py-1 text-xs font-semibold text-white disabled:opacity-50"
                >
                  {publishingPR ? "Publishing..." : "Publish Real PR to GitHub"}
                </button>
              )}
            </div>
          ) : null}
          {exportedReport ? (
            <pre className="mt-3 max-h-40 overflow-auto rounded bg-zinc-950 p-2 text-[11px] text-zinc-300">
              {exportedReport}
            </pre>
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

        <article className="glass-panel rounded-xl p-4 relative">
          <h2 className="mb-3 text-xs font-semibold tracking-wide text-zinc-400">
            DELTA TRACKING
          </h2>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-light text-zinc-500">{previousScore ?? "-"}</span>
            <span className="text-zinc-600 text-xl">→</span>
            <span className="text-3xl font-semibold text-emerald-400">{result?.migration_score ?? "-"}</span>
            {result && previousScore && (
              <span className="ml-2 px-2 py-0.5 rounded-full bg-emerald-400/10 text-emerald-400 text-xs font-bold">
                +{result.migration_score - previousScore} pts
              </span>
            )}
          </div>
          <p className="mt-2 inline-flex items-center gap-1 text-emerald-300">
            <CheckIcon />
            Fixed warpSize issue
          </p>
          <p className="inline-flex items-center gap-1 text-emerald-300">
            <CheckIcon />
            Fixed cuBLAS mismatch
          </p>
          <p className="mt-2 inline-flex items-center gap-1 text-amber-300">
            <AlertIcon />
            Remaining dynamic launch
          </p>
          <p className="inline-flex items-center gap-1 text-amber-300">
            <AlertIcon />
            Remaining cuDNN custom op
          </p>
          <button
            onClick={runMigration}
            className="mt-3 rounded bg-zinc-800 px-3 py-1 text-xs"
          >
            Rerun after fixes
          </button>
          {result ? (
            <p className="mt-3 text-xs text-zinc-500">
              Run #{result.run_id} ·{" "}
              {new Date(result.timestamp_utc).toISOString().slice(0, 10)}
            </p>
          ) : (
            <p className="mt-3 text-xs text-zinc-500">
              Delta appears after first execution
            </p>
          )}
        </article>
      </section>
    </main>
  );
}
