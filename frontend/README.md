# WarpShift Frontend (Spec v6.0)

Interactive dashboard for CUDA -> ROCm migration workflow.

## Implemented Panels

1. Recent Analyses (history and quick project load)
2. Pipeline (4 stages + SSE progress + live/full mode)
3. Output (score/confidence/decision + tabs for risk report, benchmark, PR)
4. Delta Tracking (score progression + fixed/remaining issues + rerun)

## UX Features

- Real-time stage streaming via SSE (`/analyze/stream`)
- Inline code-aware diff annotation with collapsible `WarpShift Insight`
- Benchmark chart with Recharts
- Persistent history loaded from backend (`/history`)
- Run ID + timestamp display

## Run

```powershell
cd "D:\Projetos 2.0\Migrate AI\frontend"
npm install
npm run dev
```

Open `http://localhost:3000`.
