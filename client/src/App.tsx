import { useRef, useState } from "react";

import { useStream, type Badge } from "./hooks/useStream";

const BADGE_STYLES: Record<Badge, { label: string; cls: string }> = {
  offline: { label: "OFFLINE", cls: "text-zinc-500" },
  idle: { label: "IDLE", cls: "text-sky-400" },
  generating: { label: "GEN", cls: "text-amber-400 animate-pulse" },
  live: { label: "LIVE", cls: "text-emerald-400" },
};

function StatChip({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2">
      <div className="text-[0.6rem] font-semibold uppercase tracking-[0.14em] text-zinc-500">{label}</div>
      <div className="font-mono text-sm text-zinc-200">{value ?? "—"}</div>
    </div>
  );
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const { session, badge, status, log, stats, pendingJobs, connect, generate } = useStream(videoRef);

  const [refImage, setRefImage] = useState("");
  const [prompt, setPrompt] = useState("");
  const [seamlessEnding, setSeamlessEnding] = useState(true);

  const connected = session !== null;
  const b = BADGE_STYLES[badge];

  return (
    <main className="relative mx-auto flex min-h-screen max-w-5xl flex-col items-center px-4 py-10">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-32 top-0 h-96 w-96 rounded-full bg-emerald-500/10 blur-3xl" />
        <div className="absolute -right-24 bottom-0 h-80 w-80 rounded-full bg-sky-500/10 blur-3xl" />
      </div>

      <header className="relative mb-8 text-center">
        <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/80 px-3 py-1 text-xs font-medium text-zinc-400 backdrop-blur">
          <span className={`h-1.5 w-1.5 rounded-full ${connected ? "bg-emerald-400" : "bg-zinc-600"}`} />
          Live fMP4 stream · seamless loop cuts
        </div>
        <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">Wan Live Stream</h1>
      </header>

      <div className="relative flex w-full flex-col items-start gap-8 sm:flex-row sm:justify-center">
        {/* --- player --- */}
        <div
          className={`relative aspect-[240/416] w-[240px] shrink-0 overflow-hidden rounded-2xl border-2 bg-zinc-900 shadow-2xl shadow-black/40 transition-all duration-300 ${
            badge === "live" ? "border-emerald-500 shadow-glow" : "border-zinc-800"
          }`}
        >
          <video ref={videoRef} muted playsInline autoPlay className="h-full w-full object-cover" />
          <div
            className={`absolute left-3 top-3 rounded-full border border-white/10 bg-black/55 px-2.5 py-1 text-[0.65rem] font-bold uppercase tracking-[0.14em] backdrop-blur-md ${b.cls}`}
          >
            {b.label}
            {pendingJobs > 1 ? ` ·${pendingJobs}` : ""}
          </div>
        </div>

        {/* --- controls --- */}
        <div className="flex w-full max-w-sm flex-col gap-4">
          {!connected ? (
            <form
              className="flex flex-col gap-3"
              onSubmit={(e) => {
                e.preventDefault();
                void connect(refImage.trim());
              }}
            >
              <label className="text-xs font-medium uppercase tracking-wider text-zinc-500">
                Reference image (server path)
              </label>
              <input
                value={refImage}
                onChange={(e) => setRefImage(e.target.value)}
                placeholder="examples/test.png"
                className="rounded-xl border border-zinc-800 bg-zinc-900/80 px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-600 outline-none transition focus:border-emerald-600"
              />
              <button
                type="submit"
                className="rounded-xl bg-emerald-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-emerald-500 active:scale-[0.98]"
              >
                Connect
              </button>
            </form>
          ) : (
            <form
              className="flex flex-col gap-3"
              onSubmit={(e) => {
                e.preventDefault();
                if (prompt.trim()) void generate(prompt.trim(), refImage.trim(), seamlessEnding);
              }}
            >
              <label className="text-xs font-medium uppercase tracking-wider text-zinc-500">Prompt</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={3}
                placeholder="Describe what should happen next…"
                className="resize-none rounded-xl border border-zinc-800 bg-zinc-900/80 px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-600 outline-none transition focus:border-emerald-600"
              />
              <label className="flex cursor-pointer items-center gap-2 text-xs text-zinc-400">
                <input
                  type="checkbox"
                  checked={seamlessEnding}
                  onChange={(e) => setSeamlessEnding(e.target.checked)}
                  className="h-3.5 w-3.5 accent-emerald-500"
                />
                End on reference frame (seamless return to idle)
              </label>
              <button
                type="submit"
                disabled={!prompt.trim()}
                className="rounded-xl bg-emerald-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-emerald-500 active:scale-[0.98] disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-500"
              >
                Generate
              </button>
            </form>
          )}

          <p className="text-sm text-zinc-400">{status}</p>

          <div className="grid grid-cols-3 gap-2">
            <StatChip label="generation" value={stats.lastGenS !== null ? `${stats.lastGenS.toFixed(1)}s` : null} />
            <StatChip
              label="push → ws"
              value={stats.lastPushToWsMs !== null ? `${stats.lastPushToWsMs.toFixed(0)}ms` : null}
            />
            <StatChip
              label="boundary wait"
              value={stats.lastBoundaryWaitS !== null ? `${stats.lastBoundaryWaitS.toFixed(1)}s` : null}
            />
          </div>

          <div className="flex min-h-24 flex-col gap-1 rounded-xl border border-zinc-800/80 bg-zinc-900/40 p-3">
            {log.length === 0 ? (
              <span className="font-mono text-xs text-zinc-600">event log</span>
            ) : (
              log.map((row, i) => (
                <div
                  key={`${row.at}-${i}`}
                  className={`flex gap-2.5 font-mono text-xs ${row.highlight ? "text-emerald-400" : "text-zinc-500"}`}
                >
                  <span className="shrink-0 tabular-nums">{row.at}</span>
                  <span>{row.text}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
