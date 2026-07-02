import { useCallback, useEffect, useRef, useState } from "react";

import { createStream, generateClip, type StreamSession } from "../lib/api";
import { Fmp4Player, type PlayerEvent } from "../lib/player";

export type Badge = "offline" | "idle" | "generating" | "live";

export interface LogRow {
  at: string;
  text: string;
  highlight?: boolean;
}

export interface Stats {
  lastGenS: number | null;
  lastPushToWsMs: number | null;
  lastBoundaryWaitS: number | null;
}

function ts(): string {
  return new Date().toTimeString().slice(0, 8);
}

export function useStream(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const [session, setSession] = useState<StreamSession | null>(null);
  const [badge, setBadge] = useState<Badge>("offline");
  const [status, setStatus] = useState("Enter a reference image path and connect.");
  const [log, setLog] = useState<LogRow[]>([]);
  const [pendingJobs, setPendingJobs] = useState(0);
  const [stats, setStats] = useState<Stats>({ lastGenS: null, lastPushToWsMs: null, lastBoundaryWaitS: null });

  const playerRef = useRef<Fmp4Player | null>(null);
  const genStartRef = useRef<number | null>(null);

  const addLog = useCallback((text: string, highlight = false) => {
    setLog((rows) => [{ at: ts(), text, highlight }, ...rows].slice(0, 10));
  }, []);

  useEffect(() => () => playerRef.current?.destroy(), []);

  const handlePlayerEvent = useCallback(
    (e: PlayerEvent) => {
      switch (e.type) {
        case "ws-open":
          addLog("stream connected");
          break;
        case "ws-closed":
          addLog(`stream closed (${e.code})`);
          setBadge("offline");
          setStatus("Stream closed.");
          break;
        case "codec":
          break;
        case "real-received": {
          const genS = genStartRef.current ? (Date.now() - genStartRef.current) / 1000 : null;
          setStats((s) => ({ ...s, lastGenS: genS, lastPushToWsMs: e.pushToWsMs }));
          addLog(
            `clip received (${e.durationS.toFixed(1)}s${genS ? `, generated in ${genS.toFixed(1)}s` : ""}) — waiting for loop boundary`,
            true,
          );
          setStatus("Clip ready — starts at the next loop boundary.");
          break;
        }
        case "real-playing":
          setStats((s) => ({ ...s, lastBoundaryWaitS: e.waitedMs / 1000 }));
          setPendingJobs((n) => Math.max(0, n - 1));
          setBadge("live");
          addLog(`▶ new clip now playing (waited ${(e.waitedMs / 1000).toFixed(1)}s for the boundary)`, true);
          setStatus("Playing generated clip.");
          break;
        case "real-ended":
          setBadge((b) => (b === "live" ? "idle" : b));
          addLog("clip ended — idle loop");
          setStatus("Idle looping. Send another prompt.");
          break;
        case "error":
          addLog(`error: ${e.message}`);
          break;
      }
    },
    [addLog],
  );

  const connect = useCallback(
    async (refImage: string) => {
      if (!videoRef.current) return;
      setStatus("Creating stream…");
      try {
        const s = await createStream(refImage || null);
        setSession(s);
        setBadge("idle");
        setStatus(refImage ? "Connected — idle clips are generating…" : "Connected.");
        const player = new Fmp4Player(videoRef.current, handlePlayerEvent);
        playerRef.current = player;
        player.connect(s.ws_url);
      } catch (err) {
        setStatus(`Failed to create stream: ${err}`);
      }
    },
    [videoRef, handlePlayerEvent],
  );

  const generate = useCallback(
    async (prompt: string, refImage: string, seamlessEnding: boolean) => {
      if (!session) return;
      genStartRef.current = Date.now();
      setBadge("generating");
      setPendingJobs((n) => n + 1);
      addLog(`prompt submitted: "${prompt.slice(0, 48)}${prompt.length > 48 ? "…" : ""}"`);
      setStatus("Generating…");
      try {
        await generateClip(session, prompt, refImage || null, seamlessEnding);
      } catch (err) {
        setPendingJobs((n) => Math.max(0, n - 1));
        setBadge("idle");
        setStatus(`Generate failed: ${err}`);
      }
    },
    [session, addLog],
  );

  return { session, badge, status, log, stats, pendingJobs, connect, generate };
}
