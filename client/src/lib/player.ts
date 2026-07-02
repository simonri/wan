/**
 * Fmp4Player — MSE player for the Wan live-stream WebSocket protocol.
 *
 * Protocol (server → client):
 *   [JSON] {type:"init",   codec:"avc1.640015"}
 *   [BIN]  <ftyp+moov>
 *   [JSON] {type:"reinit", codec:"..."}            — on SPS change (rare)
 *   [BIN]  <ftyp+moov>
 *   [JSON] {type:"chunk",  pts_s, duration_s, pushed_at_ms?}  — idle or real
 *   [BIN]  <moof+mdat>
 *
 * Design: the stream is a continuous timeline of whole clips. Idle clips are
 * FLF loops (first frame == last frame == reference image); real clips start
 * from the same reference, so every clip boundary is a seamless cut. The
 * player therefore NEVER seeks — real content is simply appended after the
 * in-flight idle loop and playback flows into it at the boundary.
 */

export type PlayerEvent =
  | { type: "ws-open" }
  | { type: "ws-closed"; code: number }
  | { type: "codec"; codec: string }
  | { type: "real-received"; ptsS: number; durationS: number; pushToWsMs: number }
  | { type: "real-playing"; ptsS: number; waitedMs: number }
  | { type: "real-ended" }
  | { type: "error"; message: string };

interface RealRange {
  start: number;
  end: number;
  receivedAt: number;
  announced: boolean;
}

type Appendable = ArrayBuffer | { changeType: string };

export class Fmp4Player {
  private video: HTMLVideoElement;
  private onEvent: (e: PlayerEvent) => void;

  private ws: WebSocket | null = null;
  private ms: MediaSource | null = null;
  private sb: SourceBuffer | null = null;
  private sbReady = false;

  private pendingCtrl: Record<string, unknown> | null = null;
  private appendQueue: Appendable[] = [];

  private realRanges: RealRange[] = [];
  private playingReal = false;

  private trimTimer: number | null = null;

  constructor(video: HTMLVideoElement, onEvent: (e: PlayerEvent) => void) {
    this.video = video;
    this.onEvent = onEvent;
    this.video.addEventListener("timeupdate", this.onTimeUpdate);
  }

  connect(wsPath: string): void {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${scheme}://${location.host}${wsPath}`);
    this.ws.binaryType = "arraybuffer";
    this.ws.onopen = () => this.onEvent({ type: "ws-open" });
    this.ws.onclose = (e) => this.onEvent({ type: "ws-closed", code: e.code });
    this.ws.onerror = () => this.onEvent({ type: "error", message: "WebSocket error" });
    this.ws.onmessage = (e) => this.onMessage(e);

    // Trim already-played media so long sessions never hit the MSE quota.
    this.trimTimer = window.setInterval(() => this.trimBuffer(), 15_000);
  }

  destroy(): void {
    if (this.trimTimer !== null) window.clearInterval(this.trimTimer);
    this.video.removeEventListener("timeupdate", this.onTimeUpdate);
    this.ws?.close();
    this.ws = null;
    if (this.ms?.readyState === "open") {
      try {
        this.ms.endOfStream();
      } catch {
        /* already closed */
      }
    }
    if (this.video.src.startsWith("blob:")) {
      URL.revokeObjectURL(this.video.src);
      this.video.removeAttribute("src");
      this.video.load();
    }
  }

  // -- protocol ---------------------------------------------------------------

  private onMessage(e: MessageEvent): void {
    if (typeof e.data === "string") {
      this.pendingCtrl = JSON.parse(e.data);
      return;
    }
    const ctrl = this.pendingCtrl;
    this.pendingCtrl = null;
    if (!ctrl) return;
    const data = e.data as ArrayBuffer;
    if (ctrl.type === "init") this.handleInit(ctrl.codec as string, data);
    else if (ctrl.type === "reinit") this.handleReinit(ctrl.codec as string, data);
    else if (ctrl.type === "chunk") this.handleChunk(ctrl, data);
  }

  private handleInit(codec: string, initSegment: ArrayBuffer): void {
    const mime = `video/mp4; codecs="${codec}"`;
    if (!MediaSource.isTypeSupported(mime)) {
      this.onEvent({ type: "error", message: `codec not supported: ${mime}` });
      return;
    }
    this.onEvent({ type: "codec", codec });
    this.sbReady = false;
    this.appendQueue = [];

    this.ms = new MediaSource();
    this.video.src = URL.createObjectURL(this.ms);
    this.ms.addEventListener(
      "sourceopen",
      () => {
        if (!this.ms) return;
        this.sb = this.ms.addSourceBuffer(mime);
        this.sb.mode = "sequence";
        this.sb.addEventListener("updateend", () => this.drain());
        this.sb.addEventListener("error", () =>
          this.onEvent({ type: "error", message: `SourceBuffer error (video: ${this.video.error?.message ?? "?"})` }),
        );
        this.sbReady = true;
        this.enqueue(initSegment);
      },
      { once: true },
    );

    this.video.addEventListener(
      "canplay",
      () => {
        this.video.play().catch(() => this.onEvent({ type: "error", message: "autoplay blocked — click the video" }));
      },
      { once: true },
    );
  }

  private handleReinit(codec: string, initSegment: ArrayBuffer): void {
    const mime = `video/mp4; codecs="${codec}"`;
    if (this.sb?.changeType && MediaSource.isTypeSupported(mime)) {
      this.onEvent({ type: "codec", codec });
      this.appendQueue.push({ changeType: mime });
      this.enqueue(initSegment);
    } else {
      this.handleInit(codec, initSegment);
    }
  }

  private handleChunk(ctrl: Record<string, unknown>, media: ArrayBuffer): void {
    const ptsS = ctrl.pts_s as number;
    const durationS = (ctrl.duration_s as number) ?? 0;
    if (ctrl.pushed_at_ms != null) {
      const now = Date.now();
      this.realRanges.push({ start: ptsS, end: ptsS + durationS, receivedAt: now, announced: false });
      this.onEvent({
        type: "real-received",
        ptsS,
        durationS,
        pushToWsMs: now - (ctrl.pushed_at_ms as number),
      });
    }
    this.enqueue(media);
  }

  // -- MSE plumbing -----------------------------------------------------------

  private enqueue(data: ArrayBuffer): void {
    this.appendQueue.push(data);
    this.drain();
  }

  private drain(): void {
    if (!this.sbReady || !this.sb || this.sb.updating) return;
    const item = this.appendQueue.shift();
    if (item === undefined) return;
    if ("changeType" in (item as object)) {
      this.sb.changeType((item as { changeType: string }).changeType);
      this.drain();
      return;
    }
    try {
      this.sb.appendBuffer(item as ArrayBuffer);
    } catch (err) {
      if ((err as DOMException).name === "QuotaExceededError") {
        this.appendQueue.unshift(item);
        this.trimBuffer();
      } else {
        this.onEvent({ type: "error", message: `appendBuffer: ${(err as Error).message}` });
      }
    }
  }

  private trimBuffer(): void {
    if (!this.sb || this.sb.updating) return;
    const keepFrom = this.video.currentTime - 10;
    const buffered = this.video.buffered;
    if (buffered.length > 0 && keepFrom > buffered.start(0) + 5) {
      try {
        this.sb.remove(0, keepFrom);
      } catch {
        /* raced with an append; next tick retries */
      }
    }
  }

  // -- real-content tracking ----------------------------------------------------

  private onTimeUpdate = (): void => {
    const t = this.video.currentTime;
    // drop ranges we have fully played
    while (this.realRanges.length > 0 && t >= this.realRanges[0].end - 0.05) {
      this.realRanges.shift();
      if (this.realRanges.length === 0 && this.playingReal) {
        this.playingReal = false;
        this.onEvent({ type: "real-ended" });
      }
    }
    const current = this.realRanges[0];
    if (current && !current.announced && t >= current.start && t < current.end) {
      current.announced = true;
      this.playingReal = true;
      this.onEvent({ type: "real-playing", ptsS: current.start, waitedMs: Date.now() - current.receivedAt });
    }
  };
}
