export interface StreamSession {
  stream_id: string;
  ws_url: string;
  width: number;
  height: number;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<T>;
}

export function createStream(inputReference: string | null, width = 240, height = 416): Promise<StreamSession> {
  return post<StreamSession>("/v1/streams", { width, height, input_reference: inputReference });
}

export function generateClip(
  session: StreamSession,
  prompt: string,
  inputReference: string | null,
  seamlessEnding: boolean,
): Promise<{ job_id: string; status: string }> {
  return post(`/v1/streams/${session.stream_id}/generate`, {
    prompt,
    input_reference: inputReference,
    // FLF back to the reference frame => the cut back into the idle loop is
    // seamless too, not just the cut in.
    end_image: seamlessEnding ? inputReference : null,
    width: session.width,
    height: session.height,
    enable_frame_interpolation: true,
  });
}
