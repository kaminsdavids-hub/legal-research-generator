// Typed client for the Legal Research Generator backend.

// Same origin by default: `/api/*` is handled by the Netlify function that
// holds the backend key. Set NEXT_PUBLIC_API_URL only for local development
// against a backend running on this machine, where there is no proxy.
export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

// --------------------------------------------------------------------------
// Credentials
//
// The backend key is NOT here and must never be. It lives in the Netlify
// function's environment (netlify/functions/api-proxy.mts), which runs on a
// server; this bundle is public and anything compiled into it is readable.
//
// What the browser may hold is the proxy password, which is a different secret:
// it authorises calls to the proxy, not to the backend, so it can be rotated
// without touching the host and it confers no direct access to it. If the site
// is gated another way -- Netlify site-level password protection, or an access
// rule in front of it -- leave this unset and the prompt never appears.
// --------------------------------------------------------------------------

const PASSWORD_STORAGE = "lrg.proxyPassword";

export function getProxyPassword(): string {
  if (typeof window === "undefined") return ""; // static export prerenders on the server
  return window.localStorage.getItem(PASSWORD_STORAGE) ?? "";
}

export function setProxyPassword(password: string): void {
  if (typeof window === "undefined") return;
  const trimmed = password.trim();
  if (trimmed) window.localStorage.setItem(PASSWORD_STORAGE, trimmed);
  else window.localStorage.removeItem(PASSWORD_STORAGE);
}

export function clearProxyPassword(): void {
  setProxyPassword("");
}

/** A 401. Distinct so the UI can ask for the password rather than showing a
 *  generic failure -- "unauthorized" and "the server is down" call for
 *  completely different things from the user. */
export class UnauthorizedError extends Error {
  constructor(message = "missing or invalid proxy password") {
    super(message);
    this.name = "UnauthorizedError";
  }
}

function authHeaders(): Record<string, string> {
  const password = getProxyPassword();
  return password ? { "X-Proxy-Password": password } : {};
}

export type IdeaStatus = "proposed" | "keep" | "cut";
export type SectionStatus = "idea" | "drafted" | "cited" | "verified";
export type CiteStatus = "pending" | "verified" | "needs_review" | "removed";

export interface BrainstormTurn {
  role: "scholar" | "interviewer" | "system";
  content: string;
}

export interface Idea {
  id: string;
  text: string;
  angle: string;
  novelty_note: string;
  status: IdeaStatus;
  priority: number;
}

export interface OutlineSection {
  id: string;
  title: string;
  content: string;
  status: SectionStatus;
  idea_ids: string[];
  citation_ids: string[];
}

export interface Authority {
  record_id: string;
  citation: string;
  relation: "supporting" | "contrary" | "neutral";
  proposition: string;
  passage: string;
}

export interface Citation {
  id: string;
  record_id: string;
  proposition: string;
  quote?: string | null;
  pin_cite?: string | null;
  supporting_passage: string;
  from_retrieval: boolean;
  status: CiteStatus;
  note: string;
}

export interface VerificationResult {
  citation_id: string;
  record_id: string;
  status: CiteStatus;
  reason: string;
  supporting_passage: string;
}

export interface Novelty {
  contribution: string;
  distinguished_from: string[];
  score: number;
  grounded: boolean;
}

export interface Blackboard {
  session_id: string;
  title: string;
  thesis: string;
  brainstorm: BrainstormTurn[];
  ideas: Idea[];
  outline: OutlineSection[];
  authorities: Authority[];
  citations: Citation[];
  verifications: VerificationResult[];
  footnotes: Record<string, { number: number; text: string; citation_id: string }[]>;
  toa: Record<string, string[]>;
  novelty: Novelty | null;
}

export interface AppConfig {
  llm_mode: string;
  retriever_mode: string;
  embed_model: string;
  support_scorer: string;
  pdf_renderer: string;
  citation_style: string;
  manuscript_target_min_words: number;
  manuscript_target_max_words: number;
  disclaimer: string;
}

export interface SocraticTurn {
  role: "user" | "assistant";
  content: string;
}

export interface SocraticReviseResponse {
  section_id: string;
  paragraph_index: number;
  assistant: string;
  suggested_revision: string;
  applied: boolean;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    // Spread init first: the caller may pass its own headers, and these must
    // not be droppable by it.
    headers: { "Content-Type": "application/json", ...authHeaders(), ...(init?.headers ?? {}) },
  });
  if (res.status === 401) throw new UnauthorizedError(await res.text());
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

/** Fetch a binary response and hand back an object URL.
 *
 *  An <a href> cannot carry a header, so a gated download cannot be a plain
 *  link -- the browser would request it unauthenticated and get a 401. The
 *  caller is responsible for revoking the URL when the download is done. */
export async function fetchBlobUrl(path: string): Promise<string> {
  const res = await fetch(`${API_BASE}${path}`, { headers: authHeaders() });
  if (res.status === 401) throw new UnauthorizedError();
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return URL.createObjectURL(await res.blob());
}

// --------------------------------------------------------------------------
// Slow steps
//
// These drive a model and can run for minutes. Sent through the job routes so
// every individual HTTP call is fast: the Netlify function in front of this has
// a 26-second ceiling, and a five-minute request cannot be made to fit under it
// however it is written.
//
// The synchronous routes still exist on the backend and are correct for a
// loopback client with nothing in between. This client does not use them,
// because it is the one that runs behind the proxy.
// --------------------------------------------------------------------------

export type JobState = "running" | "succeeded" | "failed";

export interface JobStatus {
  job_id: string;
  session_id: string;
  step: string;
  state: JobState;
  error: string;
}

export type JobStep =
  | "brainstorm"
  | "ideate"
  | "outline"
  | "research"
  | "draft"
  | "voice"
  | "verify"
  | "format"
  | "novelty"
  | "mechanism";

/** Submit a step and resolve when it finishes, or reject with what went wrong.
 *
 *  `onState` fires on every poll so a caller can show progress; a step that
 *  takes four minutes with no feedback is indistinguishable from a hang.
 *
 *  The 2s interval is a deliberate floor: these steps take tens of seconds at
 *  best, so polling faster only multiplies function invocations, which on
 *  Netlify are metered. */
export async function runStep(
  sessionId: string,
  step: JobStep,
  onState?: (state: JobState) => void,
  intervalMs = 2000
): Promise<Blackboard> {
  const started = await jsonFetch<JobStatus>(`/api/sessions/${sessionId}/jobs`, {
    method: "POST",
    body: JSON.stringify({ step }),
  });
  onState?.(started.state);

  let state = started.state;
  while (state === "running") {
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
    const status = await jsonFetch<JobStatus>(`/api/jobs/${started.job_id}`);
    state = status.state;
    onState?.(state);
    if (state === "failed") throw new Error(`${step} failed: ${status.error}`);
  }
  // Fetched once, on success. The poll deliberately does not carry the
  // blackboard: it is large, and a poller would re-download it every 2s.
  return jsonFetch<Blackboard>(`/api/sessions/${sessionId}`);
}

export const api = {
  config: () => jsonFetch<AppConfig>("/api/config"),
  createSession: (title: string) =>
    jsonFetch<Blackboard>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ title }),
    }),
  getSession: (id: string) => jsonFetch<Blackboard>(`/api/sessions/${id}`),
  brainstorm: (id: string, message: string | null) =>
    jsonFetch<Blackboard>(`/api/sessions/${id}/brainstorm`, {
      method: "POST",
      body: JSON.stringify({ message }),
    }),
  ideate: (id: string, seed: string | null) =>
    jsonFetch<Blackboard>(`/api/sessions/${id}/ideate`, {
      method: "POST",
      body: JSON.stringify({ seed }),
    }),
  updateIdea: (id: string, ideaId: string, status: IdeaStatus, priority?: number) =>
    jsonFetch<Blackboard>(`/api/sessions/${id}/ideas/${ideaId}`, {
      method: "PATCH",
      body: JSON.stringify({ status, priority: priority ?? null }),
    }),
  outline: (id: string) =>
    jsonFetch<Blackboard>(`/api/sessions/${id}/outline`, { method: "POST" }),
  research: (id: string) =>
    jsonFetch<Blackboard>(`/api/sessions/${id}/research`, { method: "POST" }),
  draft: (id: string) =>
    jsonFetch<Blackboard>(`/api/sessions/${id}/draft`, { method: "POST" }),
  voice: (id: string) =>
    jsonFetch<Blackboard>(`/api/sessions/${id}/voice`, { method: "POST" }),
  verify: (id: string) =>
    jsonFetch<Blackboard>(`/api/sessions/${id}/verify`, { method: "POST" }),
  format: (id: string) =>
    jsonFetch<Blackboard>(`/api/sessions/${id}/format`, { method: "POST" }),
  novelty: (id: string) =>
    jsonFetch<Blackboard>(`/api/sessions/${id}/novelty`, { method: "POST" }),
  revise: (id: string, sectionId: string, instruction: string) =>
    jsonFetch<Blackboard>(`/api/sessions/${id}/revise`, {
      method: "POST",
      body: JSON.stringify({ section_id: sectionId, instruction }),
    }),
  socraticRevise: (
    id: string,
    sectionId: string,
    paragraphIndex: number,
    message: string,
    history: SocraticTurn[] = [],
    applyRevision = false
  ) =>
    jsonFetch<SocraticReviseResponse>(`/api/sessions/${id}/revise/socratic`, {
      method: "POST",
      body: JSON.stringify({
        section_id: sectionId,
        paragraph_index: paragraphIndex,
        message,
        history,
        apply_revision: applyRevision,
      }),
    }),
  runAll: (id: string, idea: string, title: string) =>
    jsonFetch<{ session_id: string; steps: { agent: string; runtime: string; summary: string }[]; shippable: boolean }>(
      `/api/sessions/${id}/run-all`,
      { method: "POST", body: JSON.stringify({ idea, title }) }
    ),
  report: (id: string) =>
    jsonFetch<{ markdown: string }>(`/api/sessions/${id}/report`),
  preview: (id: string) =>
    jsonFetch<{ html: string }>(`/api/sessions/${id}/preview`),
  /** Kept for callers that only need the address (e.g. to display it). It is
   *  NOT usable as a download link once a key is set -- use `downloadPdf`. */
  pdfUrl: (id: string) => `${API_BASE}/api/sessions/${id}/pdf`,
  downloadPdf: (id: string) => fetchBlobUrl(`/api/sessions/${id}/pdf`),
};
