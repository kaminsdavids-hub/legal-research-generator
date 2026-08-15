// Typed client for the Legal Research Generator backend.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

// --------------------------------------------------------------------------
// The API key
//
// The backend gates every /api route except /api/health on a shared key. This
// bundle is static and served from a public URL, so the key must NOT be built
// into it -- an NEXT_PUBLIC_* value is readable by anyone who opens dev tools,
// which would make the gate decoration. The user supplies it once and the
// browser keeps it.
//
// localStorage, not a cookie: nothing here is same-origin with the API, so a
// cookie would have to be third-party and would be dropped by default in most
// browsers. The tradeoff is that any script running on this origin can read the
// key, which is the honest cost of a single-page app holding a credential at
// all.
// --------------------------------------------------------------------------

const KEY_STORAGE = "lrg.apiKey";

export function getApiKey(): string {
  if (typeof window === "undefined") return ""; // static export prerenders on the server
  return window.localStorage.getItem(KEY_STORAGE) ?? "";
}

export function setApiKey(key: string): void {
  if (typeof window === "undefined") return;
  const trimmed = key.trim();
  if (trimmed) window.localStorage.setItem(KEY_STORAGE, trimmed);
  else window.localStorage.removeItem(KEY_STORAGE);
}

export function clearApiKey(): void {
  setApiKey("");
}

/** A 401 from the backend. Distinct so the UI can ask for a key rather than
 *  showing a generic failure -- "unauthorized" and "the server is down" call
 *  for completely different things from the user. */
export class UnauthorizedError extends Error {
  constructor(message = "missing or invalid API key") {
    super(message);
    this.name = "UnauthorizedError";
  }
}

function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return key ? { "X-API-Key": key } : {};
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
