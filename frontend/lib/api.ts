// Typed client for the Legal Research Generator backend.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

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
  /** Non-empty when the last interview turn fell back to a canned question. */
  brainstorm_degraded?: string;
}

export interface AppConfig {
  debug_endpoints_enabled: boolean;
  llm_mode: string;
  retriever_mode: string;
  embed_model: string;
  support_scorer: string;
  saul_model: string;
  writer_model: string;
  gemma_model: string;
  hermes_model: string;
  hermes3_model: string;
  multi_chat_models: Record<string, string>;
  multi_chat_verifiers: Record<string, string>;
  dialectic_models?: Record<string, string>;
  grammar_chain_enabled: boolean;
  grammar_chain_roles: string[];
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

export type SocraticMode =
  | "strengthen_doctrine"
  | "expand_analysis"
  | "counter_rebuttal"
  | "policy_implications"
  | "comparative_framework";

export interface SocraticReviseResponse {
  section_id: string;
  paragraph_index: number;
  assistant: string;
  suggested_revision: string;
  applied: boolean;
  mode: SocraticMode;
  cycle_break_triggered: boolean;
  novelty_score: number;
  rewrite_delta_score: number;
}

export interface MultiChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface MultiChatModelAnswer {
  model: string;
  content: string;
}

export interface MultiChatVerifierResult {
  model: string;
  verdict: string;
}

export interface MultiChatCitationFinding {
  citation: string;
  kind: string;
  status: "supported" | "unconfirmed" | "misattributed" | "not_in_corpus" | string;
  detail: string;
  record_id: string;
  support: number;
}

export interface MultiChatGrounding {
  available: boolean;
  summary: string;
  note: string;
  authorities: string[];
  findings: MultiChatCitationFinding[];
  verified_count: number;
  unverified_count: number;
  unconfirmed_count: number;
}

export interface MultiChatResponse {
  final_answer: string;
  model_answers: MultiChatModelAnswer[];
  verifiers: MultiChatVerifierResult[];
  grounding?: MultiChatGrounding;
}

export type DialecticWeight = "controlling" | "persuasive" | "supporting" | "contra";
export type DialecticSlotStatus = "pending" | "not_found" | "verified";

export interface DialecticSlot {
  proposition: string;
  court_hint: string;
  weight: DialecticWeight | string;
  status: DialecticSlotStatus | string;
  cluster_id: string;
  normalized_cite: string;
  note: string;
}

export interface DialecticPosition {
  side: string;
  model: string;
  family: string;
  propositions: DialecticSlot[];
}

export interface DialecticCrux {
  thesis_prop: DialecticSlot;
  antithesis_prop: DialecticSlot;
  negates: boolean;
  partition: string;
  winner: "thesis" | "antithesis" | "none" | string;
  /**
   * True when both sides carry controlling or persuasive weight. A crux between
   * two `supporting` propositions is still a contradiction, just not resolvable
   * by authority — use this to rank, not to filter.
   */
  outcome_bearing: boolean;
  /**
   * Which NLI path produced the label. "heuristic" means the configured NLI
   * model was absent or its answer did not parse, so the offline fallback
   * answered — a weaker basis than "model".
   */
  nli_source: "model" | "heuristic" | string;
}

export interface DialecticResponse {
  question: string;
  thesis: DialecticPosition;
  antithesis: DialecticPosition;
  synthesis: string;
  cruxes: DialecticCrux[];
  calls_spent: number;
  /** Regeneration attempts spent across both positions and the synthesis. */
  regenerated: number;
  /** Why the crux table is empty, when it is. Empty string when it is not. */
  crux_note: string;
  /** Server-rendered copy payloads; these preserve [UNSUPPORTED] markers. */
  copy_exchange: string;
  copy_thesis: string;
  copy_antithesis: string;
  copy_crux_table: string;
}

export interface RetrievalSmokeHit {
  record_id: string;
  score: number;
  locator: string;
  text: string;
}

export interface RetrievalSmokeResponse {
  retriever_mode: string;
  embed_model: string;
  retriever_impl: string;
  hits: RetrievalSmokeHit[];
}

function parseErrorDetail(detail: string): string {
  const trimmed = detail.trim();
  if (!trimmed) return "";
  try {
    const parsed = JSON.parse(trimmed) as { detail?: unknown };
    if (typeof parsed.detail === "string") {
      return parsed.detail;
    }
  } catch {
    // keep raw detail fallback
  }
  return trimmed;
}

function sanitizeErrorMessage(status: number, detail: string): string {
  const normalized = detail.trim();
  const lower = normalized.toLowerCase();
  if (status === 404 && lower.includes("unknown session")) {
    return "Session expired or not found. Refresh and start a new session.";
  }
  const unsafeSignals = [
    "traceback",
    "exception",
    "runtimeerror",
    "valueerror",
    "keyerror",
    "readtimeout",
    "httpx",
    "stack",
  ];
  if (!normalized || unsafeSignals.some((signal) => lower.includes(signal))) {
    return `Request failed (${status}). Please retry.`;
  }
  return normalized;
}

async function jsonFetch<T>(path: string, init?: RequestInit, timeoutMs = 0): Promise<T> {
  const controller = new AbortController();
  const timer = timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : null;
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
      signal: controller.signal,
    });
    if (!res.ok) {
      const rawDetail = await res.text();
      const parsedDetail = parseErrorDetail(rawDetail);
      const safeMessage = sanitizeErrorMessage(res.status, parsedDetail);
      throw new Error(safeMessage);
    }
    return res.json() as Promise<T>;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    throw err;
  } finally {
    if (timer) clearTimeout(timer);
  }
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
  draftEssay: (id: string, targetWords = 2000) =>
    jsonFetch<Blackboard>(`/api/sessions/${id}/draft/essay`, {
      method: "POST",
      body: JSON.stringify({ target_words: targetWords }),
    }),
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
    applyRevision = false,
    mode: SocraticMode = "strengthen_doctrine"
  ) =>
    jsonFetch<SocraticReviseResponse>(`/api/sessions/${id}/revise/socratic`, {
      method: "POST",
      body: JSON.stringify({
        section_id: sectionId,
        paragraph_index: paragraphIndex,
        message,
        history,
        apply_revision: applyRevision,
        mode,
      }),
    }),
  multiChat: (id: string, message: string, history: MultiChatTurn[] = []) =>
    // Backend's multi_chat_max_latency_seconds budget is 260s, plus citation
    // grounding/repair on top of that; give real margin so the client never
    // aborts a request the backend is still legitimately working on.
    jsonFetch<MultiChatResponse>(`/api/sessions/${id}/multi-chat`, {
      method: "POST",
      body: JSON.stringify({ message, history }),
    }, 330_000),
  dialectic: (id: string, message: string) =>
    // Three sequential model calls plus an NLI pass; give the same generous
    // margin as the jury so the client never aborts live work on the Spark.
    jsonFetch<DialecticResponse>(`/api/sessions/${id}/dialectic`, {
      method: "POST",
      body: JSON.stringify({ message }),
    }, 420_000),
  retrievalSmoke: (query: string, k = 3) =>
    jsonFetch<RetrievalSmokeResponse>("/api/retrieval/smoke", {
      method: "POST",
      body: JSON.stringify({ query, k }),
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
  pdfUrl: (id: string) => `${API_BASE}/api/sessions/${id}/pdf`,
};
