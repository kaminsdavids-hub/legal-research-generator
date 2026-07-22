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
}

export interface AppConfig {
  llm_mode: string;
  retriever_mode: string;
  pdf_renderer: string;
  citation_style: string;
  disclaimer: string;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return res.json() as Promise<T>;
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
