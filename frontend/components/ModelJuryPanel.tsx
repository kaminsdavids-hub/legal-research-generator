"use client";

import {
  AlertTriangle,
  Bot,
  Check,
  Copy,
  HelpCircle,
  Library,
  Loader2,
  Send,
  ShieldCheck,
} from "lucide-react";
import { useMemo, useState } from "react";
import type { MultiChatModelAnswer, MultiChatResponse, MultiChatTurn } from "@/lib/api";
import { Button, Card, Empty } from "./ui";

const CONSOLIDATION_TIMEOUT_PREFIX =
  "model jury timed out before producing a consolidated answer";

function isUnavailableAnswer(text: string): boolean {
  const t = text.trim().toLowerCase();
  return (
    !t ||
    t.startsWith("(unavailable:") ||
    t.startsWith("(skipped:") ||
    t.startsWith("(no response")
  );
}

function bestPanelAnswer(modelAnswers: MultiChatModelAnswer[]): string | null {
  const candidates = modelAnswers
    .map((answer) => answer.content.trim())
    .filter((text) => text && !isUnavailableAnswer(text));
  if (candidates.length === 0) return null;
  return candidates.sort((a, b) => b.length - a.length)[0];
}

function deterministicClientFallback(prompt: string): string {
  const cleaned = prompt.trim().replace(/\s+/g, " ").slice(0, 220);
  return (
    "Degraded-mode analytical answer (jury consolidation unavailable): " +
    `For '${cleaned}', start by identifying the controlling rule and dividing it into concrete doctrinal elements. ` +
    "Then apply the known facts element-by-element, separating direct evidence from inference and flagging where proof is uncertain. " +
    "Finally, test the analysis against the strongest counterargument and explain why that counterposition narrows, but does not necessarily defeat, the claim. " +
    "The factual account should distinguish established facts from open questions, especially on mental state, causation, and alternative explanations."
  );
}

function resolveAssistantAnswer(
  response: MultiChatResponse,
  prompt: string
): { answer: string; notice: string } {
  const finalAnswer = response.final_answer.trim();
  const timedOut = finalAnswer.toLowerCase().startsWith(CONSOLIDATION_TIMEOUT_PREFIX);
  if (!timedOut) {
    if (finalAnswer) {
      return { answer: finalAnswer, notice: "" };
    }
    return {
      answer: deterministicClientFallback(prompt),
      notice: "Jury returned no consolidated text, so showing an analytical fallback answer.",
    };
  }
  const fallback = bestPanelAnswer(response.model_answers);
  if (fallback) {
    return {
      answer: fallback,
      notice:
        "Jury synthesis timed out, so showing the best available single-model answer below.",
    };
  }
  return {
    answer: deterministicClientFallback(prompt),
    notice:
      "Jury synthesis timed out and panel outputs were unavailable, so showing an analytical fallback answer.",
  };
}

export function ModelJuryPanel({
  modelMap,
  verifierMap,
  onAsk,
}: {
  modelMap: Record<string, string>;
  verifierMap: Record<string, string>;
  onAsk: (message: string, history: MultiChatTurn[]) => Promise<MultiChatResponse>;
}) {
  const [draft, setDraft] = useState("");
  const [history, setHistory] = useState<MultiChatTurn[]>([]);
  const [latest, setLatest] = useState<MultiChatResponse | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [copiedKey, setCopiedKey] = useState("");
  const [busy, setBusy] = useState(false);

  const modelSummary = useMemo(
    () => Object.entries(modelMap).map(([name, model]) => `${name}: ${model}`),
    [modelMap]
  );
  const verifierSummary = useMemo(
    () => Object.entries(verifierMap).map(([name, model]) => `${name}: ${model}`),
    [verifierMap]
  );

  async function send() {
    const text = draft.trim();
    if (!text || busy) return;
    setBusy(true);
    setError("");
    setNotice("");
    const nextHistory: MultiChatTurn[] = [...history, { role: "user", content: text }];
    setHistory(nextHistory);
    setDraft("");
    try {
      const response = await onAsk(text, history);
      const resolved = resolveAssistantAnswer(response, text);
      setLatest(response);
      setNotice(resolved.notice);
      setHistory([...nextHistory, { role: "assistant", content: resolved.answer }]);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function copyText(key: string, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      setTimeout(() => {
        setCopiedKey((current) => (current === key ? "" : current));
      }, 1400);
    } catch {
      setError("Unable to copy text from this browser context.");
    }
  }

  return (
    <Card
      title="Model Jury Chat"
      icon={<Bot size={16} className="text-accent" />}
      actions={
        <div className="hidden max-w-[36rem] text-right text-[11px] text-slate-400 xl:block">
          <div>{modelSummary.join(" · ")}</div>
          <div className="mt-0.5">verify: {verifierSummary.join(" · ")}</div>
        </div>
      }
      className="h-full"
    >
      <div className="flex h-full flex-col">
        <div className="space-y-3 overflow-auto pr-1">
          {error && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
              {error}
            </div>
          )}

          {notice && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
              {notice}
            </div>
          )}

          {history.length === 0 && !latest && (
            <Empty>
              Ask one question. The module will query Gemma 4, Apertus, GPT-OSS,
              Nemotron, and Hermes 3, then run Gemma 3 + Saul verification.
            </Empty>
          )}

          {history.map((turn, idx) => (
            <div
              key={idx}
              className={
                turn.role === "user"
                  ? "ml-8 rounded-lg bg-accent/10 px-3 py-2 text-sm text-slate-800"
                  : "mr-8 rounded-lg bg-slate-100 px-3 py-2 text-sm text-slate-700"
              }
            >
              <div className="mb-0.5 flex items-center justify-between text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                <span>{turn.role}</span>
                <button
                  type="button"
                  onClick={() => void copyText(`turn-${idx}`, turn.content)}
                  className="inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-[10px] normal-case text-slate-500 hover:bg-slate-50"
                >
                  {copiedKey === `turn-${idx}` ? <Check size={11} /> : <Copy size={11} />}
                  {copiedKey === `turn-${idx}` ? "Copied" : "Copy"}
                </button>
              </div>
              <div className="whitespace-pre-wrap">{turn.content}</div>
            </div>
          ))}

          {busy && (
            <div className="mr-8 rounded-lg bg-slate-100 px-3 py-2 text-sm text-slate-600">
              <Loader2 className="inline animate-spin" size={14} /> Running model panel...
            </div>
          )}

          {latest && (
            <div className="space-y-2 rounded-lg border border-slate-200 bg-white p-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Per-model responses
              </div>
              {latest.model_answers.map((answer) => (
                <div
                  key={answer.model}
                  className={
                    isUnavailableAnswer(answer.content)
                      ? "rounded-md border border-amber-200 bg-amber-50 px-2.5 py-2"
                      : "rounded-md bg-slate-50 px-2.5 py-2"
                  }
                >
                  <div className="mb-1 flex items-center justify-between text-[11px] font-semibold text-slate-500">
                    <span>{answer.model}</span>
                    <button
                      type="button"
                      onClick={() => void copyText(`model-${answer.model}`, answer.content)}
                      className="inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-[10px] font-medium text-slate-500 hover:bg-slate-50"
                    >
                      {copiedKey === `model-${answer.model}` ? <Check size={11} /> : <Copy size={11} />}
                      {copiedKey === `model-${answer.model}` ? "Copied" : "Copy"}
                    </button>
                  </div>
                  <div
                    className={
                      isUnavailableAnswer(answer.content)
                        ? "whitespace-pre-wrap text-xs text-amber-800"
                        : "whitespace-pre-wrap text-xs text-slate-700"
                    }
                  >
                    {answer.content}
                  </div>
                </div>
              ))}
              {latest.grounding && (
                <>
                  <div className="pt-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                    <Library size={12} className="mr-1 inline text-indigo-600" /> Citation grounding
                  </div>
                  <div
                    className={
                      latest.grounding.unverified_count > 0
                        ? "rounded-md border border-rose-200 bg-rose-50 px-2.5 py-2"
                        : "rounded-md border border-indigo-100 bg-indigo-50 px-2.5 py-2"
                    }
                  >
                    <div
                      className={
                        latest.grounding.unverified_count > 0
                          ? "text-xs font-medium text-rose-800"
                          : "text-xs font-medium text-indigo-900"
                      }
                    >
                      {latest.grounding.available
                        ? latest.grounding.summary
                        : latest.grounding.note || "Citation grounding unavailable."}
                    </div>

                    {latest.grounding.findings.length > 0 && (
                      <ul className="mt-1.5 space-y-1">
                        {latest.grounding.findings.map((finding, idx) => (
                          <li
                            key={`${finding.citation}-${idx}`}
                            className="flex items-start gap-1.5 text-[11px] leading-snug"
                          >
                            {finding.status === "supported" ? (
                              <Check size={12} className="mt-0.5 shrink-0 text-emerald-600" />
                            ) : finding.status === "unconfirmed" ? (
                              <HelpCircle size={12} className="mt-0.5 shrink-0 text-amber-600" />
                            ) : (
                              <AlertTriangle size={12} className="mt-0.5 shrink-0 text-rose-600" />
                            )}
                            <span
                              className={
                                finding.status === "supported"
                                  ? "text-slate-700"
                                  : finding.status === "unconfirmed"
                                    ? "text-amber-800"
                                    : "text-rose-800"
                              }
                            >
                              <span className="font-semibold">{finding.citation}</span>
                              {" \u2014 "}
                              {finding.detail}
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}

                    {latest.grounding.authorities.length > 0 && (
                      <div className="mt-2 border-t border-indigo-100 pt-1.5">
                        <div className="text-[10px] font-semibold uppercase tracking-wide text-indigo-700">
                          Retrieved authority
                        </div>
                        <ul className="mt-0.5 space-y-0.5">
                          {latest.grounding.authorities.map((authority) => (
                            <li key={authority} className="text-[11px] text-slate-700">
                              {authority}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </>
              )}

              <div className="pt-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                <ShieldCheck size={12} className="mr-1 inline text-emerald-600" /> Verifier checks
              </div>
              {latest.verifiers.map((verifier) => (
                <div key={verifier.model} className="rounded-md border border-emerald-100 bg-emerald-50 px-2.5 py-2">
                  <div className="mb-1 flex items-center justify-between text-[11px] font-semibold text-emerald-700">
                    <span>{verifier.model}</span>
                    <button
                      type="button"
                      onClick={() => void copyText(`verifier-${verifier.model}`, verifier.verdict)}
                      className="inline-flex items-center gap-1 rounded border border-emerald-200 bg-white px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 hover:bg-emerald-50"
                    >
                      {copiedKey === `verifier-${verifier.model}` ? <Check size={11} /> : <Copy size={11} />}
                      {copiedKey === `verifier-${verifier.model}` ? "Copied" : "Copy"}
                    </button>
                  </div>
                  <div className="whitespace-pre-wrap text-xs text-emerald-900">{verifier.verdict}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="mt-3 flex items-end gap-2 border-t border-slate-100 pt-3">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            rows={2}
            placeholder="Ask a legal or strategy question..."
            className="flex-1 resize-none rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-accent focus:outline-none"
          />
          <Button onClick={() => void send()} disabled={busy || !draft.trim()}>
            <Send size={14} /> Ask
          </Button>
        </div>
      </div>
    </Card>
  );
}
