"use client";

import {
  BookOpen,
  CheckCircle2,
  FlaskConical,
  ListTree,
  Loader2,
  PenLine,
  PlayCircle,
  Quote,
  ScrollText,
  Sparkles,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { BrainstormPanel } from "@/components/BrainstormPanel";
import { CitationsPanel } from "@/components/CitationsPanel";
import { DialecticPanel } from "@/components/DialecticPanel";
import { IdeaBoard } from "@/components/IdeaBoard";
import { ManuscriptPanel } from "@/components/ManuscriptPanel";
import { ModelJuryPanel } from "@/components/ModelJuryPanel";
import { Button } from "@/components/ui";
import {
  api,
  type AppConfig,
  type Blackboard,
  type IdeaStatus,
  type SocraticMode,
  type SocraticReviseResponse,
  type SocraticTurn,
} from "@/lib/api";

type Tab = "manuscript" | "citations" | "dialectic";

const STEPS = [
  { key: "outline", label: "Outline", icon: ListTree, run: api.outline },
  { key: "research", label: "Research", icon: BookOpen, run: api.research },
  { key: "draft", label: "Draft", icon: PenLine, run: api.draft },
  { key: "voice", label: "Grammar", icon: Sparkles, run: api.voice },
  { key: "verify", label: "Verify", icon: CheckCircle2, run: api.verify },
  { key: "format", label: "Format", icon: Quote, run: api.format },
  { key: "novelty", label: "Novelty", icon: FlaskConical, run: api.novelty },
] as const;

//: Where the active session id lives between page loads. Without this the app
//: created a new manuscript on every refresh (see the mount effect below).
const SESSION_KEY = "lrg.session_id";

export default function Home() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [bb, setBb] = useState<Blackboard | null>(null);
  const [tab, setTab] = useState<Tab>("manuscript");
  const [report, setReport] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [title, setTitle] = useState("Untitled Research Paper");

  useEffect(() => {
    (async () => {
      const cfg = await api.config();
      setConfig(cfg);

      // Resume the stored session before creating one. This effect used to POST
      // /api/sessions unconditionally on every mount, so a page refresh silently
      // started a new manuscript and orphaned the previous one on the server --
      // every idea, draft and citation the author had built was still on disk
      // but unreachable from the UI.
      const stored =
        typeof window === "undefined" ? null : window.localStorage.getItem(SESSION_KEY);
      if (stored) {
        try {
          const existing = await api.getSession(stored);
          setBb(existing);
          if (existing.title) setTitle(existing.title);
          return;
        } catch {
          // The id no longer resolves -- the backend was restarted, or the
          // session was pruned. Fall through and start a fresh one rather than
          // leaving the app with no session at all.
        }
      }

      const session = await api.createSession(title);
      if (typeof window !== "undefined") {
        window.localStorage.setItem(SESSION_KEY, session.session_id);
      }
      setBb(session);
    })().catch(console.error);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Resuming by default means there has to be a deliberate way out, or the
  // author is stuck in one manuscript forever with no route to a second.
  const startNewSession = useCallback(async () => {
    const session = await api.createSession("Untitled Research Paper");
    if (typeof window !== "undefined") {
      window.localStorage.setItem(SESSION_KEY, session.session_id);
    }
    setTitle(session.title || "Untitled Research Paper");
    setReport("");
    setBb(session);
  }, []);

  const refreshReport = useCallback(async (id: string) => {
    try {
      setReport((await api.report(id)).markdown);
    } catch {
      /* report only available after verify */
    }
  }, []);

  // The full pipeline (run-all) is a single long-running request that can take
  // many minutes against real models. Poll the session while it's in flight so
  // the manuscript panel shows sections filling in live instead of leaving the
  // user staring at a spinner with no feedback that anything is happening.
  useEffect(() => {
    if (busy !== "run-all" || !bb) return;
    const sessionId = bb.session_id;
    const interval = setInterval(() => {
      api
        .getSession(sessionId)
        .then(setBb)
        .catch(() => {
          /* transient poll failure; next tick will retry */
        });
    }, 4000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busy, bb?.session_id]);

  async function socraticRevise(
    sectionId: string,
    paragraphIndex: number,
    message: string,
    history: SocraticTurn[],
    applyRevision: boolean,
    mode: SocraticMode
  ): Promise<SocraticReviseResponse> {
    if (!bb) {
      throw new Error("No active session");
    }
    setBusy("socratic");
    try {
      const sessionId = bb.session_id;
      const response = await api.socraticRevise(
        sessionId,
        sectionId,
        paragraphIndex,
        message,
        history,
        applyRevision,
        mode
      );
      if (response.applied) {
        const next = await api.getSession(sessionId);
        setBb(next);
        await refreshReport(next.session_id);
      }
      return response;
    } finally {
      setBusy(null);
    }
  }

  async function withBusy(key: string, fn: () => Promise<Blackboard>) {
    setBusy(key);
    try {
      const next = await fn();
      setBb(next);
      await refreshReport(next.session_id);
    } catch (e) {
      console.error(e);
      if (key === "run-all" && bb) {
        // The backend mutates the session in place as each pipeline stage
        // completes, so even if this request was interrupted (client abort,
        // proxy/funnel idle timeout, network drop), whatever the pipeline
        // finished before the interruption is still on the server. Recover
        // it instead of discarding a possibly-complete manuscript.
        try {
          const recovered = await api.getSession(bb.session_id);
          setBb(recovered);
          await refreshReport(recovered.session_id);
          alert(
            "The full-pipeline request was interrupted, but partial progress " +
              "was recovered from the server. Check the manuscript panel; you " +
              "may want to re-run remaining steps individually."
          );
        } catch (recoverError) {
          console.error(recoverError);
          alert(e instanceof Error ? e.message : "Request failed. Please retry.");
        }
      } else {
        alert(e instanceof Error ? e.message : "Request failed. Please retry.");
      }
    } finally {
      setBusy(null);
    }
  }

  if (!bb || !config) {
    return (
      <div className="grid h-screen place-items-center text-slate-400">
        <div className="flex items-center gap-2">
          <Loader2 className="animate-spin" size={18} /> Connecting to backend…
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-5 py-3">
        <div className="flex items-center gap-3">
          <ScrollText className="text-accent" size={22} />
          <div>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              onBlur={() => bb && setBb({ ...bb, title })}
              className="w-[28rem] max-w-full rounded border-none text-lg font-semibold text-ink focus:outline-none"
            />
            <div className="text-xs text-slate-400">
              {config.llm_mode} · {config.retriever_mode} retriever · {config.support_scorer} verify ·
              {config.embed_model.split("/").at(-1) ?? config.embed_model} · {config.citation_style} · grammar {config.grammar_chain_enabled ? config.grammar_chain_roles.join("→") : "off"} · target {config.manuscript_target_min_words.toLocaleString()}-
              {config.manuscript_target_max_words.toLocaleString()} words
            </div>
            <div className="text-[11px] text-slate-400">
              saul={config.saul_model} · writer={config.writer_model} · gemma={config.gemma_model} ·
              hermes={config.hermes_model} · hermes3={config.hermes3_model}
            </div>
            <div className="text-[11px] text-slate-400">
              module chat={Object.values(config.multi_chat_models).join(" | ")} ·
              verify={Object.values(config.multi_chat_verifiers).join(" | ")}
            </div>
            <div className="text-[11px] text-slate-400">
              debug endpoints {config.debug_endpoints_enabled ? "on" : "off"}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            onClick={() =>
              withBusy("run-all", async () => {
                await api.runAll(bb.session_id, bb.thesis || title, title);
                return api.getSession(bb.session_id);
              })
            }
            disabled={busy !== null}
          >
            {busy === "run-all" ? <Loader2 className="animate-spin" size={14} /> : <PlayCircle size={14} />}
            Run full pipeline
          </Button>
          <Button
            variant="subtle"
            onClick={() => {
              if (
                window.confirm(
                  "Start a new paper? The current one stays on the server but " +
                    "this browser will stop resuming it."
                )
              ) {
                startNewSession().catch(console.error);
              }
            }}
          >
            New paper
          </Button>
          <a
            href={api.pdfUrl(bb.session_id)}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-700"
          >
            <ScrollText size={14} /> Export
          </a>
        </div>
      </header>

      <div className="flex items-center gap-1.5 border-b border-slate-200 bg-white px-5 py-2">
        {STEPS.map((step) => (
          <Button
            key={step.key}
            variant="subtle"
            disabled={busy !== null}
            onClick={() => withBusy(step.key, () => step.run(bb.session_id))}
          >
            {busy === step.key ? (
              <Loader2 className="animate-spin" size={14} />
            ) : (
              <step.icon size={14} />
            )}
            {step.label}
          </Button>
        ))}
        <Button
          variant="subtle"
          disabled={busy !== null}
          onClick={() => withBusy("draft-essay", () => api.draftEssay(bb.session_id, 2000))}
        >
          {busy === "draft-essay" ? <Loader2 className="animate-spin" size={14} /> : <PenLine size={14} />}
          Draft ~2,000 words
        </Button>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 p-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <div className="grid min-h-0 grid-rows-3 gap-4">
          <BrainstormPanel
            bb={bb}
            onRefresh={setBb}
            onIdeate={(seed) =>
              withBusy("ideate", async () => api.ideate(bb.session_id, seed || null))
            }
          />
          <ModelJuryPanel
            modelMap={config.multi_chat_models}
            verifierMap={config.multi_chat_verifiers}
            onAsk={(message, history) =>
              api.multiChat(bb.session_id, message, history)
            }
          />
          <IdeaBoard
            bb={bb}
            onUpdate={(ideaId: string, status: IdeaStatus, priority?: number) =>
              withBusy("idea", async () => api.updateIdea(bb.session_id, ideaId, status, priority))
            }
          />
        </div>

        <div className="flex min-h-0 flex-col">
          <div className="mb-2 flex gap-1.5">
            {(["manuscript", "citations", "dialectic"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={
                  "rounded-lg px-3 py-1.5 text-sm font-medium capitalize transition " +
                  (tab === t ? "bg-accent text-white" : "bg-white text-slate-600 hover:bg-slate-50")
                }
              >
                {t}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1">
            {tab === "manuscript" && (
              <ManuscriptPanel
                bb={bb}
                socraticBusy={busy === "socratic"}
                onRevise={(sectionId, instruction) =>
                  withBusy("revise", async () =>
                    api.revise(bb.session_id, sectionId, instruction)
                  )
                }
                onSocratic={socraticRevise}
              />
            )}
            {tab === "citations" && <CitationsPanel bb={bb} report={report} />}
            {tab === "dialectic" && (
              <DialecticPanel
                modelMap={config.dialectic_models}
                onAsk={(message) => api.dialectic(bb.session_id, message)}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
