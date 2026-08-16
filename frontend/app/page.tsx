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
  type LucideIcon,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { BrainstormPanel } from "@/components/BrainstormPanel";
import { CitationsPanel } from "@/components/CitationsPanel";
import { IdeaBoard } from "@/components/IdeaBoard";
import { ManuscriptPanel } from "@/components/ManuscriptPanel";
import { ModelJuryPanel } from "@/components/ModelJuryPanel";
import { Button } from "@/components/ui";
import {
  api,
  askAsJob,
  runAll,
  runStep,
  setProxyPassword,
  UnauthorizedError,
  type JobEvent,
  type JobStep,
  type AppConfig,
  type Blackboard,
  type IdeaStatus,
  type MultiChatResponse,
  type SocraticReviseResponse,
  type SocraticTurn,
} from "@/lib/api";

type Tab = "manuscript" | "citations" | "jury";

// Run as jobs, not through the synchronous routes: those hold a request open
// for the whole step, which nothing in front of this app is willing to wait for.
const STEPS: readonly { key: JobStep; label: string; icon: LucideIcon }[] = [
  { key: "outline", label: "Outline", icon: ListTree },
  { key: "research", label: "Research", icon: BookOpen },
  { key: "draft", label: "Draft", icon: PenLine },
  { key: "voice", label: "Voice", icon: Sparkles },
  { key: "verify", label: "Verify", icon: CheckCircle2 },
  { key: "format", label: "Format", icon: Quote },
  { key: "novelty", label: "Novelty", icon: FlaskConical },
];

/** One line of progress, in the words of whatever produced it.
 *
 * Deliberately not a percentage. Only run-all knows how many stages it has, and
 * a bar that guessed at the others would be inventing a denominator — the panel
 * finishes in whatever order the models finish, and the dialectic's stages take
 * wildly different times. What a person waiting actually wants is evidence that
 * something moved, and which thing it was. */
function describeEvent(e: JobEvent): string {
  switch (e.event) {
    // multi-chat
    case "panel_started":
      return `asking ${e.models?.length ?? 0} models`;
    case "model_answered":
      return e.answered
        ? `${e.model} answered · ${e.seconds}s · ${e.characters} chars`
        : `${e.model} returned nothing (timed out or failed)`;
    case "synthesising":
      return e.answers === undefined ? "synthesising" : `synthesising ${e.answers} answers`;
    // dialectic
    case "generating":
      return `generating ${e.side}`;
    case "position_generated":
      return `${e.side}: ${e.propositions} propositions${e.retries ? ` (${e.retries} retries)` : ""}`;
    case "retrieving":
      return "retrieving authority";
    case "retrieved":
      return `retrieved ${e.filled} citations`;
    case "verifying":
      return "verifying against CourtListener";
    case "verified":
      return `verified · ${e.calls_spent} calls`;
    case "cruxes_extracted":
      return e.count ? `${e.count} cruxes` : e.note || "no cruxes";
    // run-all
    case "step_completed":
      return `${e.index}. ${e.agent}${e.degraded ? " — fell back" : ""}`;
    default:
      return e.event;
  }
}

export default function Home() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [bb, setBb] = useState<Blackboard | null>(null);
  const [tab, setTab] = useState<Tab>("manuscript");
  const [report, setReport] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState<JobEvent[]>([]);
  const [title, setTitle] = useState("Untitled Research Paper");
  const [needsKey, setNeedsKey] = useState(false);
  const [keyInput, setKeyInput] = useState("");
  const [connectError, setConnectError] = useState<string | null>(null);

  const connect = useCallback(async () => {
    setConnectError(null);
    try {
      const [cfg, session] = await Promise.all([api.config(), api.createSession(title)]);
      setConfig(cfg);
      setBb(session);
      setNeedsKey(false);
    } catch (err) {
      // A 401 asks for a key; anything else is a broken or unreachable backend
      // and must say so. Previously every failure landed in console.error and
      // the page sat on "Connecting to backend…" forever, which made an
      // unreachable backend and a wrong key look identical.
      if (err instanceof UnauthorizedError) setNeedsKey(true);
      else setConnectError(err instanceof Error ? err.message : String(err));
    }
  }, [title]);

  useEffect(() => {
    void connect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshReport = useCallback(async (id: string) => {
    try {
      setReport((await api.report(id)).markdown);
    } catch {
      /* report only available after verify */
    }
  }, []);

  async function socraticRevise(
    sectionId: string,
    paragraphIndex: number,
    message: string,
    history: SocraticTurn[],
    applyRevision: boolean
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
        applyRevision
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

  async function withBusy(
    key: string,
    fn: (onProgress: (event: JobEvent) => void) => Promise<Blackboard>
  ) {
    setBusy(key);
    // Cleared when the next step starts, not when this one ends: the record of
    // what just happened is most useful immediately after it happened.
    setProgress([]);
    try {
      const next = await fn((event) => setProgress((prior) => [...prior, event]));
      setBb(next);
      await refreshReport(next.session_id);
    } catch (e) {
      if (e instanceof UnauthorizedError) {
        setNeedsKey(true);
        return;
      }
      console.error(e);
      alert(String(e));
    } finally {
      setBusy(null);
    }
  }

  if (needsKey) {
    return (
      <div className="grid h-screen place-items-center bg-slate-50 px-4">
        <form
          className="w-full max-w-md rounded-xl border border-slate-200 bg-white p-6 shadow-sm"
          onSubmit={(e) => {
            e.preventDefault();
            setProxyPassword(keyInput);
            setKeyInput("");
            void connect();
          }}
        >
          <div className="mb-1 flex items-center gap-2">
            <ScrollText className="text-accent" size={20} />
            <h1 className="text-lg font-semibold text-slate-900">Password required</h1>
          </div>
          <p className="mb-4 text-sm text-slate-500">
            This site proxies to a private backend. The password authorises the proxy — it
            is not the backend&rsquo;s key, which stays on the server and never reaches
            your browser.
          </p>
          <input
            autoFocus
            type="password"
            value={keyInput}
            onChange={(e) => setKeyInput(e.target.value)}
            placeholder="PROXY_PASSWORD"
            className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-accent focus:outline-none"
          />
          <button
            type="submit"
            disabled={!keyInput.trim()}
            className="mt-3 w-full rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white hover:bg-violet-700 disabled:opacity-50"
          >
            Connect
          </button>
        </form>
      </div>
    );
  }

  if (connectError) {
    return (
      <div className="grid h-screen place-items-center bg-slate-50 px-4">
        <div className="w-full max-w-md rounded-xl border border-slate-200 bg-white p-6 text-center shadow-sm">
          <h1 className="mb-1 text-lg font-semibold text-slate-900">Cannot reach the backend</h1>
          <p className="mb-4 break-words text-sm text-slate-500">{connectError}</p>
          <button
            onClick={() => void connect()}
            className="rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white hover:bg-violet-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
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
              {config.embed_model.split("/").at(-1) ?? config.embed_model} · {config.citation_style} · target {config.manuscript_target_min_words.toLocaleString()}-
              {config.manuscript_target_max_words.toLocaleString()} words
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            onClick={() =>
              withBusy("run-all", async (onProgress) => {
                const { blackboard } = await runAll(
                  bb.session_id,
                  bb.thesis || title,
                  title,
                  undefined,
                  onProgress
                );
                return blackboard;
              })
            }
            disabled={busy !== null}
          >
            {busy === "run-all" ? <Loader2 className="animate-spin" size={14} /> : <PlayCircle size={14} />}
            Run full pipeline
          </Button>
          {/* Not an <a href>: a link cannot carry the key header, so a gated
              backend would answer the browser's unauthenticated request with a
              401 and the user would see a broken download with no explanation. */}
          <button
            onClick={async () => {
              try {
                const url = await api.downloadPdf(bb.session_id);
                window.open(url, "_blank", "noreferrer");
                // Give the new tab time to read it before the blob is dropped.
                setTimeout(() => URL.revokeObjectURL(url), 60_000);
              } catch (err) {
                if (err instanceof UnauthorizedError) setNeedsKey(true);
                else alert(err instanceof Error ? err.message : String(err));
              }
            }}
            className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-700"
          >
            <ScrollText size={14} /> Export
          </button>
        </div>
      </header>

      <div className="flex items-center gap-1.5 border-b border-slate-200 bg-white px-5 py-2">
        {STEPS.map((step) => (
          <Button
            key={step.key}
            variant="subtle"
            disabled={busy !== null}
            onClick={() =>
              withBusy(step.key, (onProgress) =>
                runStep(bb.session_id, step.key, undefined, 2000, onProgress)
              )
            }
          >
            {busy === step.key ? (
              <Loader2 className="animate-spin" size={14} />
            ) : (
              <step.icon size={14} />
            )}
            {step.label}
          </Button>
        ))}
      </div>

      {/* Only rendered when there is something to say. An empty strip that
          appears and disappears is noise; the absence of one is information. */}
      {progress.length > 0 && (
        <div className="border-b border-slate-200 bg-slate-50 px-5 py-2">
          <ol className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500">
            {progress.map((event, i) => (
              <li
                key={i}
                className={
                  event.degraded || event.answered === false
                    ? "text-amber-600"
                    : i === progress.length - 1 && busy !== null
                      ? "font-medium text-slate-700"
                      : ""
                }
              >
                {describeEvent(event)}
              </li>
            ))}
            {busy !== null && (
              <li className="flex items-center gap-1 text-slate-400">
                <Loader2 className="animate-spin" size={11} />
              </li>
            )}
          </ol>
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 p-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <div className="grid min-h-0 grid-rows-2 gap-4">
          <BrainstormPanel
            bb={bb}
            onRefresh={setBb}
            onIdeate={(seed) =>
              withBusy("ideate", async () => api.ideate(bb.session_id, seed || null))
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
            {(["manuscript", "citations", "jury"] as Tab[]).map((t) => (
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
            {tab === "manuscript" ? (
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
            ) : tab === "citations" ? (
              <CitationsPanel bb={bb} report={report} />
            ) : (
              <ModelJuryPanel
                modelMap={config.multi_chat_models}
                panelMembers={config.multi_chat_panel_members}
                verifierMap={config.multi_chat_verifiers}
                // As a job, not the synchronous route: the panel takes over two
                // minutes on this hardware and no HTTP request should be held
                // open that long. The progress callback is why the panel can
                // name each model as it lands instead of showing a spinner.
                onAsk={(message, history, onProgress) =>
                  askAsJob<MultiChatResponse>(
                    bb.session_id,
                    "multi-chat",
                    message,
                    history,
                    undefined,
                    (event) => onProgress(describeEvent(event))
                  )
                }
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
