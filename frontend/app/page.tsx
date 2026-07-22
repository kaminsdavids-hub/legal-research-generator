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
import { IdeaBoard } from "@/components/IdeaBoard";
import { ManuscriptPanel } from "@/components/ManuscriptPanel";
import { Button } from "@/components/ui";
import { api, type AppConfig, type Blackboard, type IdeaStatus } from "@/lib/api";

type Tab = "manuscript" | "citations";

const STEPS = [
  { key: "outline", label: "Outline", icon: ListTree, run: api.outline },
  { key: "research", label: "Research", icon: BookOpen, run: api.research },
  { key: "draft", label: "Draft", icon: PenLine, run: api.draft },
  { key: "voice", label: "Voice", icon: Sparkles, run: api.voice },
  { key: "verify", label: "Verify", icon: CheckCircle2, run: api.verify },
  { key: "format", label: "Format", icon: Quote, run: api.format },
  { key: "novelty", label: "Novelty", icon: FlaskConical, run: api.novelty },
] as const;

export default function Home() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [bb, setBb] = useState<Blackboard | null>(null);
  const [tab, setTab] = useState<Tab>("manuscript");
  const [report, setReport] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [title, setTitle] = useState("Untitled Research Paper");

  useEffect(() => {
    (async () => {
      const [cfg, session] = await Promise.all([api.config(), api.createSession(title)]);
      setConfig(cfg);
      setBb(session);
    })().catch(console.error);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshReport = useCallback(async (id: string) => {
    try {
      setReport((await api.report(id)).markdown);
    } catch {
      /* report only available after verify */
    }
  }, []);

  async function withBusy(key: string, fn: () => Promise<Blackboard>) {
    setBusy(key);
    try {
      const next = await fn();
      setBb(next);
      await refreshReport(next.session_id);
    } catch (e) {
      console.error(e);
      alert(String(e));
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
              {config.llm_mode} · {config.retriever_mode} retriever · {config.citation_style}
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
      </div>

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
            {(["manuscript", "citations"] as Tab[]).map((t) => (
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
                onRevise={(sectionId, instruction) =>
                  withBusy("revise", async () =>
                    api.revise(bb.session_id, sectionId, instruction)
                  )
                }
              />
            ) : (
              <CitationsPanel bb={bb} report={report} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
