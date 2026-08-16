"use client";

import { Check, Copy, FileText, Loader2, MessageCircle, Wand2 } from "lucide-react";
import { useRef, useState } from "react";
import type { Blackboard, SocraticMode, SocraticReviseResponse, SocraticTurn } from "@/lib/api";
import { Badge, Card, Empty } from "./ui";

type BubbleTurn = SocraticTurn & {
  cycleBreakTriggered?: boolean;
  noveltyScore?: number;
  rewriteDeltaScore?: number;
};

const SOCRATIC_MODE_OPTIONS: { value: SocraticMode; label: string }[] = [
  { value: "strengthen_doctrine", label: "Doctrine" },
  { value: "expand_analysis", label: "Expand" },
  { value: "counter_rebuttal", label: "Counter/Rebuttal" },
  { value: "policy_implications", label: "Policy" },
  { value: "comparative_framework", label: "Comparative" },
];

const SOCRATIC_MODE_PROMPTS: Record<SocraticMode, string> = {
  strengthen_doctrine:
    "Identify the controlling legal test and rewrite this paragraph so each element is explicitly applied to the argument.",
  expand_analysis:
    "Expand this paragraph into a fuller substantive analysis with more doctrinal mechanics, intermediate reasoning steps, and explicit legal stakes.",
  counter_rebuttal:
    "Integrate the strongest counterargument and then answer it with a precise, narrow rebuttal grounded in legal reasoning.",
  policy_implications:
    "Develop the policy and institutional implications with concrete consequences and why they matter doctrinally.",
  comparative_framework:
    "Compare at least two plausible legal frameworks and justify why this paragraph should adopt one over the other.",
};

function renderMarkers(text: string) {
  // Turn [^n] footnote markers into superscripts for display.
  const parts = text.split(/(\[\^\d+\])/g);
  return parts.map((p, i) => {
    const m = p.match(/^\[\^(\d+)\]$/);
    return m ? (
      <sup key={i} className="text-accent">
        {m[1]}
      </sup>
    ) : (
      <span key={i}>{p}</span>
    );
  });
}

function splitParagraphs(text: string) {
  return text
    .split(/\n\s*\n/g)
    .map((p) => p.trim())
    .filter(Boolean);
}

export function ManuscriptPanel({
  bb,
  onRevise,
  onSocratic,
  socraticBusy = false,
}: {
  bb: Blackboard;
  onRevise: (sectionId: string, instruction: string) => void | Promise<void>;
  onSocratic: (
    sectionId: string,
    paragraphIndex: number,
    message: string,
    history: SocraticTurn[],
    applyRevision: boolean,
    mode: SocraticMode
  ) => Promise<SocraticReviseResponse>;
  socraticBusy?: boolean;
}) {
  const [active, setActive] = useState<string | null>(null);
  const [instruction, setInstruction] = useState("");
  const [socraticEnabled, setSocraticEnabled] = useState(false);
  const [activeBubble, setActiveBubble] = useState<string | null>(null);
  const [bubbleInput, setBubbleInput] = useState<Record<string, string>>({});
  const [bubbleHistory, setBubbleHistory] = useState<Record<string, BubbleTurn[]>>({});
  const [bubbleLoading, setBubbleLoading] = useState<Record<string, boolean>>({});
  const [bubbleMode, setBubbleMode] = useState<Record<string, SocraticMode>>({});
  const [sectionMode, setSectionMode] = useState<Record<string, SocraticMode>>({});
  const [expandingSection, setExpandingSection] = useState<string | null>(null);
  const [expandProgress, setExpandProgress] = useState<Record<string, { done: number; total: number }>>({});
  const [expandAllRunning, setExpandAllRunning] = useState(false);
  const [expandAllProgress, setExpandAllProgress] = useState<{
    done: number;
    total: number;
    sectionTitle: string;
  } | null>(null);
  const [planOpen, setPlanOpen] = useState(false);
  const [planSelected, setPlanSelected] = useState<Record<string, boolean>>({});
  const [cancelExpansionRequested, setCancelExpansionRequested] = useState(false);
  const cancelExpansionRef = useRef(false);
  const [copiedKey, setCopiedKey] = useState("");
  const drafted = bb.outline.filter((s) => s.content);
  const isAnyExpansionRunning = expandingSection !== null || expandAllRunning;

  const bubbleKey = (sectionId: string, paragraphIndex: number) =>
    `${sectionId}:${paragraphIndex}`;

  async function runSocraticTurn(
    sectionId: string,
    paragraphIndex: number,
    mode: SocraticMode,
    applyRevision: boolean,
    messageOverride?: string
  ) {
    const key = bubbleKey(sectionId, paragraphIndex);
    const prompt = (messageOverride ?? bubbleInput[key] ?? "").trim() || SOCRATIC_MODE_PROMPTS[mode];
    if (!prompt) return;
    const history = bubbleHistory[key] ?? [];
    const withUser: BubbleTurn[] = [...history, { role: "user", content: prompt }];
    setBubbleHistory((prev) => ({ ...prev, [key]: withUser }));
    if (!messageOverride) {
      setBubbleInput((prev) => ({ ...prev, [key]: "" }));
    }
    setBubbleLoading((prev) => ({ ...prev, [key]: true }));
    try {
      const response = await onSocratic(sectionId, paragraphIndex, prompt, history, applyRevision, mode);
      const assistantTurns: BubbleTurn[] = [
        {
          role: "assistant",
          content: response.assistant,
          cycleBreakTriggered: response.cycle_break_triggered,
          noveltyScore: response.novelty_score,
          rewriteDeltaScore: response.rewrite_delta_score,
        },
      ];
      if (response.applied && response.suggested_revision) {
        assistantTurns.push({
          role: "assistant",
          content: `Applied revision:\n${response.suggested_revision}`,
          cycleBreakTriggered: response.cycle_break_triggered,
          noveltyScore: response.novelty_score,
          rewriteDeltaScore: response.rewrite_delta_score,
        });
      }
      setBubbleHistory((prev) => ({
        ...prev,
        [key]: [...withUser, ...assistantTurns],
      }));
    } catch (e) {
      setBubbleHistory((prev) => ({
        ...prev,
        [key]: [
          ...withUser,
          { role: "assistant", content: `Could not process Socratic revision: ${String(e)}` },
        ],
      }));
    } finally {
      setBubbleLoading((prev) => ({ ...prev, [key]: false }));
    }
  }

  async function sendBubble(
    sectionId: string,
    paragraphIndex: number,
    applyRevision: boolean,
    messageOverride?: string
  ) {
    const key = bubbleKey(sectionId, paragraphIndex);
    const mode = bubbleMode[key] ?? "strengthen_doctrine";
    await runSocraticTurn(sectionId, paragraphIndex, mode, applyRevision, messageOverride);
  }

  async function expandWholeSection(sectionId: string, paragraphCount: number) {
    if (isAnyExpansionRunning) {
      return;
    }
    const total = Math.max(1, paragraphCount);
    const mode = sectionMode[sectionId] ?? "expand_analysis";
    cancelExpansionRef.current = false;
    setCancelExpansionRequested(false);
    setExpandingSection(sectionId);
    setExpandProgress((prev) => ({ ...prev, [sectionId]: { done: 0, total } }));
    try {
      for (let idx = 0; idx < total; idx += 1) {
        if (cancelExpansionRef.current) {
          break;
        }
        await runSocraticTurn(
          sectionId,
          idx,
          mode,
          true,
          SOCRATIC_MODE_PROMPTS[mode]
        );
        setExpandProgress((prev) => ({
          ...prev,
          [sectionId]: { done: idx + 1, total },
        }));
        if (cancelExpansionRef.current) {
          break;
        }
      }
    } finally {
      setExpandingSection((current) => (current === sectionId ? null : current));
      cancelExpansionRef.current = false;
      setCancelExpansionRequested(false);
    }
  }

  function openExpansionPlan() {
    const next: Record<string, boolean> = {};
    for (const section of drafted) {
      next[section.id] = planSelected[section.id] ?? true;
    }
    setPlanSelected(next);
    setPlanOpen(true);
  }

  function runExpansionPlan() {
    const selectedIds = drafted
      .map((section) => section.id)
      .filter((sectionId) => Boolean(planSelected[sectionId]));
    if (selectedIds.length === 0) {
      return;
    }
    setPlanOpen(false);
    void expandSectionsByPlan(selectedIds);
  }

  function runExpansionPlanLegalBriefPreset() {
    const selectedIds = drafted
      .map((section) => section.id)
      .filter((sectionId) => Boolean(planSelected[sectionId]));
    if (selectedIds.length === 0) {
      return;
    }
    const preset: Record<string, SocraticMode[]> = {};
    for (const sectionId of selectedIds) {
      preset[sectionId] = ["expand_analysis", "counter_rebuttal"];
    }
    setPlanOpen(false);
    void expandSectionsByPlan(selectedIds, preset);
  }

  function runExpansionPlanPolicyHeavyPreset() {
    const selectedIds = drafted
      .map((section) => section.id)
      .filter((sectionId) => Boolean(planSelected[sectionId]));
    if (selectedIds.length === 0) {
      return;
    }
    const preset: Record<string, SocraticMode[]> = {};
    for (const sectionId of selectedIds) {
      preset[sectionId] = ["expand_analysis", "policy_implications", "counter_rebuttal"];
    }
    setPlanOpen(false);
    void expandSectionsByPlan(selectedIds, preset);
  }

  async function expandAllSections() {
    await expandSectionsByPlan(drafted.map((section) => section.id));
  }

  async function expandSectionsByPlan(
    sectionIds: string[],
    modePlan?: Record<string, SocraticMode[]>
  ) {
    if (isAnyExpansionRunning || drafted.length === 0 || sectionIds.length === 0) {
      return;
    }

    const idSet = new Set(sectionIds);
    const plan = drafted
      .filter((section) => idSet.has(section.id))
      .map((section) => ({
        id: section.id,
        title: section.title,
        paragraphCount: Math.max(1, splitParagraphs(section.content).length),
        modes:
          modePlan?.[section.id]?.length
            ? modePlan[section.id]
            : [sectionMode[section.id] ?? "expand_analysis"],
      }));
    if (plan.length === 0) {
      return;
    }
    const total = plan.reduce((sum, item) => sum + item.paragraphCount * item.modes.length, 0);

    cancelExpansionRef.current = false;
    setCancelExpansionRequested(false);
    setExpandAllRunning(true);
    setExpandAllProgress({ done: 0, total, sectionTitle: plan[0]?.title ?? "" });

    let done = 0;
    try {
      outer: for (const item of plan) {
        const sectionTotal = item.paragraphCount * item.modes.length;
        let sectionDone = 0;
        setExpandingSection(item.id);
        setExpandProgress((prev) => ({
          ...prev,
          [item.id]: { done: 0, total: sectionTotal },
        }));

        for (const mode of item.modes) {
          for (let idx = 0; idx < item.paragraphCount; idx += 1) {
            if (cancelExpansionRef.current) {
              break outer;
            }
            await runSocraticTurn(
              item.id,
              idx,
              mode,
              true,
              SOCRATIC_MODE_PROMPTS[mode]
            );
            done += 1;
            sectionDone += 1;
            setExpandProgress((prev) => ({
              ...prev,
              [item.id]: { done: sectionDone, total: sectionTotal },
            }));
            setExpandAllProgress({
              done,
              total,
              sectionTitle: item.title,
            });
          }
        }
      }
    } finally {
      setExpandAllRunning(false);
      setExpandingSection(null);
      cancelExpansionRef.current = false;
      setCancelExpansionRequested(false);
    }
  }

  function stopExpansion() {
    cancelExpansionRef.current = true;
    setCancelExpansionRequested(true);
  }

  async function copyText(key: string, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      setTimeout(() => {
        setCopiedKey((current) => (current === key ? "" : current));
      }, 1400);
    } catch {
      // Keep this silent; Socratic bubbles already surface core model errors inline.
    }
  }

  return (
    <Card
      title="Manuscript"
      icon={<FileText size={16} className="text-accent" />}
      actions={
        <div className="flex items-center gap-2">
          {socraticEnabled && (
            <>
              <button
                type="button"
                onClick={() => void expandAllSections()}
                disabled={isAnyExpansionRunning || socraticBusy}
                className="inline-flex items-center gap-1 rounded-md bg-indigo-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
                title="Apply Socratic expansion rewrite across all sections"
              >
                {expandAllRunning ? <Loader2 size={12} className="animate-spin" /> : null}
                {expandAllRunning && expandAllProgress
                  ? `Expanding all ${expandAllProgress.done}/${expandAllProgress.total}`
                  : "Expand all sections"}
              </button>
              <button
                type="button"
                onClick={openExpansionPlan}
                disabled={isAnyExpansionRunning || socraticBusy || drafted.length === 0}
                className="rounded-md bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-200 disabled:opacity-50"
                title="Choose sections and modes before running expansion"
              >
                Expansion plan
              </button>
              {isAnyExpansionRunning && (
                <button
                  type="button"
                  onClick={stopExpansion}
                  disabled={cancelExpansionRequested}
                  className="rounded-md bg-rose-100 px-2.5 py-1 text-xs font-medium text-rose-700 hover:bg-rose-200 disabled:opacity-50"
                >
                  {cancelExpansionRequested ? "Stopping..." : "Stop expansion"}
                </button>
              )}
            </>
          )}
          <button
            onClick={() => setSocraticEnabled((v) => !v)}
            className={
              "rounded-md px-2.5 py-1 text-xs font-medium transition " +
              (socraticEnabled
                ? "bg-emerald-100 text-emerald-700 hover:bg-emerald-200"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200")
            }
            title="Toggle Socratic paragraph bubbles"
          >
            {socraticEnabled ? "Socratic bubbles on" : "Socratic bubbles off"}
          </button>
        </div>
      }
      className="h-full"
    >
      {drafted.length === 0 ? (
        <Empty>No draft yet. Build the outline, then run Research and Draft.</Empty>
      ) : (
        <article className="space-y-6 font-serif leading-relaxed text-slate-800">
          <h1 className="text-2xl font-bold">{bb.title}</h1>
          {bb.thesis && (
            <p className="rounded-lg bg-slate-50 p-3 text-sm">
              <span className="font-semibold">Thesis. </span>
              {bb.thesis}
            </p>
          )}
          {drafted.map((section) => (
            <div key={section.id} className="group">
              {(() => {
                const sectionParagraphs = splitParagraphs(section.content);
                const progress = expandProgress[section.id];
                const isExpanding = expandingSection === section.id;
                return (
                  <>
              <div className="mb-1 flex items-center justify-between">
                <h2 className="text-lg font-semibold">{section.title}</h2>
                <div className="flex items-center gap-2">
                  {socraticEnabled && (
                    <>
                      <select
                        value={sectionMode[section.id] ?? "expand_analysis"}
                        onChange={(e) =>
                          setSectionMode((prev) => ({
                            ...prev,
                            [section.id]: e.target.value as SocraticMode,
                          }))
                        }
                        disabled={isAnyExpansionRunning || socraticBusy}
                        className="rounded-md border border-slate-300 bg-white px-2 py-1 text-[11px] text-slate-600"
                        title="Expansion mode for whole-section Socratic rewrite"
                      >
                        {SOCRATIC_MODE_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      <button
                        type="button"
                        onClick={() => void expandWholeSection(section.id, sectionParagraphs.length || 1)}
                        disabled={isAnyExpansionRunning || socraticBusy}
                        className="inline-flex items-center gap-1 rounded-md bg-indigo-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
                        title="Apply Socratic expansion rewrite across every paragraph in this section"
                      >
                        {isExpanding ? <Loader2 size={12} className="animate-spin" /> : null}
                        {isExpanding
                          ? `Expanding ${progress?.done ?? 0}/${(progress?.total ?? sectionParagraphs.length) || 1}`
                          : "Expand whole section"}
                      </button>
                    </>
                  )}
                  <Badge value={section.status} />
                  <button
                    onClick={() => setActive(active === section.id ? null : section.id)}
                    className="rounded-md p-1 text-slate-400 opacity-0 transition group-hover:opacity-100 hover:bg-slate-100 hover:text-accent"
                    title="Revise this section"
                  >
                    <Wand2 size={15} />
                  </button>
                </div>
              </div>
              <div className="space-y-3">
                {sectionParagraphs.map((paragraph, idx) => {
                  const key = bubbleKey(section.id, idx);
                  const turns = bubbleHistory[key] ?? [];
                  const loading = bubbleLoading[key] ?? false;
                  return (
                    <div key={key} className="rounded-md border border-transparent p-1.5 hover:border-slate-200">
                      <p className="text-[15px]">{renderMarkers(paragraph)}</p>
                      {socraticEnabled && (
                        <div className="mt-2">
                          <button
                            onClick={() => setActiveBubble(activeBubble === key ? null : key)}
                            className="inline-flex items-center gap-1 rounded-md bg-slate-100 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-200"
                            title="Open Socratic paragraph development bubble"
                          >
                            <MessageCircle size={12} />
                            Paragraph bubble
                          </button>
                          {activeBubble === key && (
                            <div className="mt-2 rounded-lg border border-slate-200 bg-slate-50 p-2.5">
                              <div className="mb-2 max-h-48 space-y-1.5 overflow-auto pr-1 text-xs">
                                {turns.length === 0 ? (
                                  <div className="rounded bg-white p-2 text-slate-500">
                                    Ask for doctrinal or structural coaching on this paragraph.
                                  </div>
                                ) : (
                                  turns.map((turn, i) => (
                                    <div
                                      key={i}
                                      className={
                                        "rounded p-2 " +
                                        (turn.role === "user"
                                          ? "ml-6 bg-accent/10 text-slate-700"
                                          : "mr-6 bg-white text-slate-700")
                                      }
                                    >
                                      <div className="mb-0.5 flex items-center justify-between text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                                        <span className="flex items-center gap-1.5">
                                          <span>{turn.role}</span>
                                          {turn.role === "assistant" && typeof turn.noveltyScore === "number" && (
                                            <span className="rounded bg-sky-100 px-1.5 py-0.5 text-[10px] font-semibold normal-case tracking-normal text-sky-700">
                                              novelty {(turn.noveltyScore * 100).toFixed(0)}%
                                            </span>
                                          )}
                                          {turn.role === "assistant" && turn.cycleBreakTriggered && (
                                            <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold normal-case tracking-normal text-amber-700">
                                              cycle-break
                                            </span>
                                          )}
                                          {turn.role === "assistant" && typeof turn.rewriteDeltaScore === "number" && turn.content.startsWith("Applied revision:") && (
                                            <span className="rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-semibold normal-case tracking-normal text-emerald-700">
                                              delta {(turn.rewriteDeltaScore * 100).toFixed(0)}%
                                            </span>
                                          )}
                                        </span>
                                        <button
                                          type="button"
                                          onClick={() => void copyText(`socratic-${key}-${i}`, turn.content)}
                                          className="inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-[10px] normal-case text-slate-500 hover:bg-slate-50"
                                        >
                                          {copiedKey === `socratic-${key}-${i}` ? <Check size={11} /> : <Copy size={11} />}
                                          {copiedKey === `socratic-${key}-${i}` ? "Copied" : "Copy"}
                                        </button>
                                      </div>
                                      <div className="whitespace-pre-wrap">{turn.content}</div>
                                    </div>
                                  ))
                                )}
                              </div>
                              <div className="flex flex-col gap-2">
                                <textarea
                                  value={bubbleInput[key] ?? ""}
                                  onChange={(e) =>
                                    setBubbleInput((prev) => ({ ...prev, [key]: e.target.value }))
                                  }
                                  rows={2}
                                  placeholder="Ask a Socratic coaching question for this paragraph..."
                                  className="w-full resize-none rounded-md border border-slate-300 px-2.5 py-1.5 text-xs focus:border-accent focus:outline-none"
                                />
                                <div className="flex flex-wrap items-center gap-2">
                                  <label className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                                    Mode
                                  </label>
                                  <select
                                    value={bubbleMode[key] ?? "strengthen_doctrine"}
                                    onChange={(e) =>
                                      setBubbleMode((prev) => ({
                                        ...prev,
                                        [key]: e.target.value as SocraticMode,
                                      }))
                                    }
                                    className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700"
                                  >
                                    {SOCRATIC_MODE_OPTIONS.map((option) => (
                                      <option key={option.value} value={option.value}>
                                        {option.label}
                                      </option>
                                    ))}
                                  </select>
                                  <button
                                    disabled={loading || socraticBusy}
                                    onClick={() =>
                                      sendBubble(
                                        section.id,
                                        idx,
                                        true,
                                        SOCRATIC_MODE_PROMPTS[bubbleMode[key] ?? "strengthen_doctrine"]
                                      )
                                    }
                                    className="rounded-md bg-indigo-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
                                  >
                                    One-click sophisticated rewrite
                                  </button>
                                </div>
                                <div className="flex gap-2">
                                  <button
                                    disabled={loading || socraticBusy}
                                    onClick={() => sendBubble(section.id, idx, false)}
                                    className="rounded-md bg-slate-200 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-300 disabled:opacity-50"
                                  >
                                    {loading ? "Thinking..." : "Ask coach"}
                                  </button>
                                  <button
                                    disabled={loading || socraticBusy}
                                    onClick={() => sendBubble(section.id, idx, true)}
                                    className="rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-white hover:bg-violet-700 disabled:opacity-50"
                                  >
                                    Apply substantive revision
                                  </button>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
              {bb.footnotes[section.id]?.length > 0 && (
                <ol className="mt-2 space-y-1 border-t border-slate-100 pt-2 text-xs text-slate-500">
                  {bb.footnotes[section.id].map((fn) => (
                    <li key={fn.number}>
                      <span className="font-semibold">{fn.number}.</span> {fn.text}
                    </li>
                  ))}
                </ol>
              )}
              {active === section.id && (
                <div className="mt-2 flex gap-2">
                  <input
                    value={instruction}
                    onChange={(e) => setInstruction(e.target.value)}
                    placeholder="Revision instruction (e.g. tighten, add counterpoint)"
                    className="flex-1 rounded-lg border border-slate-300 px-3 py-1.5 text-sm focus:border-accent focus:outline-none"
                  />
                  <button
                    onClick={() => {
                      if (instruction.trim()) {
                        onRevise(section.id, instruction.trim());
                        setInstruction("");
                        setActive(null);
                      }
                    }}
                    className="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-700"
                  >
                    Apply
                  </button>
                </div>
              )}
                  </>
                );
              })()}
            </div>
          ))}
        </article>
      )}
      {planOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4">
          <div className="w-full max-w-2xl rounded-xl border border-slate-200 bg-white shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
              <div>
                <div className="text-sm font-semibold text-slate-800">Expansion plan</div>
                <div className="text-xs text-slate-500">
                  Choose which sections to expand and tune a Socratic mode for each.
                </div>
              </div>
              <button
                type="button"
                onClick={() => setPlanOpen(false)}
                className="rounded-md px-2 py-1 text-xs text-slate-500 hover:bg-slate-100"
              >
                Close
              </button>
            </div>
            <div className="max-h-[65vh] space-y-2 overflow-auto p-4">
              <div className="mb-2 flex gap-2">
                <button
                  type="button"
                  onClick={() =>
                    setPlanSelected(
                      Object.fromEntries(drafted.map((section) => [section.id, true]))
                    )
                  }
                  className="rounded-md bg-slate-100 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-200"
                >
                  Select all
                </button>
                <button
                  type="button"
                  onClick={() =>
                    setPlanSelected(
                      Object.fromEntries(drafted.map((section) => [section.id, false]))
                    )
                  }
                  className="rounded-md bg-slate-100 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-200"
                >
                  Clear all
                </button>
              </div>
              {drafted.map((section) => {
                const paragraphCount = Math.max(1, splitParagraphs(section.content).length);
                return (
                  <div
                    key={`plan-${section.id}`}
                    className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2"
                  >
                    <label className="flex min-w-0 items-center gap-2">
                      <input
                        type="checkbox"
                        checked={Boolean(planSelected[section.id])}
                        onChange={(e) =>
                          setPlanSelected((prev) => ({
                            ...prev,
                            [section.id]: e.target.checked,
                          }))
                        }
                        className="h-4 w-4 rounded border-slate-300"
                      />
                      <span className="truncate text-sm font-medium text-slate-700">{section.title}</span>
                      <span className="text-xs text-slate-500">{paragraphCount} paragraphs</span>
                    </label>
                    <select
                      value={sectionMode[section.id] ?? "expand_analysis"}
                      onChange={(e) =>
                        setSectionMode((prev) => ({
                          ...prev,
                          [section.id]: e.target.value as SocraticMode,
                        }))
                      }
                      className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700"
                    >
                      {SOCRATIC_MODE_OPTIONS.map((option) => (
                        <option key={`plan-mode-${section.id}-${option.value}`} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </div>
                );
              })}
            </div>
            <div className="flex items-center justify-between border-t border-slate-100 px-4 py-3">
              <div className="text-xs text-slate-500">
                {drafted.filter((section) => Boolean(planSelected[section.id])).length} selected
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setPlanOpen(false)}
                  className="rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={runExpansionPlanLegalBriefPreset}
                  disabled={drafted.every((section) => !planSelected[section.id])}
                  className="rounded-md bg-violet-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-violet-700 disabled:opacity-50"
                  title="Runs Expand Analysis then Counter/Rebuttal for each selected section"
                >
                  Run legal brief preset
                </button>
                <button
                  type="button"
                  onClick={runExpansionPlanPolicyHeavyPreset}
                  disabled={drafted.every((section) => !planSelected[section.id])}
                  className="rounded-md bg-fuchsia-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-fuchsia-700 disabled:opacity-50"
                  title="Runs Expand Analysis, Policy Implications, then Counter/Rebuttal for each selected section"
                >
                  Run policy-heavy preset
                </button>
                <button
                  type="button"
                  onClick={runExpansionPlan}
                  disabled={drafted.every((section) => !planSelected[section.id])}
                  className="rounded-md bg-indigo-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
                >
                  Run selected expansion
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
