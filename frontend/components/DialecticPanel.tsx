"use client";

import { Check, Copy, Loader2, Scale, Send, ShieldAlert, ShieldCheck } from "lucide-react";
import { useMemo, useState } from "react";
import type { DialecticCrux, DialecticPosition, DialecticResponse, DialecticSlot } from "@/lib/api";
import { Button, Card, Empty } from "./ui";

const WEIGHT_STYLES: Record<string, string> = {
  controlling: "bg-violet-100 text-violet-700",
  persuasive: "bg-sky-100 text-sky-700",
  supporting: "bg-slate-100 text-slate-600",
  contra: "bg-rose-100 text-rose-700",
};

const PARTITION_STYLES: Record<string, string> = {
  "resolvable by authority": "bg-violet-50 border-violet-200 text-violet-800",
  "resolvable by fact": "bg-sky-50 border-sky-200 text-sky-800",
  open: "bg-amber-50 border-amber-200 text-amber-800",
};

/** True when the slot carries no verified authority and must render a marker. */
function isUnsupported(slot: DialecticSlot): boolean {
  return slot.status !== "verified";
}

function SlotRow({ slot, index }: { slot: DialecticSlot; index: number }) {
  const unsupported = isUnsupported(slot);
  return (
    <li className="rounded-md border border-slate-200 bg-white px-2.5 py-2">
      <div className="mb-1 flex items-center gap-2">
        <span className="text-[11px] font-semibold text-slate-400">{index + 1}</span>
        <span
          className={
            "rounded-full px-2 py-0.5 text-[10px] font-medium capitalize " +
            (WEIGHT_STYLES[slot.weight] ?? "bg-slate-100 text-slate-600")
          }
        >
          {slot.weight}
        </span>
        {unsupported ? (
          <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-amber-700">
            <ShieldAlert size={11} /> UNSUPPORTED
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-700">
            <ShieldCheck size={11} /> {slot.normalized_cite || "verified"}
          </span>
        )}
      </div>
      <div className="text-sm text-slate-700">{slot.proposition}</div>
      {unsupported && (
        // The marker text is rendered verbatim so a copied screenshot or manual
        // transcription carries the same warning the serializers emit.
        <div className="mt-1 font-mono text-[11px] text-amber-700">
          [UNSUPPORTED: {slot.weight} authority for {slot.proposition}]
        </div>
      )}
      {slot.court_hint && (
        <div className="mt-1 text-[11px] text-slate-400">hint: {slot.court_hint}</div>
      )}
    </li>
  );
}

function PositionBlock({
  position,
  label,
  onCopy,
  copied,
}: {
  position: DialecticPosition;
  label: string;
  onCopy: () => void;
  copied: boolean;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          {label}
          <span className="ml-2 font-normal normal-case text-slate-400">
            {position.model} ({position.family})
          </span>
        </div>
        <button
          type="button"
          onClick={onCopy}
          className="inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-[10px] font-medium text-slate-500 hover:bg-slate-100"
        >
          {copied ? <Check size={11} /> : <Copy size={11} />}
          {copied ? "Copied" : "Copy side"}
        </button>
      </div>
      <ul className="space-y-2">
        {position.propositions.map((slot, i) => (
          <SlotRow key={i} slot={slot} index={i} />
        ))}
      </ul>
    </div>
  );
}

function CruxRow({ crux, index }: { crux: DialecticCrux; index: number }) {
  return (
    <div
      className={
        "rounded-lg border p-2.5 " +
        (PARTITION_STYLES[crux.partition] ?? "border-slate-200 bg-white text-slate-700")
      }
    >
      <div className="mb-1.5 flex items-center justify-between text-[11px] font-semibold uppercase tracking-wide">
        <span>
          {index + 1}. {crux.partition || "unclassified"}
          {!crux.outcome_bearing && (
            <span
              className="ml-1.5 font-normal normal-case text-slate-500"
              title="Neither side carries controlling or persuasive weight, so this contradiction is not resolvable by authority."
            >
              not outcome-bearing
            </span>
          )}
        </span>
        <span className="font-normal normal-case">
          {crux.nli_source === "heuristic" && (
            <span
              className="mr-2 text-amber-700"
              title="The NLI model was unavailable or its answer did not parse, so the offline heuristic classified this pair."
            >
              heuristic
            </span>
          )}
          winner: {crux.winner === "none" ? "—" : crux.winner}
        </span>
      </div>
      <div className="space-y-1 text-xs">
        <div>
          <span className="font-semibold">Thesis:</span> {crux.thesis_prop.proposition}
          {isUnsupported(crux.thesis_prop) && (
            <span className="ml-1 font-mono text-[10px] text-amber-700">
              [UNSUPPORTED: {crux.thesis_prop.weight}]
            </span>
          )}
        </div>
        <div>
          <span className="font-semibold">Antithesis:</span> {crux.antithesis_prop.proposition}
          {isUnsupported(crux.antithesis_prop) && (
            <span className="ml-1 font-mono text-[10px] text-amber-700">
              [UNSUPPORTED: {crux.antithesis_prop.weight}]
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export function DialecticPanel({
  modelMap,
  onAsk,
}: {
  modelMap?: Record<string, string>;
  onAsk: (
    message: string,
    onProgress: (line: string) => void
  ) => Promise<DialecticResponse>;
}) {
  const [draft, setDraft] = useState("");
  const [latest, setLatest] = useState<DialecticResponse | null>(null);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState("");
  const [copiedKey, setCopiedKey] = useState("");
  const [busy, setBusy] = useState(false);

  const modelSummary = useMemo(
    () => Object.entries(modelMap ?? {}).map(([role, model]) => `${role}: ${model}`),
    [modelMap]
  );

  async function send() {
    const text = draft.trim();
    if (!text || busy) return;
    setBusy(true);
    setError("");
    setProgress("");
    try {
      setLatest(await onAsk(text, setProgress));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setProgress("");
    }
  }

  async function copyText(key: string, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      setTimeout(() => setCopiedKey((c) => (c === key ? "" : c)), 1400);
    } catch {
      setError("Unable to copy text from this browser context.");
    }
  }

  return (
    <Card
      title="Dialectic Chat"
      icon={<Scale size={16} className="text-accent" />}
      actions={
        <div className="hidden max-w-[36rem] text-right text-[11px] text-slate-400 xl:block">
          {modelSummary.join(" · ")}
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

          {!latest && !busy && (
            <Empty>
              Ask one question. Thesis argues yes and antithesis argues no, from
              distinct model families; a fourth model derives the cruxes. Cruxes
              are not filtered by weight — one between two supporting propositions
              is still a contradiction, just not resolvable by authority. Models
              never emit citations: authority is proposed by retrieval and
              confirmed by lookup, and anything unconfirmed stays flagged.
            </Empty>
          )}

          {busy && (
            <div className="rounded-lg bg-slate-100 px-3 py-2 text-sm text-slate-600">
              <Loader2 className="inline animate-spin" size={14} />{" "}
              {/* The stage name, not a spinner: these stages are heterogeneous --
                  a model call, a CourtListener round-trip, an NLI pass -- so
                  "still going" cannot distinguish a slow debate from a hung one,
                  and which stage it is in is the useful part. */}
              {progress || "Running thesis, antithesis, synthesis, and the NLI crux pass…"}
            </div>
          )}

          {latest && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium text-slate-700">{latest.question}</div>
                <button
                  type="button"
                  onClick={() => void copyText("exchange", latest.copy_exchange)}
                  className="inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-2 py-1 text-[11px] font-medium text-slate-600 hover:bg-slate-50"
                >
                  {copiedKey === "exchange" ? <Check size={12} /> : <Copy size={12} />}
                  {copiedKey === "exchange" ? "Copied" : "Copy exchange"}
                </button>
              </div>

              <PositionBlock
                position={latest.thesis}
                label="Thesis"
                copied={copiedKey === "thesis"}
                onCopy={() => void copyText("thesis", latest.copy_thesis)}
              />
              <PositionBlock
                position={latest.antithesis}
                label="Antithesis"
                copied={copiedKey === "antithesis"}
                onCopy={() => void copyText("antithesis", latest.copy_antithesis)}
              />

              {latest.synthesis && (
                <div className="rounded-lg border border-slate-200 bg-white p-3">
                  <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Synthesis
                  </div>
                  <div className="whitespace-pre-wrap text-sm text-slate-700">
                    {latest.synthesis}
                  </div>
                </div>
              )}

              <div className="rounded-lg border border-slate-200 bg-white p-3">
                <div className="mb-2 flex items-center justify-between">
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Crux table ({latest.cruxes.length})
                  </div>
                  <button
                    type="button"
                    onClick={() => void copyText("cruxes", latest.copy_crux_table)}
                    className="inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-[10px] font-medium text-slate-500 hover:bg-slate-50"
                  >
                    {copiedKey === "cruxes" ? <Check size={11} /> : <Copy size={11} />}
                    {copiedKey === "cruxes" ? "Copied" : "Copy cruxes"}
                  </button>
                </div>
                {latest.cruxes.length === 0 ? (
                  <div className="text-xs text-slate-400">
                    {/* Show the server's stated reason (how many pairs were compared)
                        rather than guessing at one. An empty table with no explanation
                        is indistinguishable from a broken extractor. */}
                    {latest.crux_note || "No contradiction found between the two positions."}
                  </div>
                ) : (
                  <div className="space-y-2">
                    {latest.cruxes.map((crux, i) => (
                      <CruxRow key={i} crux={crux} index={i} />
                    ))}
                  </div>
                )}
              </div>

              <div className="text-[11px] text-slate-400">
                CourtListener calls spent this exchange: {latest.calls_spent}
                {latest.regenerated > 0 && (
                  <span
                    className="ml-2 text-amber-600"
                    title="A turn that burned regeneration attempts is otherwise indistinguishable from a clean first pass. The per-slot note says why."
                  >
                    · regeneration attempts spent: {latest.regenerated}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="mt-3 flex items-center gap-2 border-t border-slate-100 pt-3">
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            placeholder="Pose one contested legal question…"
            className="flex-1 rounded-lg border border-slate-300 px-3 py-1.5 text-sm focus:border-accent focus:outline-none"
          />
          <Button onClick={() => void send()} disabled={busy || !draft.trim()}>
            {busy ? <Loader2 className="animate-spin" size={14} /> : <Send size={14} />}
            Argue
          </Button>
        </div>
      </div>
    </Card>
  );
}
