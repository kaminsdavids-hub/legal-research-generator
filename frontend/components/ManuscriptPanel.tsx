"use client";

import { FileText, Wand2 } from "lucide-react";
import { useState } from "react";
import type { Blackboard } from "@/lib/api";
import { Badge, Card, Empty } from "./ui";

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

export function ManuscriptPanel({
  bb,
  onRevise,
}: {
  bb: Blackboard;
  onRevise: (sectionId: string, instruction: string) => void;
}) {
  const [active, setActive] = useState<string | null>(null);
  const [instruction, setInstruction] = useState("");
  const drafted = bb.outline.filter((s) => s.content);

  return (
    <Card title="Manuscript" icon={<FileText size={16} className="text-accent" />} className="h-full">
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
              <div className="mb-1 flex items-center justify-between">
                <h2 className="text-lg font-semibold">{section.title}</h2>
                <div className="flex items-center gap-2">
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
              <p className="text-[15px]">{renderMarkers(section.content)}</p>
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
            </div>
          ))}
        </article>
      )}
    </Card>
  );
}
