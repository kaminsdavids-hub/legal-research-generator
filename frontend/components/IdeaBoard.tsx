"use client";

import { Check, Lightbulb, X } from "lucide-react";
import type { Blackboard, IdeaStatus } from "@/lib/api";
import { Badge, Card, Empty } from "./ui";

export function IdeaBoard({
  bb,
  onUpdate,
}: {
  bb: Blackboard;
  onUpdate: (ideaId: string, status: IdeaStatus, priority?: number) => void;
}) {
  return (
    <Card title="Idea Board" icon={<Lightbulb size={16} className="text-accent" />} className="h-full">
      {bb.ideas.length === 0 ? (
        <Empty>No ideas yet. Use “Generate ideas” in the brainstorm panel.</Empty>
      ) : (
        <ul className="space-y-2">
          {bb.ideas.map((idea, i) => (
            <li
              key={idea.id}
              className="rounded-lg border border-slate-200 p-3 transition hover:border-slate-300"
            >
              <div className="flex items-start justify-between gap-3">
                <p className="text-sm text-slate-700">{idea.text}</p>
                <Badge value={idea.status} />
              </div>
              <div className="mt-2 flex items-center justify-between">
                <span className="text-xs text-slate-400">{idea.novelty_note}</span>
                <div className="flex gap-1">
                  <button
                    onClick={() => onUpdate(idea.id, "keep", i)}
                    className="rounded-md p-1 text-emerald-600 hover:bg-emerald-50"
                    title="Keep"
                  >
                    <Check size={16} />
                  </button>
                  <button
                    onClick={() => onUpdate(idea.id, "cut")}
                    className="rounded-md p-1 text-rose-600 hover:bg-rose-50"
                    title="Cut"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
