"use client";

import { ShieldCheck } from "lucide-react";
import type { Blackboard } from "@/lib/api";
import { Badge, Card, Empty } from "./ui";

export function CitationsPanel({ bb, report }: { bb: Blackboard; report: string }) {
  const verified = bb.citations.filter((c) => c.status === "verified").length;
  const removed = bb.citations.filter((c) => c.status === "removed").length;

  return (
    <Card
      title="Citations & Verification"
      icon={<ShieldCheck size={16} className="text-accent" />}
      actions={
        <span className="text-xs text-slate-500">
          {verified} verified · {removed} removed
        </span>
      }
      className="h-full"
    >
      {bb.citations.length === 0 ? (
        <Empty>No citations yet. Run Research and Draft to ground the argument.</Empty>
      ) : (
        <div className="space-y-6">
          {bb.novelty && (
            <div className="rounded-lg border border-violet-200 bg-violet-50 p-3">
              <div className="mb-1 flex items-center gap-2 text-sm font-semibold text-violet-800">
                Novelty
                <Badge value={bb.novelty.grounded ? "verified" : "needs_review"} />
                <span className="text-xs font-normal text-violet-600">
                  score {bb.novelty.score.toFixed(2)}
                </span>
              </div>
              <p className="text-sm text-slate-700">{bb.novelty.contribution}</p>
            </div>
          )}

          <div>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Citation ledger
            </h3>
            <ul className="space-y-2">
              {bb.citations.map((c) => (
                <li key={c.id} className="rounded-lg border border-slate-200 p-2.5 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs text-slate-400">{c.id} · {c.record_id}</span>
                    <Badge value={c.status} />
                  </div>
                  <p className="mt-1 text-slate-700">{c.proposition}</p>
                  {c.status === "removed" && c.note && (
                    <p className="mt-1 text-xs text-rose-600">✗ {c.note}</p>
                  )}
                  {c.status === "verified" && c.supporting_passage && (
                    <p className="mt-1 border-l-2 border-emerald-200 pl-2 text-xs italic text-slate-500">
                      “{c.supporting_passage}”
                    </p>
                  )}
                </li>
              ))}
            </ul>
          </div>

          {Object.keys(bb.toa).length > 0 && (
            <div>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Table of authorities
              </h3>
              {Object.entries(bb.toa).map(([group, entries]) => (
                <div key={group} className="mb-2">
                  <div className="text-sm font-semibold text-slate-600">{group}</div>
                  <ul className="ml-4 list-disc text-sm text-slate-600">
                    {entries.map((e, i) => (
                      <li key={i}>{e}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          )}

          {report && (
            <details className="rounded-lg border border-slate-200 p-3">
              <summary className="cursor-pointer text-sm font-semibold text-slate-600">
                Verification report
              </summary>
              <pre className="mt-2 whitespace-pre-wrap text-xs text-slate-600">{report}</pre>
            </details>
          )}
        </div>
      )}
    </Card>
  );
}
