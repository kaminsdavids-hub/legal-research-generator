"use client";

import { MessageCircle, Send, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { API_BASE, type Blackboard } from "@/lib/api";
import { Button, Card, Empty } from "./ui";

export function BrainstormPanel({
  bb,
  onIdeate,
  onRefresh,
}: {
  bb: Blackboard;
  onIdeate: (seed: string) => void;
  onRefresh: (bb: Blackboard) => void;
}) {
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState("");
  const wsRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const url = `${API_BASE.replace("http", "ws")}/api/sessions/${bb.session_id}/brainstorm/stream`;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    let buffer = "";
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "token") {
        buffer += msg.data;
        setStreaming(buffer);
      } else if (msg.type === "done") {
        buffer = "";
        setStreaming("");
        void refresh();
      }
    };
    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bb.session_id]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [bb.brainstorm.length, streaming]);

  async function refresh() {
    const res = await fetch(`${API_BASE}/api/sessions/${bb.session_id}`);
    onRefresh(await res.json());
  }

  function send() {
    const text = draft.trim();
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ message: text || null }));
    setDraft("");
    setStreaming("...");
  }

  return (
    <Card
      title="Socratic Brainstorm"
      icon={<MessageCircle size={16} className="text-accent" />}
      actions={
        <Button variant="subtle" onClick={() => onIdeate(bb.thesis || draft)}>
          <Sparkles size={14} /> Generate ideas
        </Button>
      }
      className="h-full"
    >
      <div className="flex h-full flex-col">
        <div ref={scrollRef} className="flex-1 space-y-3 overflow-auto scroll-thin pr-1">
          {bb.brainstorm.length === 0 && !streaming && (
            <Empty>Share a raw idea. The interviewer will sharpen it with questions.</Empty>
          )}
          {bb.brainstorm.map((turn, i) => (
            <div
              key={i}
              className={
                turn.role === "scholar"
                  ? "ml-8 rounded-lg bg-accent/10 px-3 py-2 text-sm text-slate-800"
                  : "mr-8 rounded-lg bg-slate-100 px-3 py-2 text-sm text-slate-700"
              }
            >
              <div className="mb-0.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                {turn.role}
              </div>
              {turn.content}
            </div>
          ))}
          {streaming && streaming !== "..." && (
            <div className="mr-8 rounded-lg bg-slate-100 px-3 py-2 text-sm text-slate-700">
              <div className="mb-0.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                interviewer
              </div>
              {streaming}
              <span className="animate-pulse">▍</span>
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
                send();
              }
            }}
            rows={2}
            placeholder="Type your idea or answer... (Enter to send)"
            className="flex-1 resize-none rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-accent focus:outline-none"
          />
          <Button onClick={send}>
            <Send size={14} /> Send
          </Button>
        </div>
      </div>
    </Card>
  );
}
