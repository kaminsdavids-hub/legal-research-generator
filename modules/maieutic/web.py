"""HTTP transport for the loop.

The package is transport-agnostic by design, so this is the only module that
knows about HTTP, exactly as `service.py` is the only one that knows about
settings. Everything that decides anything still lives in `loop.py`; these
handlers translate.

**Session ids are validated, not trusted.** Each session is a file named from an
id the client supplies, which is a path traversal waiting to happen. Ids are
generated server-side and matched against a strict pattern on the way back in;
anything else is a 400 before it reaches the filesystem. A local-first tool is
still a tool with an HTTP server in it.

**The author's channel is the only way in.** `POST /answer` writes a HUMAN node
because the request *is* the author speaking. There is deliberately no endpoint
that accepts a node, a patch, or a provenance — a client that could post
machine-drafted text as human-authored would defeat the record the whole system
keeps, and no amount of care at the UI layer would restore it.

**Refusals are part of the response, not an error.** A patch the gates decline
returns 200 with the reasons: the author asked a question and got an answer about
their work, which is a successful interaction whatever the merge decided. Sending
a 4xx would tell the client something went wrong, and nothing did.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .learn import Journal, LearnedPolicy
from .loop import Gates, Session, StepResult, TurnProvider
from .render import Audience

#: Server-generated ids only. Matched on the way back in so a crafted id cannot
#: escape the session directory.
SESSION_ID = re.compile(r"^[0-9a-f]{12}$")


class BeginRequest(BaseModel):
    thesis: str = Field(min_length=1)
    section: str = ""


class AnswerRequest(BaseModel):
    text: str = Field(min_length=1)


class QuestionOut(BaseModel):
    text: str
    gap_kind: str
    section: str = ""


class StepOut(BaseModel):
    merged: bool
    added: int = 0
    refusals: list[str] = Field(default_factory=list)
    advisories: list[str] = Field(default_factory=list)
    dropped: int = 0
    #: The exchange failed but the answer was kept. The author must be told which.
    turn_error: str = ""
    #: The gap needs an edit rather than an addition, so the answer did not close it.
    unresolved: bool = False
    next_question: QuestionOut | None = None


class StatusOut(BaseModel):
    session_id: str
    nodes: int
    gaps: int
    outstanding: int
    open_problems: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pending: QuestionOut | None = None


def _question_out(session: Session) -> QuestionOut | None:
    question = session.pending
    if question is None:
        return None
    return QuestionOut(
        text=question.text,
        gap_kind=question.gap.kind.value,
        section=question.gap.section,
    )


def _step_out(session: Session, result: StepResult) -> StepOut:
    return StepOut(
        merged=result.merged,
        added=len(result.added),
        refusals=result.refusals,
        advisories=result.report.advisories if result.report else [],
        dropped=len(result.dropped),
        turn_error=result.turn_error,
        unresolved=result.unresolved,
        next_question=_question_out(session),
    )


def create_app(
    root: Path,
    gates: Gates | None = None,
    turns: TurnProvider | None = None,
) -> FastAPI:
    """Build the loop's HTTP surface over a directory of session files."""
    app = FastAPI(title="maieutic loop", version="0.1.0")
    root.mkdir(parents=True, exist_ok=True)
    journal_path = root / "journal.json"

    def path_for(session_id: str) -> Path:
        if not SESSION_ID.match(session_id):
            raise HTTPException(status_code=400, detail="malformed session id")
        return root / f"{session_id}.json"

    def load(session_id: str) -> Session:
        path = path_for(session_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="no such session")
        session = Session.load(path)
        # The journal is shared across manuscripts: what it records is a fact
        # about the author, not about this session (REMEDIATION §20).
        session.journal = Journal.load(journal_path)
        return session

    def persist(session_id: str, session: Session) -> None:
        session.save(path_for(session_id))
        session.journal.save(journal_path)

    @app.post("/api/sessions", response_model=StatusOut)
    def begin(request: BeginRequest) -> StatusOut:
        session_id = uuid.uuid4().hex[:12]
        session = Session()
        session.journal = Journal.load(journal_path)
        try:
            session.begin(request.thesis, request.section)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session.ask()
        persist(session_id, session)
        return _status(session_id, session)

    @app.get("/api/sessions/{session_id}", response_model=StatusOut)
    def status(session_id: str) -> StatusOut:
        return _status(session_id, load(session_id))

    @app.post("/api/sessions/{session_id}/answer", response_model=StepOut)
    def answer(session_id: str, request: AnswerRequest) -> StepOut:
        session = load(session_id)
        if session.pending is None and session.ask() is None:
            raise HTTPException(status_code=409, detail="no question is pending")
        result = session.answer(request.text, gates or Gates.offline(), turns)
        if result.merged:
            session.ask()
        persist(session_id, session)
        return _step_out(session, result)

    @app.post("/api/sessions/{session_id}/skip", response_model=StatusOut)
    def skip(session_id: str) -> StatusOut:
        session = load(session_id)
        if session.decline() is None:
            raise HTTPException(status_code=409, detail="no question is pending")
        session.ask()
        persist(session_id, session)
        return _status(session_id, session)

    @app.get("/api/sessions/{session_id}/manuscript")
    def manuscript(session_id: str, review: bool = False) -> dict[str, Any]:
        session = load(session_id)
        rendered = session.manuscript(
            Audience.REVIEW if review else Audience.MANUSCRIPT
        )
        return {
            "text": rendered.text,
            "open_problems": rendered.open_problems,
            "warnings": rendered.warnings,
        }

    @app.get("/api/policy")
    def policy() -> dict[str, Any]:
        journal = Journal.load(journal_path)
        return {
            "summary": journal.summary(),
            "explain": LearnedPolicy(journal).explain(),
            "signal": (
                "engagement is measured from whether the author answered, never "
                "from whether the gates accepted it"
            ),
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    def _status(session_id: str, session: Session) -> StatusOut:
        rendered = session.manuscript()
        return StatusOut(
            session_id=session_id,
            nodes=len(session.graph.nodes),
            gaps=len(session.gaps()),
            outstanding=len(session.outstanding()),
            open_problems=rendered.open_problems,
            warnings=rendered.warnings,
            pending=_question_out(session),
        )

    return app


#: A single self-contained page. Deliberately not a build step: the loop's value
#: is the questions, and a client that needs a toolchain to try is a client
#: nobody tries. Integrating with the Next.js frontend is separate work.
PAGE = """<!doctype html>
<meta charset="utf-8"><title>maieutic</title>
<style>
 body{font:16px/1.6 Georgia,serif;max-width:44rem;margin:3rem auto;padding:0 1rem}
 textarea{width:100%;font:inherit;padding:.6rem;min-height:7rem}
 button{font:inherit;padding:.4rem 1rem;margin-right:.5rem}
 .q{background:#f6f4ef;border-left:3px solid #999;padding:1rem;margin:1.5rem 0}
 .refused{color:#8a1f11}.advisory{color:#7a6a1f}.note{color:#555;font-size:.9em}
 pre{white-space:pre-wrap;background:#fafafa;padding:1rem;font:14px/1.5 ui-monospace,monospace}
</style>
<h1>maieutic</h1>
<div id="start">
  <p>Open a manuscript with your own thesis.</p>
  <textarea id="thesis" placeholder="Your thesis..."></textarea>
  <button onclick="begin()">Begin</button>
</div>
<div id="loop" hidden>
  <div class="q" id="question"></div>
  <textarea id="answer" placeholder="Your answer..."></textarea>
  <button onclick="answer()">Answer</button>
  <button onclick="skip()">Skip this question</button>
  <div id="out"></div>
  <h2>Manuscript</h2><pre id="ms"></pre>
</div>
<script>
let sid=null;
const $=id=>document.getElementById(id);
async function api(path,opts){const r=await fetch(path,opts);if(!r.ok){const e=await r.json().catch(()=>({detail:r.statusText}));throw new Error(e.detail)}return r.json()}
function showQuestion(q){
  if(!q){$('question').textContent='No unasked gaps remain.';$('answer').hidden=true;return}
  $('answer').hidden=false;
  $('question').innerHTML=`<strong>${q.gap_kind.replace(/_/g,' ')}</strong><br>${q.text}`;
}
async function begin(){
  const thesis=$('thesis').value.trim(); if(!thesis)return;
  const s=await api('/api/sessions',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({thesis})});
  sid=s.session_id;$('start').hidden=true;$('loop').hidden=false;showQuestion(s.pending);refresh();
}
async function answer(){
  const text=$('answer').value.trim(); if(!text)return;
  let r;
  try{ r=await api(`/api/sessions/${sid}/answer`,{method:'POST',
    headers:{'content-type':'application/json'},body:JSON.stringify({text})}) }
  catch(e){ $('out').innerHTML=`<p class="refused">${e.message}</p>`; return }
  let html = r.merged ? `<p>Merged: ${r.added} node(s).</p>`
                      : `<p class="refused">Not merged. The question stays open.</p>`;
  r.refusals.forEach(x=>html+=`<p class="refused">refused — ${x}</p>`);
  r.advisories.forEach(x=>html+=`<p class="advisory">advisory — ${x}</p>`);
  if(r.dropped)html+=`<p class="note">${r.dropped} machine node(s) could not be grounded; your answer merged without them.</p>`;
  if(r.turn_error)html+=`<p class="note">The exchange failed (${r.turn_error}); your answer was kept.</p>`;
  if(r.unresolved)html+=`<p class="note">This gap needs an edit rather than an addition, so your answer did not close it.</p>`;
  $('out').innerHTML=html;
  if(r.merged)$('answer').value='';
  showQuestion(r.next_question);refresh();
}
async function skip(){const s=await api(`/api/sessions/${sid}/skip`,{method:'POST'});showQuestion(s.pending);refresh()}
async function refresh(){const m=await api(`/api/sessions/${sid}/manuscript`);$('ms').textContent=m.text||'(empty)'}
</script>
"""


def _app() -> FastAPI:  # pragma: no cover - the uvicorn entry point
    """Factory for `make maieutic-web`. Sessions live under `.maieutic/`."""
    return create_app(Path(".maieutic"))
