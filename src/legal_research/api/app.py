"""FastAPI application: REST endpoints + a WebSocket for streaming the brainstorm.

Holds one :class:`Blackboard` per session in memory and drives the pipeline. All
processing is local; the frontend talks only to this server.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .. import DISCLAIMER
from ..blackboard import Blackboard
from ..config import get_settings
from ..pipeline import LegalResearchPipeline
from .schemas import (
    BrainstormRequest,
    ConfigResponse,
    CreateSessionRequest,
    IdeaStatusUpdate,
    IdeateRequest,
    RenderPdfRequest,
    ReviseRequest,
    RunAllRequest,
    RunAllResponse,
    SelectIdeasRequest,
    StepInfo,
)

app = FastAPI(title="legal research generator", version="0.1.0")

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    # Allow localhost, IDE browser-preview proxies (random 127.0.0.1 port), and any
    # private-LAN host so devices on the same network can reach the API.
    allow_origin_regex=(
        r"https?://(localhost|127\.0\.0\.1|"
        r"10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|"
        r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = LegalResearchPipeline(_settings)
_sessions: dict[str, Blackboard] = {}


def _get(session_id: str) -> Blackboard:
    bb = _sessions.get(session_id)
    if bb is None:
        raise HTTPException(status_code=404, detail=f"unknown session {session_id}")
    return bb


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    s = get_settings()
    return ConfigResponse(
        llm_mode=s.llm_mode,
        retriever_mode=s.retriever_mode,
        pdf_renderer=s.pdf_renderer,
        citation_style=s.citation_style,
        disclaimer=DISCLAIMER,
    )


@app.post("/api/sessions", response_model=Blackboard)
def create_session(req: CreateSessionRequest) -> Blackboard:
    bb = pipeline.new_session(req.title)
    _sessions[bb.session_id] = bb
    return bb


@app.get("/api/sessions/{session_id}", response_model=Blackboard)
def get_session(session_id: str) -> Blackboard:
    return _get(session_id)


@app.post("/api/sessions/{session_id}/brainstorm", response_model=Blackboard)
def brainstorm(session_id: str, req: BrainstormRequest) -> Blackboard:
    bb = _get(session_id)
    pipeline.brainstorm(bb, scholar_input=req.message)
    return bb


@app.post("/api/sessions/{session_id}/ideate", response_model=Blackboard)
def ideate(session_id: str, req: IdeateRequest) -> Blackboard:
    bb = _get(session_id)
    pipeline.ideate(bb, seed=req.seed)
    return bb


@app.post("/api/sessions/{session_id}/ideas/select", response_model=Blackboard)
def select_ideas(session_id: str, req: SelectIdeasRequest) -> Blackboard:
    bb = _get(session_id)
    pipeline.select_ideas(bb, [(s.idea_id, s.priority) for s in req.selections])
    return bb


@app.patch("/api/sessions/{session_id}/ideas/{idea_id}", response_model=Blackboard)
def update_idea(session_id: str, idea_id: str, req: IdeaStatusUpdate) -> Blackboard:
    bb = _get(session_id)
    try:
        bb.set_idea_status(idea_id, req.status, priority=req.priority)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return bb


@app.post("/api/sessions/{session_id}/outline", response_model=Blackboard)
def outline(session_id: str) -> Blackboard:
    bb = _get(session_id)
    pipeline.build_outline(bb)
    return bb


@app.post("/api/sessions/{session_id}/research", response_model=Blackboard)
def research(session_id: str) -> Blackboard:
    bb = _get(session_id)
    pipeline.research(bb)
    return bb


@app.post("/api/sessions/{session_id}/draft", response_model=Blackboard)
def draft(session_id: str) -> Blackboard:
    bb = _get(session_id)
    pipeline.draft(bb)
    return bb


@app.post("/api/sessions/{session_id}/verify", response_model=Blackboard)
def verify(session_id: str) -> Blackboard:
    bb = _get(session_id)
    pipeline.verify(bb)
    return bb


@app.post("/api/sessions/{session_id}/format", response_model=Blackboard)
def format_citations(session_id: str) -> Blackboard:
    bb = _get(session_id)
    pipeline.format_citations(bb)
    return bb


@app.post("/api/sessions/{session_id}/voice", response_model=Blackboard)
def voice(session_id: str) -> Blackboard:
    bb = _get(session_id)
    pipeline.edit_voice(bb)
    return bb


@app.post("/api/sessions/{session_id}/novelty", response_model=Blackboard)
def novelty(session_id: str) -> Blackboard:
    bb = _get(session_id)
    pipeline.assess_novelty(bb)
    return bb


@app.post("/api/sessions/{session_id}/revise", response_model=Blackboard)
def revise(session_id: str, req: ReviseRequest) -> Blackboard:
    bb = _get(session_id)
    try:
        pipeline.revise(bb, req.section_id, req.instruction)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return bb


@app.get("/api/sessions/{session_id}/report")
def report(session_id: str) -> dict[str, str]:
    bb = _get(session_id)
    return {"markdown": pipeline.verification_report(bb)}


@app.get("/api/sessions/{session_id}/document")
def document(session_id: str) -> dict[str, object]:
    bb = _get(session_id)
    return pipeline.build_document(bb).model_dump()


@app.get("/api/sessions/{session_id}/preview")
def preview(session_id: str) -> dict[str, str]:
    from ..pdf.html_renderer import HtmlPdfRenderer

    bb = _get(session_id)
    html = HtmlPdfRenderer().render_source(pipeline.build_document(bb))
    return {"html": html}


@app.post("/api/sessions/{session_id}/pdf")
def render_pdf(session_id: str, req: RenderPdfRequest) -> FileResponse:
    bb = _get(session_id)
    out_dir = Path(tempfile.gettempdir()) / "lrg" / bb.session_id
    path = pipeline.render_pdf(bb, out_dir / "paper", author=req.author, date=req.date)
    media = "application/pdf" if path.suffix == ".pdf" else "text/html"
    return FileResponse(str(path), media_type=media, filename=path.name)


@app.post("/api/sessions/{session_id}/run-all", response_model=RunAllResponse)
def run_all(session_id: str, req: RunAllRequest) -> RunAllResponse:
    bb = _get(session_id)
    bb.title = req.title
    result = pipeline.run_all(req.idea, title=req.title, max_ideas=req.max_ideas)
    # Replace the stored blackboard with the fully-run one, preserving the id.
    result.blackboard.session_id = session_id
    _sessions[session_id] = result.blackboard
    return RunAllResponse(
        session_id=session_id,
        steps=[StepInfo(agent=s.agent, runtime=s.runtime, summary=s.summary) for s in result.steps],
        shippable=result.blackboard.is_shippable(),
    )


@app.websocket("/api/sessions/{session_id}/brainstorm/stream")
async def brainstorm_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    bb = _sessions.get(session_id)
    if bb is None:
        await websocket.send_json({"type": "error", "detail": "unknown session"})
        await websocket.close()
        return
    try:
        while True:
            payload = await websocket.receive_json()
            message = payload.get("message")
            result = pipeline.brainstorm(bb, scholar_input=message)
            question = result.payload.get("question", result.summary)
            for token in str(question).split(" "):
                await websocket.send_json({"type": "token", "data": token + " "})
                await asyncio.sleep(0.02)
            await websocket.send_json(
                {"type": "done", "question": question, "turns": result.payload.get("turns")}
            )
    except WebSocketDisconnect:
        return


def main() -> None:
    import uvicorn

    s = get_settings()
    uvicorn.run("legal_research.api.app:app", host=s.host, port=s.port, reload=False)


if __name__ == "__main__":
    main()
