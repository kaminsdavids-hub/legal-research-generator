"""FastAPI application: REST endpoints + a WebSocket for streaming the brainstorm.

Holds one :class:`Blackboard` per session in memory and drives the pipeline. All
processing is local; the frontend talks only to this server.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .. import DISCLAIMER
from ..blackboard import Blackboard
from ..config import get_settings
from ..multi_chat import MultiChatTurn as MultiChatEngineTurn
from ..multi_chat import MultiModelChat
from ..pipeline import LegalResearchPipeline
from .auth import ApiKeyMiddleware, warn_if_unprotected
from .jobs import JobStore, SessionBusy
from .schemas import (
    BrainstormRequest,
    ConfigResponse,
    CreateSessionRequest,
    DialecticRequest,
    DialecticResponse,
    DraftEssayRequest,
    IdeaStatusUpdate,
    IdeateRequest,
    JobRequest,
    JobResponse,
    MultiChatRequest,
    MultiChatResponse,
    RenderPdfRequest,
    RetrievalSmokeRequest,
    RetrievalSmokeResponse,
    ReviseRequest,
    RunAllRequest,
    RunAllResponse,
    SelectIdeasRequest,
    SocraticReviseRequest,
    SocraticReviseResponse,
    StepInfo,
)

app = FastAPI(title="legal research generator", version="0.1.0")

_settings = get_settings()
# Added before CORS, so CORS ends up outermost and still answers preflight;
# the gate exempts OPTIONS for the same reason.
app.add_middleware(ApiKeyMiddleware, key=_settings.api_key)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    # Allow localhost, IDE browser-preview proxies (random 127.0.0.1 port), private
    # LAN hosts, and Netlify deploy-preview aliases for this app.
    allow_origin_regex=(
        r"https?://(localhost|127\.0\.0\.1|"
        r"10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|"
        r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|"
        r"([a-z0-9-]+--)?legal-research-generator-app\.netlify\.app)(:\d+)?"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = LegalResearchPipeline(_settings)
multi_chat = MultiModelChat(_settings)
_sessions: dict[str, Blackboard] = {}
_jobs = JobStore()

#: Built lazily: the family guard raises if two debate roles share a base model
#: family, and that must surface as a request error rather than a dead import.
_dialectic_chat: object | None = None
_dialectic_error: str = ""


def _get_dialectic() -> object:
    global _dialectic_chat, _dialectic_error
    if _dialectic_chat is None and not _dialectic_error:
        try:
            from modules.dialectic.service import build_dialectic_chat

            _dialectic_chat = build_dialectic_chat(get_settings())
        except Exception as exc:  # noqa: BLE001 - reported to the caller as 503
            _dialectic_error = f"{type(exc).__name__}: {exc}"
    if _dialectic_chat is None:
        raise HTTPException(
            status_code=503, detail=f"dialectic module unavailable: {_dialectic_error}"
        )
    return _dialectic_chat


def _get(session_id: str) -> Blackboard:
    bb = _sessions.get(session_id)
    if bb is None:
        raise HTTPException(status_code=404, detail=f"unknown session {session_id}")
    return bb


@app.post("/api/retrieval/smoke", response_model=RetrievalSmokeResponse)
def retrieval_smoke(req: RetrievalSmokeRequest) -> RetrievalSmokeResponse:
    s = get_settings()
    if not s.debug_endpoints_enabled:
        raise HTTPException(status_code=404, detail="debug endpoints are disabled")
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    k = max(1, min(int(req.k), 10))
    hits = pipeline.retriever.search(query, k=k)
    return RetrievalSmokeResponse(
        retriever_mode=s.retriever_mode,
        embed_model=s.embed_model,
        retriever_impl=type(pipeline.retriever).__name__,
        hits=[
            {
                "record_id": hit.record_id,
                "score": float(hit.score),
                "locator": hit.locator,
                "text": hit.text,
            }
            for hit in hits
        ],
    )


@app.post("/api/sessions/{session_id}/multi-chat", response_model=MultiChatResponse)
def session_multi_chat(session_id: str, req: MultiChatRequest) -> MultiChatResponse:
    _get(session_id)
    try:
        result = multi_chat.chat(
            req.message,
            history=[
                MultiChatEngineTurn(
                    role="assistant" if turn.role.lower() == "assistant" else "user",
                    content=turn.content,
                )
                for turn in req.history
            ],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"multi-chat unavailable: {exc}") from exc

    return _multi_chat_response(result)


def _multi_chat_response(result: object) -> MultiChatResponse:
    """Build the wire response from an engine result.

    Extracted so the synchronous route and the job produce the same object. Two
    hand-built copies of a twenty-field response would drift, and the drift
    would show up as a client that works on one path and not the other.
    """

    grounding = result.grounding  # type: ignore[attr-defined]
    return MultiChatResponse(
        final_answer=result.final_answer,  # type: ignore[attr-defined]
        model_answers=[
            {"model": answer.model, "content": answer.content}
            for answer in result.model_answers  # type: ignore[attr-defined]
        ],
        verifiers=[
            {"model": verdict.model, "verdict": verdict.verdict}
            for verdict in result.verifiers  # type: ignore[attr-defined]
        ],
        grounding={
            "available": grounding.available,
            "summary": grounding.summary(),
            "note": grounding.note,
            "authorities": list(grounding.authorities),
            "findings": [
                {
                    "citation": finding.citation,
                    "kind": finding.kind.value,
                    "status": finding.status.value,
                    "detail": finding.detail,
                    "record_id": finding.record_id,
                    "support": finding.support,
                }
                for finding in grounding.findings
            ],
            "verified_count": len(grounding.supported),
            "unverified_count": len(grounding.problems),
            "unconfirmed_count": len(grounding.unconfirmed),
        },
    )


@app.post("/api/sessions/{session_id}/dialectic", response_model=DialecticResponse)
def session_dialectic(session_id: str, req: DialecticRequest) -> DialecticResponse:
    """Run one dialectic exchange: thesis, antithesis, synthesis, and cruxes."""

    _get(session_id)
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="message must not be empty")

    chat = _get_dialectic()
    try:
        turn = chat.chat(question)  # type: ignore[attr-defined]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"dialectic unavailable: {exc}") from exc

    return _dialectic_response(turn)


def _dialectic_response(turn: object) -> DialecticResponse:
    """Build the wire response from a dialectic turn.

    Extracted for the same reason as the multi-chat builder: one construction,
    so the synchronous route and the job cannot drift apart.
    """

    from modules.dialectic.copy import copy_crux_table, copy_exchange, copy_position

    def slot(s: object) -> dict[str, object]:
        return {
            "proposition": s.proposition,  # type: ignore[attr-defined]
            "court_hint": s.court_hint,  # type: ignore[attr-defined]
            "weight": s.weight,  # type: ignore[attr-defined]
            "status": s.status,  # type: ignore[attr-defined]
            "cluster_id": s.cluster_id,  # type: ignore[attr-defined]
            "normalized_cite": s.normalized_cite,  # type: ignore[attr-defined]
            "note": s.note,  # type: ignore[attr-defined]
        }

    def position(p: object) -> dict[str, object]:
        return {
            "side": p.side,  # type: ignore[attr-defined]
            "model": p.model,  # type: ignore[attr-defined]
            "family": p.family,  # type: ignore[attr-defined]
            "propositions": [slot(s) for s in p.propositions],  # type: ignore[attr-defined]
        }

    return DialecticResponse(
        question=turn.question,  # type: ignore[attr-defined]
        thesis=position(turn.thesis),  # type: ignore[arg-type, attr-defined]
        antithesis=position(turn.antithesis),  # type: ignore[arg-type, attr-defined]
        synthesis=turn.synthesis,  # type: ignore[attr-defined]
        cruxes=[
            {
                "thesis_prop": slot(c.thesis_prop),
                "antithesis_prop": slot(c.antithesis_prop),
                "negates": c.negates,
                "partition": c.partition,
                "winner": c.winner,
                "outcome_bearing": c.outcome_bearing,
                "nli_source": c.nli_source,
            }
            for c in turn.cruxes  # type: ignore[attr-defined]
        ],  # type: ignore[arg-type]
        calls_spent=turn.calls_spent,
        regenerated=turn.regenerated,
        crux_note=turn.crux_note,
        copy_exchange=copy_exchange(turn),
        copy_thesis=copy_position(turn, "thesis"),
        copy_antithesis=copy_position(turn, "antithesis"),
        copy_crux_table=copy_crux_table(turn),
    )


@app.post("/api/sessions/{session_id}/revise/socratic", response_model=SocraticReviseResponse)
def socratic_revise(session_id: str, req: SocraticReviseRequest) -> SocraticReviseResponse:
    bb = _get(session_id)
    try:
        result = pipeline.socratic_revise_paragraph(
            bb,
            section_id=req.section_id,
            paragraph_index=req.paragraph_index,
            message=req.message,
            history=[t.model_dump() for t in req.history],
            apply_revision=req.apply_revision,
            mode=req.mode,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SocraticReviseResponse(**result.payload)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    s = get_settings()
    return ConfigResponse(
        debug_endpoints_enabled=s.debug_endpoints_enabled,
        llm_mode=s.llm_mode,
        retriever_mode=s.retriever_mode,
        embed_model=s.embed_model,
        support_scorer=s.support_scorer,
        saul_model=s.saul_model,
        writer_model=s.writer_model,
        gemma_model=s.gemma_model,
        hermes_model=s.hermes_model,
        hermes3_model=s.hermes3_model,
        multi_chat_models={
            "gpt_oss": s.multi_chat_gpt_oss_model,
            "gemma4": s.multi_chat_gemma4_model,
            "apertus": s.multi_chat_apertus_model,
            "nemotron": s.multi_chat_nemotron_model,
            "hermes3": s.multi_chat_hermes3_model,
        },
        multi_chat_verifiers={
            "gemma3": s.multi_chat_verifier_gemma3_model,
            "saul": s.multi_chat_verifier_saul_model,
        },
        dialectic_models={
            "thesis": s.dialectic_thesis_model,
            "antithesis": s.dialectic_antithesis_model,
            "synthesis": s.dialectic_synthesis_model,
            "nli": s.dialectic_nli_model,
        },
        grammar_chain_enabled=s.grammar_chain_enabled,
        grammar_chain_roles=[r.strip() for r in s.grammar_chain_roles.split(",") if r.strip()],
        pdf_renderer=s.pdf_renderer,
        citation_style=s.citation_style,
        manuscript_target_min_words=s.manuscript_target_min_words,
        manuscript_target_max_words=s.manuscript_target_max_words,
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
    try:
        pipeline.brainstorm(bb, scholar_input=req.message)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"brainstorm unavailable: {exc}") from exc
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


@app.post("/api/sessions/{session_id}/draft/essay", response_model=Blackboard)
def draft_essay(session_id: str, req: DraftEssayRequest) -> Blackboard:
    bb = _get(session_id)
    target_words = max(1200, min(12000, int(req.target_words)))
    pipeline.draft(bb, target_words=target_words)
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
    try:
        result = pipeline.run_all(req.idea, title=req.title, max_ideas=req.max_ideas, bb=bb)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "run-all unavailable: manuscript pipeline encountered an internal "
                "failure before fallback recovery"
            ),
        ) from exc
    # Replace the stored blackboard with the fully-run one, preserving the id.
    result.blackboard.session_id = session_id
    _sessions[session_id] = result.blackboard
    return RunAllResponse(
        session_id=session_id,
        steps=[StepInfo(agent=s.agent, runtime=s.runtime, summary=s.summary) for s in result.steps],
        shippable=result.blackboard.is_shippable(
            block_on_mechanism=bool(get_settings().mechanism_gate_blocks_ship)
        ),
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
            try:
                result = pipeline.brainstorm(bb, scholar_input=message)
            except Exception as exc:
                await websocket.send_json({"type": "error", "detail": f"brainstorm unavailable: {exc}"})
                continue
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
    warn_if_unprotected(s.host, s.api_key)
    uvicorn.run("legal_research.api.app:app", host=s.host, port=s.port, reload=False)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# Slow steps as jobs
#
# The synchronous routes above stay exactly as they are. They are correct for a
# loopback client with no proxy in between, they are what the tests exercise,
# and removing them would break every caller to fix a problem only some callers
# have. These routes are the alternative for clients behind something with a
# shorter patience than a five-minute model call.
# --------------------------------------------------------------------------- #

#: Steps that may be run as a job, and how. Every one of these drives a model.
#:
#: A table rather than a generic `getattr(pipeline, step)`: that would turn a
#: path segment into an attribute lookup on a live object, letting a caller
#: reach anything the pipeline exposes. The allowed set is stated here.
_JOB_STEPS: dict[str, Callable[[Blackboard], object]] = {
    "brainstorm": lambda bb: pipeline.brainstorm(bb, None),
    "ideate": lambda bb: pipeline.ideate(bb, None),
    "outline": pipeline.build_outline,
    "research": pipeline.research,
    "draft": pipeline.draft,
    "voice": pipeline.edit_voice,
    "verify": pipeline.verify,
    "format": pipeline.format_citations,
    "novelty": pipeline.assess_novelty,
    "mechanism": pipeline.mechanism_gate,
}

#: Steps that answer a question rather than advancing the manuscript. They are
#: exempt from the one-writer-per-session rule: a user can already hold two
#: conversations at once, and putting them under the writers' lock would remove
#: that to prevent a corruption they cannot cause.
#:
#: `socratic` is not in this set even though it usually belongs there, because
#: whether it writes is decided by the request and not by the step -- see
#: :func:`_mutates`.
QUESTION_STEPS = frozenset({"multi-chat", "dialectic"})

#: Every step that needs a message to act on.
NEEDS_MESSAGE = QUESTION_STEPS | {"socratic"}


def _mutates(req: JobRequest) -> bool:
    """Whether this submission will write to the blackboard.

    A property of the request, not of the step. `socratic` answers a question
    about a paragraph and, if `apply_revision` is set, rewrites it -- so the
    same step name is read-only on one call and exclusive on the next. Deciding
    exclusivity from the name alone would either lock out concurrent questions
    that harm nothing, or let a revision land while a draft is rewriting the
    same section.
    """

    if req.step in QUESTION_STEPS:
        return False
    if req.step == "socratic":
        return req.apply_revision
    return True

#: The whole pipeline in one job. Kept out of `_JOB_STEPS` because it is the one
#: step that takes arguments and the one that produces something the blackboard
#: does not hold -- the per-agent step log and the shippable verdict.
RUN_ALL = "run-all"


def _socratic_work(bb: Blackboard, req: JobRequest) -> dict[str, object]:
    """One Socratic exchange about a paragraph, optionally applying the result."""

    result = pipeline.socratic_revise_paragraph(
        bb,
        section_id=req.section_id,
        paragraph_index=req.paragraph_index,
        message=req.message,
        history=[t.model_dump() for t in req.history],
        apply_revision=req.apply_revision,
        mode=req.mode,
    )
    return dict(SocraticReviseResponse(**result.payload).model_dump())


def _multi_chat_work(req: JobRequest) -> dict[str, object]:
    """One multi-model exchange, in the shape the synchronous route returns.

    `.model_dump()` rather than a hand-built dict: the response model is the
    contract, and re-describing it here would let the two drift apart silently.
    """

    result = multi_chat.chat(
        req.message,
        history=[
            MultiChatEngineTurn(
                role="assistant" if turn.role.lower() == "assistant" else "user",
                content=turn.content,
            )
            for turn in req.history
        ],
    )
    return _multi_chat_response(result).model_dump()


def _dialectic_work(req: JobRequest) -> dict[str, object]:
    """One dialectic exchange, in the shape the synchronous route returns."""

    return _dialectic_response(_get_dialectic().chat(req.message.strip())).model_dump()  # type: ignore[attr-defined]


def _run_all_work(session_id: str, bb: Blackboard, req: JobRequest) -> dict[str, object]:
    """Run the full pipeline and return the summary the blackboard cannot carry.

    `pipeline.run_all` mutates `bb` in place and has its own per-stage fallbacks,
    so a failure part-way leaves the session advanced as far as it got. That is
    the honest outcome and it is observable: the client fetches the session and
    sees exactly which stages completed. Rolling back to a snapshot would throw
    away work the run really did.
    """

    bb.title = req.title
    result = pipeline.run_all(req.idea, title=req.title, max_ideas=req.max_ideas, bb=bb)
    result.blackboard.session_id = session_id
    _sessions[session_id] = result.blackboard
    return {
        "steps": [
            {"agent": s.agent, "runtime": s.runtime, "summary": s.summary} for s in result.steps
        ],
        "shippable": result.blackboard.is_shippable(
            block_on_mechanism=bool(get_settings().mechanism_gate_blocks_ship)
        ),
    }


@app.post("/api/sessions/{session_id}/jobs", response_model=JobResponse, status_code=202)
def submit_job(session_id: str, req: JobRequest) -> JobResponse:
    """Start a slow step and return at once with an id to poll.

    202, not 200: the work has been accepted and has not been done. A 200 here
    would tell every cache and client library that it holds a result.
    """

    bb = _get(session_id)
    if req.step in NEEDS_MESSAGE and not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    if req.step == "socratic" and not any(s.id == req.section_id for s in bb.outline):
        # The synchronous route answers an unknown section with a 404. Accepting
        # a job that is certain to fail would turn that into a poll and a
        # failure state, which is a worse way to learn the same thing.
        raise HTTPException(status_code=404, detail=f"no such section: {req.section_id!r}")

    if req.step == RUN_ALL:
        task: Callable[[], object] = lambda: _run_all_work(session_id, bb, req)  # noqa: E731
    elif req.step == "multi-chat":
        task = lambda: _multi_chat_work(req)  # noqa: E731
    elif req.step == "dialectic":
        task = lambda: _dialectic_work(req)  # noqa: E731
    elif req.step == "socratic":
        task = lambda: _socratic_work(bb, req)  # noqa: E731
    else:
        step = _JOB_STEPS.get(req.step)
        if step is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown step {req.step!r}; expected one of "
                    f"{sorted([*_JOB_STEPS, RUN_ALL, *QUESTION_STEPS, 'socratic'])}"
                ),
            )
        task = lambda: step(bb)  # noqa: E731
    try:
        job = _jobs.submit(session_id, req.step, task, mutates=_mutates(req))
    except SessionBusy as busy:
        # 409, not 429: this is not rate limiting, it is a statement that the
        # session is in a state where a second step cannot start. The caller
        # should poll the job it already has rather than retry this.
        raise HTTPException(status_code=409, detail=str(busy)) from busy
    return JobResponse(job_id=job.id, session_id=session_id, step=job.step, state=job.state)


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    """Poll a job. Answers in microseconds, which is the whole point."""

    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return JobResponse(
        job_id=job.id,
        session_id=job.session_id,
        step=job.step,
        state=job.state,
        error=job.error,
        # `result` is a small summary and only run-all sets one. The blackboard
        # is never inlined: it is large, a poller would fetch it repeatedly for
        # no reason, and it is already at /api/sessions/{id}, which the client
        # should call once, on success.
        result=job.result,
    )
