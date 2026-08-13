"""Corpus rebuild from opinion text. Offline: no network."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from evals.fetch_opinion_text import (
    MIN_WORDS,
    build_parser,
    cluster_id,
    fetch,
    main,
    opinion_text,
    select,
)

CLUSTER_URL = "https://www.courtlistener.com/opinion/764117/bernstein-v-united-states/"

#: Captured before any test patches `httpx.Client`; the helper below would
#: otherwise call its own replacement and recurse.
_REAL_CLIENT = httpx.Client


def _client(handler) -> httpx.Client:  # type: ignore[no-untyped-def]
    return _REAL_CLIENT(transport=httpx.MockTransport(handler))


def _para(word: str, n: int = 40) -> str:
    return " ".join([word] * n) + "."


# --------------------------------------------------------------------------- #
# Cluster ids come from the record, not from a lookup
# --------------------------------------------------------------------------- #
def test_the_cluster_id_is_read_from_the_records_own_url() -> None:
    """One request per case instead of two, so the daily budget is never the
    reason a rebuild fails.
    """
    assert cluster_id({"url": CLUSTER_URL}) == "764117"


def test_a_record_with_no_usable_url_is_reported_not_guessed() -> None:
    assert cluster_id({"url": "https://example.com/x"}) == ""
    outcome = fetch({"id": "r", "url": ""}, _client(lambda r: httpx.Response(200)), "t")
    assert not outcome.ok
    assert "no cluster id" in outcome.error


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def test_plain_text_is_preferred() -> None:
    assert opinion_text({"plain_text": "The text.", "html": "<p>Other</p>"}) == "The text."


def test_html_is_stripped_when_plain_text_is_absent() -> None:
    text = opinion_text({"html_with_citations": "<p>First para.</p><p>Second para.</p>"})
    assert "<p>" not in text
    assert "First para." in text and "Second para." in text


def test_scripts_and_styles_do_not_become_passages() -> None:
    text = opinion_text({"html": "<style>p{color:red}</style><p>Real text here.</p>"})
    assert "color" not in text
    assert "Real text here." in text


def test_an_opinion_with_no_text_in_any_field_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"plain_text": "", "html": ""}]})

    outcome = fetch({"id": "r", "url": CLUSTER_URL}, _client(handler), "t")
    assert not outcome.ok
    assert "no text" in outcome.error


def test_a_non_200_is_recorded_rather_than_raised() -> None:
    outcome = fetch(
        {"id": "r", "url": CLUSTER_URL},
        _client(lambda r: httpx.Response(429)),
        "t",
    )
    assert not outcome.ok
    assert "429" in outcome.error


def test_the_token_is_sent_as_an_authorization_header() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200, json={"results": [{"plain_text": _para("word", 60)}]})

    fetch({"id": "r", "url": CLUSTER_URL}, _client(handler), "secret")
    assert seen["auth"] == "Token secret"


# --------------------------------------------------------------------------- #
# Passage selection: the whole point of the exercise
# --------------------------------------------------------------------------- #
def test_short_fragments_never_become_passages() -> None:
    """The old corpus's median passage was 12 words. That is what this stops."""
    text = "Too short.\n\n" + _para("substantive", 40)
    passages = select(text, [])
    assert passages
    assert all(len(p.split()) >= MIN_WORDS for p in passages)


def test_passages_are_ranked_against_the_headnotes() -> None:
    """The headnotes record what the case was admitted to the corpus for, so
    ranking against them keeps the curation instead of taking the procedural
    history that opens most opinions.
    """
    procedural = "Appellant filed notice of appeal following entry of judgment below " * 3
    onpoint = "Encryption source code is expression protected by the First Amendment " * 3
    text = f"{procedural}\n\n{onpoint}"
    passages = select(text, ["Encryption source code as protected expression."])
    assert "Encryption" in passages[0]


def test_a_case_sharing_nothing_with_its_headnotes_still_yields_passages() -> None:
    """Terse headnotes should not empty a record."""
    text = _para("unrelated", 40)
    assert select(text, ["Completely different vocabulary."])


def test_selection_is_capped() -> None:
    text = "\n\n".join(_para(f"para{i}", 40) for i in range(40))
    assert len(select(text, [], limit=5)) == 5


# --------------------------------------------------------------------------- #
# The rebuild never destroys the curated corpus
# --------------------------------------------------------------------------- #
def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "in.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "id": "case-1",
                    "type": "case",
                    "title": "A Case",
                    "url": CLUSTER_URL,
                    "passages": ["Encryption source code as protected expression."],
                },
                {
                    "id": "reg-1",
                    "type": "regulation",
                    "title": "A Reg",
                    "passages": ["A regulation headnote."],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_the_input_corpus_is_never_modified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    source = _corpus(tmp_path)
    before = source.read_text()

    import evals.fetch_opinion_text as mod

    monkeypatch.setattr(
        mod.httpx,
        "Client",
        lambda **k: _client(
            lambda r: httpx.Response(
                200,
                json={"results": [{"plain_text": _para("encryption", 40)}]},
            )
        ),
    )
    assert main(["--corpus", str(source), "--out", str(tmp_path / "out.jsonl"), "--delay", "0"]) == 0
    capsys.readouterr()
    assert source.read_text() == before


def test_headnotes_are_preserved_not_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    """Losing them would lose the record of why each case is in the corpus."""
    source = _corpus(tmp_path)
    out = tmp_path / "out.jsonl"

    import evals.fetch_opinion_text as mod

    monkeypatch.setattr(
        mod.httpx,
        "Client",
        lambda **k: _client(
            lambda r: httpx.Response(
                200, json={"results": [{"plain_text": _para("encryption", 40)}]}
            )
        ),
    )
    main(["--corpus", str(source), "--out", str(out), "--delay", "0"])
    capsys.readouterr()

    rebuilt = {json.loads(line)["id"]: json.loads(line) for line in out.read_text().splitlines()}
    case = rebuilt["case-1"]
    assert case["headnotes"] == ["Encryption source code as protected expression."]
    assert len(case["passages"][0].split()) >= MIN_WORDS


def test_non_case_records_pass_through_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    """CourtListener indexes case law; a regulation is not its to answer for."""
    source = _corpus(tmp_path)
    out = tmp_path / "out.jsonl"

    import evals.fetch_opinion_text as mod

    monkeypatch.setattr(
        mod.httpx, "Client", lambda **k: _client(lambda r: httpx.Response(404))
    )
    main(["--corpus", str(source), "--out", str(out), "--delay", "0"])
    capsys.readouterr()

    rebuilt = {json.loads(line)["id"]: json.loads(line) for line in out.read_text().splitlines()}
    assert rebuilt["reg-1"]["passages"] == ["A regulation headnote."]
    assert "headnotes" not in rebuilt["reg-1"]


def test_a_record_whose_fetch_failed_keeps_its_headnotes_as_passages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    """Degrading to a label beats degrading to nothing, and the run says which
    records did not fill.
    """
    source = _corpus(tmp_path)
    out = tmp_path / "out.jsonl"

    import evals.fetch_opinion_text as mod

    monkeypatch.setattr(
        mod.httpx, "Client", lambda **k: _client(lambda r: httpx.Response(500))
    )
    main(["--corpus", str(source), "--out", str(out), "--delay", "0"])
    assert "unfilled: case-1" in capsys.readouterr().out

    rebuilt = {json.loads(line)["id"]: json.loads(line) for line in out.read_text().splitlines()}
    assert rebuilt["case-1"]["passages"] == ["Encryption source code as protected expression."]


def test_a_missing_token_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    import evals.fetch_opinion_text as mod

    monkeypatch.setattr(mod, "get_settings", lambda: type("S", (), {"courtlistener_token": ""})())
    assert main(["--corpus", str(_corpus(tmp_path))]) == 1
    assert "no CourtListener token" in capsys.readouterr().err


def test_the_parser_defaults_to_a_separate_output_file() -> None:
    args = build_parser().parse_args([])
    assert args.out != args.corpus
    assert "fulltext" in str(args.out)


def test_resume_skips_records_already_filled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    """The daily budget is 125 requests, so re-fetching what is already there
    can be the reason a rebuild never finishes.
    """
    source = _corpus(tmp_path)
    out = tmp_path / "out.jsonl"
    calls: list[str] = []

    import evals.fetch_opinion_text as mod

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"results": [{"plain_text": _para("encryption", 40)}]})

    monkeypatch.setattr(mod.httpx, "Client", lambda **k: _client(handler))
    main(["--corpus", str(source), "--out", str(out), "--delay", "0"])
    first = len(calls)
    assert first == 1

    main(["--corpus", str(source), "--out", str(out), "--delay", "0", "--resume"])
    assert len(calls) == first, "a resumed run must not re-spend on a filled record"
    assert "resuming: 1 record(s) already filled" in capsys.readouterr().out


def test_a_429_is_retried_rather_than_recorded_as_a_dead_record() -> None:
    """Pacing is not the same as the record being unavailable."""
    codes = iter([429, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        code = next(codes)
        if code == 429:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"results": [{"plain_text": _para("word", 40)}]})

    outcome = fetch({"id": "r", "url": CLUSTER_URL}, _client(handler), "t")
    assert outcome.ok


def test_the_default_pacing_respects_the_documented_rate_limit() -> None:
    """5 requests/minute is what CourtListener allows an authenticated user, and
    is the same budget RateBudget enforces for verification.
    """
    assert build_parser().parse_args([]).delay >= 12.0
