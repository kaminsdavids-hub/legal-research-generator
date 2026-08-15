"""Corpus ingestion from real legal sources — exercised fully offline.

Network ingestors are driven by a fake :class:`Fetcher` returning fixtures, so no
test touches the network. This is the layer that lets real authority (case law,
statutes, regulations, uploads) enter the corpus the Verifier trusts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from legal_research.citations.corpus import load_corpus
from legal_research.ingest.base import chunk_passages, write_jsonl
from legal_research.ingest.cap import CapIngestor
from legal_research.ingest.courtlistener import CourtListenerIngestor, parse_citation
from legal_research.ingest.statutes import CfrIngestor, UsCodeIngestor
from legal_research.ingest.uploads import UploadIngestor
from legal_research.models import SourceType


class FakeFetcher:
    def __init__(
        self,
        json_by_url: dict[str, Any] | None = None,
        text_by_url: dict[str, str] | None = None,
    ) -> None:
        self._json = json_by_url or {}
        self._text = text_by_url or {}
        self.calls: list[str] = []

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        self.calls.append(url)
        return self._json[url]

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        self.calls.append(url)
        return self._text[url]


def test_chunk_passages_splits_and_normalizes() -> None:
    text = "First sentence is here.   Second sentence follows.  " + ("word " * 400)
    passages = chunk_passages(text, max_chars=200)
    assert len(passages) >= 2
    assert all(len(p) <= 200 for p in passages)
    assert all(p == p.strip() and "  " not in p for p in passages)


def test_parse_citation() -> None:
    assert parse_citation("425 U.S. 185") == (425, "U.S.", 185)
    assert parse_citation("not a citation") is None


def test_uploads_txt(tmp_path: Path) -> None:
    f = tmp_path / "my_brief.txt"
    f.write_text("The doctrine of stare decisis binds lower courts. " * 5, encoding="utf-8")
    record = UploadIngestor().record_from_file(f)
    assert record is not None
    assert record.type is SourceType.SECONDARY
    assert record.title == "My Brief"
    assert record.passages
    assert record.id.startswith("upload-my-brief-")


def test_uploads_id_is_stable(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("identical content for hashing", encoding="utf-8")
    r1 = UploadIngestor().record_from_file(f)
    r2 = UploadIngestor().record_from_file(f)
    assert r1 is not None and r2 is not None
    assert r1.id == r2.id


def test_cap_case_mapping() -> None:
    case = {
        "id": 123,
        "name_abbreviation": "Ernst & Ernst v. Hochfelder",
        "decision_date": "1976-03-30",
        "citations": [{"cite": "425 U.S. 185", "type": "official"}],
        "court": {"name": "Supreme Court of the United States"},
        "casebody": {
            "data": {
                "opinions": [
                    {"text": "Section 10(b) and Rule 10b-5 require a showing of scienter."}
                ]
            }
        },
    }
    record = CapIngestor().record_from_case(case)
    assert record is not None
    assert record.type is SourceType.CASE
    assert record.id == "us-425-185"
    assert (record.volume, record.reporter, record.page) == (425, "U.S.", 185)
    assert record.year == 1976
    assert record.passages


def test_courtlistener_ingest_with_fake_fetcher() -> None:
    search_url = "https://www.courtlistener.com/api/rest/v4/search/"
    op_url = "https://www.courtlistener.com/api/rest/v4/opinions/9001/"
    fetcher = FakeFetcher(
        json_by_url={
            search_url: {
                "results": [
                    {
                        "cluster_id": 1,
                        "caseName": "Ernst & Ernst v. Hochfelder",
                        "citation": ["425 U.S. 185"],
                        "court": "Supreme Court",
                        "dateFiled": "1976-03-30",
                        "absolute_url": "/opinion/109389/ernst/",
                        "opinions": [{"id": 9001, "snippet": "scienter required"}],
                    }
                ],
                "next": None,
            },
            op_url: {
                "plain_text": "Section 10(b) and Rule 10b-5 require a showing of scienter."
            },
        }
    )
    records = CourtListenerIngestor(fetcher).ingest("scienter", count=5)
    assert len(records) == 1
    r = records[0]
    assert r.id == "us-425-185"
    assert r.type is SourceType.CASE
    assert r.year == 1976
    assert r.url == "https://www.courtlistener.com/opinion/109389/ernst/"
    assert any("scienter" in p for p in r.passages)


USLM = """<?xml version="1.0"?>
<uscDoc xmlns="http://xml.house.gov/schemas/uslm/1.0">
  <main>
    <section identifier="/us/usc/t15/s78j">
      <num value="78j">§ 78j.</num>
      <heading>Manipulation of security prices</heading>
      <subsection identifier="/us/usc/t15/s78j/b">
        <num value="b">(b)</num>
        <content>It shall be unlawful for any person to use or employ any manipulative
          or deceptive device or contrivance in contravention of such rules as the
          Commission may prescribe.</content>
      </subsection>
    </section>
  </main>
</uscDoc>"""


def test_usc_uslm_parsing() -> None:
    records = UsCodeIngestor().records_from_uslm(USLM)
    assert len(records) == 1
    r = records[0]
    assert r.type is SourceType.STATUTE
    assert r.id == "usc-15-78j"
    assert r.code == "15 U.S.C."
    assert r.section == "78j"
    assert "Manipulation" in r.title
    assert any("unlawful" in p for p in r.passages)


CFR_XML = """<?xml version="1.0"?>
<CFRGRANULE>
  <DIV1 N="17" TYPE="TITLE"><HEAD>Title 17</HEAD>
    <DIV8 N="240.10b-5" TYPE="SECTION">
      <HEAD>§ 240.10b-5   Employment of manipulative and deceptive devices.</HEAD>
      <P>It shall be unlawful to make any untrue statement of a material fact.</P>
      <P>It shall be unlawful to engage in any act which operates as a fraud.</P>
    </DIV8>
  </DIV1>
</CFRGRANULE>"""


def test_cfr_gpo_parsing() -> None:
    records = CfrIngestor().records_from_gpo_xml(CFR_XML)
    assert len(records) == 1
    r = records[0]
    assert r.type is SourceType.REGULATION
    assert r.id == "cfr-17-240-10b-5"
    assert r.code == "17 C.F.R."
    assert r.section == "240.10b-5"
    assert "manipulative" in r.title.lower()
    assert len(r.passages) >= 1


def test_write_jsonl_roundtrip_and_dedupe(tmp_path: Path) -> None:
    ingestor = UsCodeIngestor()
    records = ingestor.records_from_uslm(USLM) + ingestor.records_from_uslm(USLM)
    out = tmp_path / "corpus.jsonl"

    written = write_jsonl(records, out)
    assert written == 1  # duplicate id collapsed
    assert len(load_corpus(out).records) == 1

    # Appending the same records is idempotent.
    assert write_jsonl(records, out, append=True) == 0


# --------------------------------------------------------------------------- #
# A dissent is not authority
# --------------------------------------------------------------------------- #
def _cap_case(opinions: list[dict]) -> dict:
    return {
        "id": 1,
        "name_abbreviation": "Test v. Test",
        "decision_date": "1999-05-06",
        "citations": [{"type": "official", "cite": "176 F.3d 1132"}],
        "court": {"name": "United States Court of Appeals for the Ninth Circuit"},
        "casebody": {"opinions": opinions},
    }


MAJORITY = (
    "FLETCHER, Circuit Judge: We hold that the challenged regulations constitute a "
    "prior restraint on speech that fails to give adequate procedural safeguards, and "
    "we therefore affirm the judgment of the district court in all respects."
)
DISSENT = (
    "T.G. NELSON, Circuit Judge, dissenting: Source code is a means of commanding a "
    "computer to perform a function, and functionality is not protected expression, so "
    "I would reverse and hold the regulations valid in their entirety."
)


def test_cap_ingest_keeps_only_the_majority() -> None:
    """Joining every opinion made a dissent's language indistinguishable from the
    holding once chunked: the verifier would confirm the quotation as verbatim and
    present the losing argument as the court's."""

    from legal_research.ingest.cap import CapIngestor

    record = CapIngestor().record_from_case(
        _cap_case([
            {"type": "majority", "text": MAJORITY},
            {"type": "dissent", "text": DISSENT},
        ])
    )

    text = " ".join(record.passages)
    assert "prior restraint" in text
    assert "functionality is not protected expression" not in text
    assert "dissenting" not in text


def test_cap_ingest_falls_back_when_no_opinion_is_typed() -> None:
    """Older CAP records carry untyped opinions; those keep the old behaviour."""

    from legal_research.ingest.cap import CapIngestor

    record = CapIngestor().record_from_case(_cap_case([{"text": MAJORITY}]))

    assert "prior restraint" in " ".join(record.passages)


def test_a_withdrawn_opinion_is_not_in_force() -> None:
    """The corpus recorded Bernstein as in_force. The Ninth Circuit's own order
    says otherwise: "The three-judge panel opinion ... is withdrawn.\""""

    from legal_research.citations.corpus import AuthorityStatus, load_corpus

    record = next(
        r for r in load_corpus("data/corpus/openweights.jsonl").records
        if r.id == "bernstein-9th-1999"
    )

    assert record.status is AuthorityStatus.WITHDRAWN
    assert "192 F.3d 1308" in record.status_note
    # And the order that withdrew it is citable in its own right.
    order = next(
        r for r in load_corpus("data/corpus/openweights.jsonl").records
        if r.id == "f3d-192-1308"
    )
    assert any("is withdrawn" in p for p in order.passages)
