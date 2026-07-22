# Legal Research Generator

A local, multi-agent studio that turns a raw idea into a **grounded, human-voiced
legal research paper** — with every legal assertion tied to a real, retrieved,
verified authority. Designed to run entirely on a local machine (e.g. an NVIDIA DGX
Spark) with **no cloud LLMs**, and to run fully offline in tests via a deterministic
mock backend.

> **Disclaimer.** Output is a research aid, not legal advice, and may be wrong. Every
> citation must be independently confirmed before any reliance.

---

## Why this exists

Law-review-style writing has two failure modes that generative models make worse:

1. **Hallucinated authority** — plausible-looking cites to cases that do not exist or
   do not hold what they are cited for.
2. **"AI voice"** — hedge-heavy, transition-clustered, uniform-cadence prose that
   reads like a machine and says little.

This project attacks both directly. Authorities can **only** enter through a
retriever over a fixed corpus; an adversarial verifier removes anything it cannot
confirm verbatim; and a voice lint rejects the tell-tale patterns of machine prose.

---

## Architecture

```
raw idea
   │
   ▼
 Router ──picks expert + decoding policy (legal→cold+RAG, brainstorm→hot)
   │
   ▼
 Blackboard  ◀────────── shared state for all agents ──────────▶
   │
   ├─ 1. Socratic Interviewer   sharpen thesis / scope / novelty
   ├─ 2. Ideator                candidate angles (hot decoding)
   ├─ 3. Legal Researcher       authority map (supporting + contrary), retrieval only
   ├─ 4. Finance Analyst        banking/finance substance tied to the argument
   ├─ 5. Argument Architect     outline (IRAC/CREAC), roadmap, rebuttal
   ├─ 6. Writer / Voice         distinctive prose; cites via retriever; blocks ungrounded
   ├─ 7. Editor / Reviser       line edits, revision requests, anti-AI-voice pass
   ├─ 8. Citation Formatter     Bluebook footnotes, short forms, table of authorities
   └─ 9. Verifier               adversarial gate: removes any unverifiable cite
   │
   ▼
 PDF / HTML manuscript + verification report
```

- **Runtimes.** All nine agents run behind a single `AgentRuntime` interface, with two
  implementations: **OpenHands** (primary orchestrator, plan-first) and **NemoClaw**
  (second cooperating runtime). Agents are runtime-agnostic.
- **Expert pool.** A router sends each request to the right local model
  (a legal model "Saul", a finance model, a writer/reasoner, and a small router model)
  with an appropriate decoding policy (`cold` / `balanced` / `hot`).
- **Everything is injectable** and defaults to a deterministic mock so the full
  pipeline runs with no GPU and no network.

---

## Quick start (mock backend, no GPU)

```bash
# 1. Create an environment (Python 3.11+). uv is recommended:
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# 2. Run the end-to-end demo (writes ./paper/)
.venv/bin/lrg-demo

# 3. Start the API
.venv/bin/lrg-api            # http://127.0.0.1:8000

# 4. Run the tests
.venv/bin/pytest
```

The demo prints each agent step, the citation ledger (verified/removed), the
anti-AI-voice lint per section, and writes `paper/manuscript.md`,
`paper/verification_report.md`, `paper/paper.html`, and a rendered document.

---

## Run the studio (UI + API)

```bash
scripts/setup.sh   # installs backend (.venv) + frontend (npm)
scripts/run.sh     # backend :8000 + frontend :3000 (localhost only)
```

Open http://localhost:3000.

### Live mode with Ollama

The backend defaults to `LRG_LLM_MODE=openai` and routes every expert role through a local Ollama OpenAI-compatible endpoint (`http://127.0.0.1:11434/v1`). Make sure Ollama is running and the expected models are pulled:

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b

# SaulLM-7B (Equall/Saul-7B-Instruct-v1) — pull the GGUF from Hugging Face and
# give it a clean local name:
ollama pull hf.co/mradermacher/Saul-7B-Instruct-v1-GGUF:Q5_K_M
ollama cp   hf.co/mradermacher/Saul-7B-Instruct-v1-GGUF:Q5_K_M saul:7b-instruct-v1
```

Default role mapping:

| Role | Model |
| --- | --- |
| Router | `llama3.1:8b` |
| Legal (Saul) | `saul:7b-instruct-v1` |
| Finance | `llama3.1:8b` |
| Writer | `qwen2.5-coder:7b` |

Override any endpoint or model in `.env` (copy from `.env.example`). To run fully mocked again, set `LRG_LLM_MODE=mock`.

### Share on your local network

```bash
scripts/serve_lan.sh   # auto-detects your LAN IP, binds both servers to 0.0.0.0
```

It prints a shareable URL like `http://192.168.x.y:3000` that any device on the same
Wi-Fi/LAN can open. Override the detected address with `LAN_IP=192.168.x.y scripts/serve_lan.sh`.

> **macOS firewall.** If remote devices can't connect, the Application Firewall is
> likely blocking incoming connections. Allow Node + Python under
> *System Settings → Network → Firewall*, or run
> `sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setblockall off`.

---

## Configuration

Copy `.env.example` to `.env` and adjust. Key switches:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LRG_LLM_MODE` | `openai` | `mock` or `openai` (any local OpenAI-compatible server) |
| `LRG_RETRIEVER_MODE` | `mock` | `mock` (lexical) or `faiss` (faiss + sentence-transformers) |
| `LRG_SUPPORT_SCORER` | `lexical` | Verifier support test: `lexical`, `embedding`, or `nli` (entailment) |
| `LRG_PDF_RENDERER` | `html` | `html` (WeasyPrint if installed) or `latex` |
| `LRG_CITATION_STYLE` | `bluebook` | citation style |
| `LRG_CORPUS_PATH` | `data/corpus/sample_corpus.jsonl` | retrieval corpus |

Model endpoints (when `LRG_LLM_MODE=openai`) are configured per expert — see
`.env.example` for the `LRG_SAUL_*`, `LRG_FINANCE_*`, `LRG_WRITER_*`, `LRG_ROUTER_*`
variables.

---

## Anti-hallucination guarantees

- **Grounding at write time.** The Writer must call the `CitationGuard`, which grounds
  a proposition against the retriever. A proposition with no support is **dropped**,
  never invented, and reported as a blocked assertion.
- **Adversarial verification.** The Verifier re-checks every cite: the source must
  resolve in the corpus, must actually support the proposition, and any quote must
  appear verbatim. Failures are marked `REMOVED` and stripped from the manuscript.
- **Semantic support test.** "Actually supports" is a pluggable scorer. The default
  is deterministic lexical recall (CI); on the Spark, `LRG_SUPPORT_SCORER=nli` swaps
  in a cross-encoder that scores entailment P(passage ⊨ proposition), catching
  misattributed holdings that merely share vocabulary. See
  `src/legal_research/citations/support.py`.
- **Shippability.** A paper cannot ship while any citation is unverified. The final
  document contains only verified authorities, plus an appended verification report.

See `tests/test_citation_guard.py`, `tests/test_verifier.py`, and
`tests/test_support_scorer.py`.

---

## Building the retrieval corpus

The Verifier only trusts records in the corpus, so its coverage defines what the
system may cite. Build it from real sources with `lrg-ingest` (writes/append
`CorpusRecord` JSONL; point `LRG_CORPUS_PATH` at the result):

```bash
# Case law
lrg-ingest courtlistener --query "scienter 10b-5" --count 25   # CourtListener v4
lrg-ingest cap --file cap_bulk.jsonl                           # Caselaw Access Project bulk

# Statutes & regulations
lrg-ingest usc --file usc-title15.uslm.xml                     # U.S. Code (USLM XML)
lrg-ingest cfr --file cfr-title17.xml --title 17               # CFR (eCFR/GPO XML)

# The scholar's own documents
lrg-ingest uploads ./briefs ./memos --type secondary          # PDF/TXT/MD/HTML
```

A CourtListener API token (optional, lifts rate limits) is read from
`LRG_COURTLISTENER_TOKEN`. PDF uploads require the `ingest` extra
(`pip install -e '.[ingest]'`). All network access is behind an injectable
`Fetcher`, so ingestion is fully unit-tested offline (`tests/test_ingest.py`).

---

## Running on a DGX Spark (local models)

See `deploy/` for setup and run scripts. In short:

```bash
scripts/setup.sh      # create venv, install with the gpu extra
scripts/serve_models.sh   # launch local OpenAI-compatible model servers
LRG_LLM_MODE=openai LRG_RETRIEVER_MODE=faiss scripts/run.sh
```

---

## Project layout

```
src/legal_research/
  agents/       nine agents + runtime interface (OpenHands / NemoClaw)
  citations/    corpus, retriever, verifier, support scorer, Bluebook formatter, report
  ingest/       corpus ingestion: CourtListener, CAP, USC/CFR, uploads (lrg-ingest)
  llm/          client protocol, deterministic mock, OpenAI-compat client, pool
  pdf/          renderer interface + HTML and LaTeX backends
  voice/        anti-template, anti-AI-voice, novelty checks
  router.py     request → expert + decoding policy
  blackboard.py shared state
  pipeline.py   end-to-end orchestration
  api/          FastAPI app (REST + WebSocket streaming)
  demo.py       CLI end-to-end demo
tests/          pytest suite (mock backend) + golden HTML
frontend/       Next.js studio UI
data/corpus/    sample legal corpus (JSONL)
```

---

## License

MIT. See `LICENSE`.
