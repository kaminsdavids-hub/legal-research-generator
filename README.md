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
   ├─ Mechanism gate            not an agent: a deterministic scan (plus an
   │                            optional critic) that refuses prose resting on
   │                            an operation it never states
   ▼
 PDF / HTML manuscript + verification report
```

- **Runtimes.** All nine agents run behind a single `AgentRuntime` interface, with two
  implementations: **OpenHands** (primary orchestrator, plan-first) and **NemoClaw**
  (second cooperating runtime). Agents are runtime-agnostic.
- **Expert pool.** A router sends each request to the right local model
  (a legal model "Saul", a general-purpose analyst "Gemma", a writer/reasoner,
  an ideation engine "Hermes" backed by Nemotron Nano, and a final grammar pass
  role "Hermes3") with an appropriate decoding policy (`cold` / `balanced` / `hot`).
- **Everything is injectable.** The default runtime profile targets local live
  models + dense retrieval; tests pin deterministic mock backends so CI runs
  with no GPU and no network.

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
ollama pull gemma3:4b
ollama pull nemotron-3-nano:4b
ollama pull hermes3:8b

# SaulLM-7B (Equall/Saul-7B-Instruct-v1) — pull the GGUF from Hugging Face and
# give it a clean local name:
ollama pull hf.co/mradermacher/Saul-7B-Instruct-v1-GGUF:Q5_K_M
ollama cp   hf.co/mradermacher/Saul-7B-Instruct-v1-GGUF:Q5_K_M saul:7b-instruct-v1
```

Default role mapping:

| Role | Model | Used for |
| --- | --- | --- |
| Legal (Saul) | `saul:7b-instruct-v1` | Legal doctrine (forced-cite, cold) |
| Writer | `llama3.1:8b` | Section prose / synthesis / editing |
| Gemma | `gemma3:4b` | General-purpose analysis + router tie-break |
| Hermes | `nemotron-3-nano:4b` | Ideation / brainstorming (Ideator), reasoning |
| Hermes3 | `hermes3:8b` | Final grammar/fluency pass in the post-draft chain |

Post-draft grammar correction runs sequentially by default:

1. `hermes` (Nemotron Nano)
2. `gemma`
3. `hermes3`

Override any endpoint or model in `.env` (copy from `.env.example`). To run fully mocked again, set `LRG_LLM_MODE=mock`. The table above is the *code* default; a deployment with more GPU headroom will typically point these roles at larger models in `.env` — on the Spark, `writer` runs `gpt-oss:20b` and `gemma` runs `gemma4:latest`.

### The Model Jury (secondary chat)

The chat module does not ask one model. It runs a **panel** of five, synthesizes
a single answer, then hands that answer to **two verifiers** that critique it and
score its grounding — seven model activations per query.

| Slot | Default model | `.env` key |
| --- | --- | --- |
| Panel | `gpt-oss:20b` | `LRG_MULTI_CHAT_GPT_OSS_MODEL` |
| Panel | `gemma4:latest` | `LRG_MULTI_CHAT_GEMMA4_MODEL` |
| Panel | `apertus:latest` | `LRG_MULTI_CHAT_APERTUS_MODEL` |
| Panel | `nemotron-3-nano:4b` | `LRG_MULTI_CHAT_NEMOTRON_MODEL` |
| Panel | `hermes3:8b` | `LRG_MULTI_CHAT_HERMES3_MODEL` |
| Verifier | `gemma3:4b` | `LRG_MULTI_CHAT_VERIFIER_GEMMA3_MODEL` |
| Verifier | `saul:7b-instruct-v1` | `LRG_MULTI_CHAT_VERIFIER_SAUL_MODEL` |

```bash
ollama pull gpt-oss:20b
ollama pull gemma4:latest
ollama pull apertus:latest
```

Panel membership is configuration, not code — `LRG_MULTI_CHAT_PANEL_MEMBERS`
selects which of the five actually get queried, and all five stay available to
the rescue and verifier paths regardless.

**This is the part that needs tuning before it works.** Ollama's stock
`OLLAMA_MAX_LOADED_MODELS` is 3, and seven models do not fit in three slots — the
server evicts and reloads mid-query, and the slower panel members blow past
`LRG_MULTI_CHAT_PER_MODEL_TIMEOUT_SECONDS` and get replaced with canned text. On
the Spark the seven weigh 52.8 GB of 121 GB unified memory, so the fix is server
configuration rather than hardware: see
[deploy/README.md](deploy/README.md#ollama-server-settings-the-model-jurys-real-constraint).
Confirm with `ollama ps` during a query — you want 7 resident, not 3.

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
| `LRG_RETRIEVER_MODE` | `faiss` | `mock` (lexical) or `faiss` (faiss + sentence-transformers) |
| `LRG_EMBED_MODEL` | `nvidia/Nemotron-3-Embed-1B-BF16` | embedding model used by retrieval and embedding support scorer |
| `LRG_SUPPORT_SCORER` | `nli` | Verifier support test: `lexical`, `embedding`, or `nli` (entailment) |
| `LRG_PDF_RENDERER` | `html` | `html` (WeasyPrint if installed) or `latex` |
| `LRG_CITATION_STYLE` | `bluebook` | citation style |
| `LRG_CORPUS_PATH` | `data/corpus/sample_corpus.jsonl` | retrieval corpus |

Model endpoints (when `LRG_LLM_MODE=openai`) are configured per expert — see
`.env.example` for the `LRG_SAUL_*`, `LRG_WRITER_*`, `LRG_GEMMA_*`,
`LRG_HERMES_*`, and `LRG_HERMES3_*` variables.

Grammar-chain behavior is controlled by:

- `LRG_GRAMMAR_CHAIN_ENABLED`
- `LRG_GRAMMAR_CHAIN_ROLES` (default: `hermes,gemma,hermes3`)
- `LRG_GRAMMAR_CHAIN_FAIL_OPEN`

---

## Anti-hallucination guarantees

- **Grounding at write time.** The Writer must call the `CitationGuard`, which grounds
  a proposition against the retriever. A proposition with no support is **dropped**,
  never invented, and reported as a blocked assertion.
- **Adversarial verification.** The Verifier re-checks every cite: the source must
  resolve in the corpus, must actually support the proposition, and any quote must
  appear verbatim. Failures are marked `REMOVED` and stripped from the manuscript.
- **Semantic support test.** "Actually supports" is a pluggable scorer. The live
  default profile uses `LRG_SUPPORT_SCORER=nli`, a cross-encoder that scores
  entailment P(passage ⊨ proposition), catching misattributed holdings that
  merely share vocabulary. If semantic deps are unavailable (e.g., CI), scoring
  falls back to deterministic lexical recall. See
  `src/legal_research/citations/support.py`.
- **Shippability.** A paper cannot ship while any citation is unverified. The final
  document contains only verified authorities, plus an appended verification report.
- **No black-box mechanisms.** A passage may *discuss* AI; its argument may not
  *rest* on one. "The algorithm determines eligibility" names a decider and no
  decision rule, and the legal test the paper then applies — a particular
  employment practice, a *Daubert* methodology, an agency's stated basis — has
  nothing to attach to. A deterministic scan flags mechanism-position uses (agent,
  instrument, cause, or legal conclusion), an optional LLM critic may add findings
  and can never clear one, and one bounded rewrite is accepted only if a re-scan
  shows it worked. What could not be repaired is reported in the same appendix as
  the citation results. Ported from the patent generator's `patentgen.blackbox`,
  where the rule is absolute; here only mechanism position fires, because a paper
  about the EU AI Act says "machine learning" on every page and should. See
  `src/legal_research/mechanism/blackbox.py`.

See `tests/test_citation_guard.py`, `tests/test_verifier.py`,
`tests/test_support_scorer.py`, and `tests/test_mechanism.py`.

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
LRG_LLM_MODE=openai \
LRG_RETRIEVER_MODE=faiss \
LRG_EMBED_MODEL=nvidia/Nemotron-3-Embed-1B-BF16 \
scripts/run.sh
```

If you plan to use the Model Jury, set `OLLAMA_MAX_LOADED_MODELS=8` and
`OLLAMA_NUM_PARALLEL=1` on the Ollama service first — at the stock values the
seven-model jury thrashes and silently degrades to a smaller panel. Drop-in files
and the measurement behind those numbers are in
[deploy/README.md](deploy/README.md#ollama-server-settings-the-model-jurys-real-constraint).

---

## Evaluating the lineup (and a frontier control)

The dialectic roles must come from four distinct model families — same-family
debaters have correlated errors and produce agreement dressed as debate — and the
eval judge is a fifth. Validating a lineup costs nothing:

```bash
scripts/preflight_models.py --no-probe     # families only: no calls, no key
scripts/preflight_models.py                # + one tiny completion per role
```

When a run scores badly, the open question is whether the limit is the local
models or the pipeline. The control arm answers it by holding retrieval, the
corpus, the gates, the judge and two debate roles fixed and changing only which
model drafts the thesis:

```bash
export LRG_DIALECTIC_THESIS_API_KEY='sk-...'
scripts/frontier_control.sh
```

It refuses to start without the key rather than falling back to a local model: a
"frontier" number produced by `llama3.1` would corrupt the only comparison the
run exists to make. Any role can be moved this way with
`LRG_DIALECTIC_<ROLE>_BASE_URL` / `_API_KEY`; a role served off this machine must
carry its own key, and prompt text for it leaves the machine (logged at WARNING).

---

## Learning how *you* argue

The Socratic loop (`modules/maieutic/`) already adapts which questions it asks to
which ones you answer — see `maieutic policy`. Capturing the exchanges themselves,
so a model can later be fine-tuned on your reasoning rather than the average of
everyone's, is separate and **off by default**: it writes what you wrote,
verbatim, into a file that outlives the manuscript.

```bash
export LRG_MAIEUTIC_TRAINING_CAPTURE=1
export LRG_MAIEUTIC_TRAINING_PATH=.maieutic/training.jsonl

maieutic training                                   # what has accumulated
maieutic training --export sft.jsonl                # chat-format SFT samples
maieutic training --export sft.jsonl --merged-only  # only gate-clean exchanges
```

Your answers are the training target; the machine's synthesis is excluded unless
you ask for it. Answers whose patch the gates refused are captured too — training
only on what the machinery liked would narrow the sample to its own preferences —
and whether to use them is a decision made at export, in the open.

---

## Project layout

```
src/legal_research/
  agents/       nine agents + runtime interface (OpenHands / NemoClaw)
  citations/    corpus, retriever, verifier, support scorer, Bluebook formatter, report
  ingest/       corpus ingestion: CourtListener, CAP, USC/CFR, uploads (lrg-ingest)
  llm/          client protocol, deterministic mock, OpenAI-compat client, pool
  mechanism/    the AI-as-mechanism gate: deterministic scan, LLM critic, repair
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
