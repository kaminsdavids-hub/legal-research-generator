# Deploying on a DGX Spark (fully local)

This guide runs the whole system — four expert models, retrieval, and the studio —
on a single NVIDIA DGX Spark with **no cloud calls**.

## 0. Prerequisites

- NVIDIA driver + CUDA toolkit appropriate for your Spark image.
- Python 3.11+ and Node 18+.
- `uv` (recommended) or `pip`.
- Model checkpoints pulled locally (Hugging Face cache or local paths).

## 1. Install

```bash
EXTRAS="dev,gpu,pdf" scripts/setup.sh
```

The `gpu` extra pulls `faiss-cpu` + `sentence-transformers` for real dense retrieval;
`pdf` pulls WeasyPrint for true PDF output. (Swap `faiss-cpu` for a CUDA faiss build
if you want GPU retrieval.)

## 2. Serve the expert models

Each expert is an OpenAI-compatible server (vLLM). Edit the model IDs at the top of
`scripts/serve_models.sh` to match your local checkpoints, then:

```bash
scripts/serve_models.sh
```

| Expert  | Role                                   | Default port |
| ------- | -------------------------------------- | ------------ |
| saul    | legal reasoning / authority analysis   | 8101         |
| writer  | drafting, synthesis, heavy reasoning   | 8103         |
| gemma   | general-purpose analysis (finance etc) | 8105         |
| hermes  | ideation / brainstorming               | 8106         |
| hermes3 | final grammar / fluency pass           | 8107         |

## 3. Point the backend at the local models

```bash
export LRG_LLM_MODE=openai
export LRG_RETRIEVER_MODE=faiss
export LRG_EMBED_MODEL=nvidia/Nemotron-3-Embed-1B-BF16
export LRG_SUPPORT_SCORER=nli   # semantic entailment verification (gpu extra)
export LRG_PDF_RENDERER=html   # or latex

export LRG_SAUL_BASE_URL=http://127.0.0.1:8101/v1
export LRG_WRITER_BASE_URL=http://127.0.0.1:8103/v1
export LRG_GEMMA_BASE_URL=http://127.0.0.1:8105/v1
export LRG_HERMES_BASE_URL=http://127.0.0.1:8106/v1
export LRG_HERMES3_BASE_URL=http://127.0.0.1:8107/v1

export LRG_GRAMMAR_CHAIN_ENABLED=true
export LRG_GRAMMAR_CHAIN_ROLES=hermes,gemma,hermes3
export LRG_GRAMMAR_CHAIN_FAIL_OPEN=true
```

(Or put these in `.env` at the repo root — see `.env.example`.)

## 4. Run the studio

```bash
scripts/run.sh
# backend  -> http://127.0.0.1:8000
# frontend -> http://127.0.0.1:3000
```

## 5. Build the retrieval corpus

Use `lrg-ingest` to populate the corpus from real sources, then point
`LRG_CORPUS_PATH` at the result. **The verifier only trusts records in this
corpus**, so its coverage defines what the system is allowed to cite.

```bash
lrg-ingest courtlistener --query "scienter 10b-5" --count 50
lrg-ingest cap  --file cap_bulk.jsonl
lrg-ingest usc  --file usc-title15.uslm.xml
lrg-ingest cfr  --file cfr-title17.xml --title 17
lrg-ingest uploads ./uploads --type secondary   # needs the 'ingest' extra for PDFs

export LRG_CORPUS_PATH=data/corpus/corpus.jsonl
```

Install the ingest extra alongside the others on the Spark:
`EXTRAS="dev,gpu,pdf,ingest" scripts/setup.sh`. A CourtListener token (optional)
is read from `LRG_COURTLISTENER_TOKEN`.

## 6. Ops on the Spark: reclaim disk & smoke-test models

Two committed, safe scripts replace ad-hoc inline SSH commands. **Never paste a
here-doc or pass the body as argv** — stream the committed *file* over SSH stdin so
the version-controlled script is the single source of truth. A multiplexed
`spark-node` alias (see `~/.ssh/config`: `ControlMaster auto` + `ControlPersist`)
keeps repeat connections fast.

`scripts/reclaim_hf_cache.sh` — free disk from Hugging Face hub caches. **Dry-run by
default**; refuses any path not directly under the hub cache dir. Set `APPLY=1` to
actually delete.

```bash
# Dry-run (lists sizes, deletes nothing):
ssh -o BatchMode=yes -o ConnectTimeout=10 spark-node \
    'bash -s -- meta-llama/Llama-3.1-70B-Instruct' < scripts/reclaim_hf_cache.sh
# Apply (frees the space; verify with the df line it prints after):
ssh -o BatchMode=yes -o ConnectTimeout=10 spark-node \
    'APPLY=1 bash -s -- meta-llama/Llama-3.1-70B-Instruct' < scripts/reclaim_hf_cache.sh
```

`scripts/model_smoke.sh` — assert each Ollama model returns a real response. Each
call is capped by `curl -m 120` (no hang) and the model is unloaded after; the
script exits non-zero if **any** model fails.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 spark-node \
    'bash -s -- llama3.1:8b saul:7b-instruct-v1 gemma3:4b nemotron-3-nano:4b' < scripts/model_smoke.sh
echo "exit=$?"   # 0 only if every model responded
```

Note: a model can fail the smoke test for lack of free memory while a large vLLM
server holds the GPU (e.g. `llama3.1:8b` needs ~20 GiB) — that is a correctly
detected resource limit, not a script bug.

## Notes

- The two agent runtimes (**OpenHands**, **NemoClaw**) are selectable behind the
  `AgentRuntime` interface; both drive the same nine agents.
- Everything degrades gracefully: with no GPU/models, set `LRG_LLM_MODE=mock` and
  `LRG_RETRIEVER_MODE=mock` and the full pipeline still runs deterministically.


## Deploying the frontend on Netlify

The Next.js app is a static export on Netlify; `/api/*` is served by
`netlify/functions/api-proxy.mts`, which holds the backend key server-side so it
never reaches the browser. Configuration lives in `netlify.toml`; the secrets go
in the Netlify UI, never in the repo:

| variable | what it is |
| --- | --- |
| `BACKEND_URL` | `https://host:port` of this backend, routable **from Netlify's servers** |
| `BACKEND_API_KEY` | the backend's `LRG_API_KEY`. Server-side only — never `NEXT_PUBLIC_*` |
| `PROXY_PASSWORD` | what a caller must send as `X-Proxy-Password` |
| `PROXY_ALLOW_PUBLIC` | `true` only if the site is gated some other way |

Two secrets, deliberately. The proxy password authorises calls to the proxy and
can be rotated without touching the host; the backend key authorises calls to
the host and never leaves Netlify's environment. A proxy that attached the key
to anonymous requests would not secure the backend, it would republish it at a
new address, which is why the function returns 503 when neither
`PROXY_PASSWORD` nor `PROXY_ALLOW_PUBLIC` is set.

### The timeout, and the job API that answers it

Netlify synchronous functions cap at **10 seconds by default and 26 at most**,
while this backend's budget is `LRG_LLM_TIMEOUT_SECONDS=300`. A five-minute
request cannot be made to fit under a 26-second ceiling however it is written,
so the slow steps stopped being requests:

```
POST /api/sessions/{id}/jobs   {"step": "draft"}   -> 202 {"job_id": ...}
GET  /api/jobs/{job_id}                            -> {"state": "running"|"succeeded"|"failed"}
GET  /api/sessions/{id}                            -> the result, fetched once on success
```

Every call is now fast by construction, so every hop in front of the backend
sees only short requests. `runStep()` in `frontend/lib/api.ts` is the client
half; it polls every 2 seconds and reports each state change, because a step
that takes four minutes with no feedback is indistinguishable from a hang.

Steps available as jobs: `run-all`, `brainstorm`, `ideate`, `outline`,
`research`, `draft`, `voice`, `verify`, `format`, `novelty`, `mechanism`,
`multi-chat`, `dialectic`, `socratic`. Every route that drives a model is now
one of these; nothing a proxy carries is long-running.

The last three are different in kind. They answer a question rather than
advancing the manuscript, so the whole response comes back as the job's `result`
and there is nothing to fetch afterwards — and they are generally **not
exclusive**. Several can run at once, including while a draft is in flight.
Putting them under the one-writer-per-session rule would have removed something
a user can already do, hold two conversations, in order to prevent a corruption
they cannot cause.

**`socratic` is the exception that shapes the rule.** It answers a question about
a paragraph and, when `apply_revision` is set, rewrites it. So the same step is
read-only on one call and exclusive on the next, and whether a submission writes
is a property of the *request*, not of the step name. `_mutates()` in `app.py` is
where that is decided. A name-based rule would either lock out concurrent
questions that harm nothing, or let a revision land while a draft is rewriting
the same section.

`run-all` is the only one that takes arguments (`idea`, `title`, `max_ideas`),
and the only one whose result is not simply the blackboard — the per-agent step
log and the shippable verdict exist nowhere else, so the job carries them in a
small `result` object. Its client half is `runAll()`. Note what a partial
failure leaves behind: `pipeline.run_all` mutates the session in place and has
per-stage fallbacks, so a run that dies part-way leaves the session advanced as
far as it got. That is observable — fetch the session and see which stages
completed — and it is preferable to rolling back to a snapshot, which would
discard work the run really did.

**Two properties worth knowing before relying on this.** Jobs live in the
backend process's memory and die with it, exactly like the sessions they mutate
— a restart loses both, so this is not a queue and must not be treated as one.
And only one step runs per session at a time: steps mutate a shared blackboard
in place, so a second submission gets a 409 rather than being queued, which
would hide from the caller that their step had not started.

**Still synchronous:** the original routes for everything above. They are
unchanged, still tested, and correct for a loopback client with nothing in
between.

### Streaming, and what it is actually worth here

`GET /api/jobs/{id}/events` is a Server-Sent Events stream: the client holds one
connection and the outcome arrives the moment it exists, instead of at the next
poll. `awaitJob()` in the client tries it first and falls back to polling, so the
fast path is used where it works and the durable one is always there.

Two limits, both real:

**It streams events about a job, not tokens of an answer.** The engines are batch
internally — `multi_chat.chat` runs a five-model panel and returns when the last
one finishes, and nothing above the LLM client calls its `stream()`. Token
streaming would mean threading a callback through the engines, and for
multi-chat it would help only at the very end, since the final answer is
synthesised after the panel completes. What this removes is the poll interval,
not the wait.

**It does not get past a function timeout.** A cap on a function invocation
applies to a streamed response as much as a buffered one — Netlify allows 26
seconds however the body is produced — so an answer that takes a minute dies
mid-stream behind the proxy. `GET /api/jobs/{id}` remains the path that works
there. The stream is for clients talking to the backend directly, over loopback
or the tailnet, which is also where the interactive steps feel worst under
polling.

### Reachability

`BACKEND_URL` must resolve and route from Netlify's build and function
infrastructure. A `*.ts.net` tailnet-only address does **not**: Netlify is not
on your tailnet. Making the proxy work therefore means exposing the backend to
the public internet again (Tailscale Funnel, or another ingress) — this time
behind `LRG_API_KEY`, which is the point of the exercise. The function reports
this case explicitly rather than returning a bare 502.
