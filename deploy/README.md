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

### Ollama server settings (the Model Jury's real constraint)

The jury activates **seven models per query** — five panel slots plus two
verifiers. All seven are warmed regardless of how many the panel actually
queries, so narrowing `LRG_MULTI_CHAT_PANEL_MEMBERS` does not reduce what must be
resident. They must all be *co-resident*, or Ollama evicts and reloads models
mid-query and charges that reload time to whichever models are still generating.
This is not a disk question: on the Spark the seven weigh 52.8 GB against 121 GB
of unified memory, so there is ample room, and the only thing standing between a
three-model panel and a four-model panel is these two server settings.

Ollama's stock `OLLAMA_MAX_LOADED_MODELS` is **3**, which is far too low here.
Set the drop-ins (`/etc/systemd/system/ollama.service.d/`), then reload:

```bash
sudo tee /etc/systemd/system/ollama.service.d/limits.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=8"
Environment="OLLAMA_CONTEXT_LENGTH=8192"
EOF
```

```bash
sudo tee /etc/systemd/system/ollama.service.d/parallel.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
EOF
```

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Why these values:

| Setting | Value | Reason |
| --- | --- | --- |
| `OLLAMA_MAX_LOADED_MODELS` | 8 | 7 models per jury query, plus one slot of slack. At 6 the two oldest were evicted every query. |
| `OLLAMA_NUM_PARALLEL` | 1 | Each loaded model reserves KV cache per parallel slot. At 2, seven models wanted roughly double the cache for concurrency the panel never uses — it queries *different* models, not one model repeatedly. |
| `OLLAMA_CONTEXT_LENGTH` | 8192 | Fits the panel prompt plus the authority packet. |

Verify residency during a live jury query — this should show 7, not 3:

```bash
ollama ps
```

Measured on this box (2026-08-16), five concurrent panel members, all resident:
`hermes3` 25.3s, `apertus` 47.1s, `nemotron` 70.2s, `gemma4` 75.0s, `gpt-oss`
80.9s — all inside the 110s per-model cap, whole exchange 139.5s. Before the
change, under eviction, `gpt-oss` took 101s and `gemma4`/`nemotron` missed the
cap entirely. Confirmed live as well: sampling `ollama ps` through a real HTTP
jury request showed 7 resident in 31 of 40 samples, the remainder being warmup
ramp rather than eviction.

If you deploy to a box with less unified memory, shrink
`LRG_MULTI_CHAT_PANEL_MEMBERS` rather than raising
`LRG_MULTI_CHAT_PER_MODEL_TIMEOUT_SECONDS` — a model that times out is replaced
with canned text, which is a worse answer than not asking it.

**Clearing the cap is not the same as answering.** The shipped panel is four,
not five: with all seven resident and no contention, `nemotron-3-nano:4b` still
returned `Degraded panel answer (… low-substance rewrite fallback)` well inside
the cap. That is a substance failure, and no amount of memory or timeout fixes
it. When tuning a panel, read what the slots contain — the timings alone will
tell you a canned answer arrived on schedule.

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
    'bash -s -- gpt-oss:20b gemma4:latest apertus:latest nemotron-3-nano:4b hermes3:8b gemma3:4b saul:7b-instruct-v1' < scripts/model_smoke.sh
echo "exit=$?"   # 0 only if every model responded
```

Those seven are the jury's five panel members plus its two verifiers — the set
that must be co-resident. Smoke-testing them together is also the cheapest way
to confirm the `OLLAMA_MAX_LOADED_MODELS` setting above actually took.

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

**It streams progress, not tokens.** `multi_chat.chat` now takes an `on_event`
callback and reports the panel as it runs — `panel_started` with the model list,
`model_answered` per model as each finishes, then `synthesising`. Those arrive
in *completion* order rather than the configured order the results are collected
in, because completion order is what the person waiting sees. A five-model panel
that took forty seconds and showed nothing now shows the first model at around
eight.

It is still not token streaming. The final answer is synthesised after the whole
panel finishes, so streaming its tokens would help only at the very end; the
per-model events are worth more and cost far less. The callback is invoked from
the panel's pool threads and wrapped, so a subscriber that raises cannot break
the exchange.

`dialectic` reports a sequence rather than a panel: `generating` and
`position_generated` per side, then `retrieved`, `verified`,
`cruxes_extracted`, `synthesising`. Its stages are heterogeneous — a model call,
a network round-trip to CourtListener, an NLI pass — so a spinner cannot
distinguish a slow debate from a hung one, and the stage name is the useful part
rather than the timing. `verified` carries `calls_spent`, which is what tells a
reader whether verification did any work at all; `cruxes_extracted` reports a
count of zero along with the engine's note explaining why, because an empty crux
table is a real outcome and reporting nothing would look like a stage that never
ran.

`run-all` reports `step_completed` per agent, with a 1-based `index` so a
display can show "step 4 of n" and a `degraded` flag set when that stage fell
back rather than succeeding. Every stage goes through the same recorder,
fallbacks included: a run where six of nine agents fell back is exactly the run
a caller most needs to see happening, and reporting only the healthy path would
make a limping run look like a fast one. Do not filter those events out.

Progress is carried on the poll as well as the stream, so falling back to
polling costs the immediacy and not the information.

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
