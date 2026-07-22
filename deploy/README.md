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
| finance | banking & finance substance            | 8102         |
| writer  | drafting, synthesis, heavy reasoning   | 8103         |
| router  | tie-break classification (small model) | 8104         |

## 3. Point the backend at the local models

```bash
export LRG_LLM_MODE=openai
export LRG_RETRIEVER_MODE=faiss
export LRG_SUPPORT_SCORER=nli   # semantic entailment verification (gpu extra)
export LRG_PDF_RENDERER=html   # or latex

export LRG_SAUL_BASE_URL=http://127.0.0.1:8101/v1
export LRG_FINANCE_BASE_URL=http://127.0.0.1:8102/v1
export LRG_WRITER_BASE_URL=http://127.0.0.1:8103/v1
export LRG_ROUTER_BASE_URL=http://127.0.0.1:8104/v1
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
    'bash -s -- llama3.1:8b saul:7b-instruct-v1 gemma3:4b' < scripts/model_smoke.sh
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
