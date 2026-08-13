# Running the maieutic loop as a service

`maieutic-web.service` in this directory is the unit currently running on the
DGX Spark, copied from the box rather than written from memory. Installing it
needs **no root**.

## Why a user unit

The obvious approach is a system unit in `/etc/systemd/system/`, which needs
`sudo`. On this Spark `sudo` requires a password, so that would have made the
deployment something only the operator could finish.

A *user* unit avoids it entirely, provided lingering is on — which it already
was, and which is what makes a user service start at boot and survive logout:

```bash
loginctl show-user "$(whoami)" -p Linger      # expect Linger=yes
loginctl enable-linger "$(whoami)"            # if not
```

The Spark's other services (`legal-research-api`, `patent-studio`,
`ollama-local-11435`) are user units too, so this follows the box's existing
pattern rather than introducing a second one.

## Install

```bash
mkdir -p ~/.config/systemd/user
cp deploy/maieutic-web.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now maieutic-web.service
```

Verify it is genuinely up rather than merely reported up. `Type=simple` marks the
unit active as soon as the process spawns, and startup loads the support
scorer's weights first, so "active" precedes "serving" by roughly 35 seconds:

```bash
systemctl --user status maieutic-web
curl -s -o /dev/null -w '%{http_code}\n' localhost:8015/     # expect 200
```

Two checks worth doing once, because they are the reason the unit exists:

```bash
# the environment the unit actually handed the process, not what you meant to set
tr '\0' '\n' < /proc/$(systemctl --user show maieutic-web -p MainPID --value)/environ \
  | grep -E '^(MAIEUTIC_LIVE|LRG_CORPUS_PATH)='

# that it really comes back
kill -9 $(systemctl --user show maieutic-web -p MainPID --value)
sleep 15 && systemctl --user is-active maieutic-web
```

## Configuration

| variable | effect |
|---|---|
| `MAIEUTIC_LIVE=1` | runs the real dialectic engine **and** `Gates.live`. The two are selected together on purpose: a live run behind offline gates refuses every authority for want of a verifier (REMEDIATION §12.2). Unset, the loop runs offline and gates on structure alone. |
| `LRG_CORPUS_PATH` | which corpus grounding and banality see. `openweights.jsonl` is the topic-relevant one. **Do not point this at `openweights_fulltext.jsonl` while `support_scorer` is `lexical`** — that scorer is recall of the claim's tokens with no length penalty, so long passages inflate it and the fabrication wall gets *weaker* (OBSERVABLES D29). |
| `OLLAMA_KEEP_ALIVE` | how long Ollama holds models resident. Low values make every exchange pay the load cost. |

## Binding

The unit binds `127.0.0.1`, so nothing is exposed. The loop has no authentication
and its endpoints write to disk, so an unauthenticated writable service on a LAN
interface is not something to switch on without deciding to. Reach it over a
tunnel:

```bash
ssh -L 8015:localhost:8015 spark    # then http://localhost:8015
```

## A live answer is a job

An exchange takes minutes (236–294s measured on a Mac; faster on the Spark with
models resident). `POST /answer` therefore returns **202** with a job and the
client polls `GET /answer`; offline the same endpoint answers **200**
synchronously. A proxy in front of this must not impose a short read timeout on
the polling endpoint, and must not buffer it.

## Ordering against Ollama

A user unit cannot reliably order against Ollama, which is a system service. The
unit does not pretend otherwise: a start before Ollama is up is handled by
`Restart=on-failure` with `RestartSec=10` rather than by an ordering directive
that would not be honoured.
