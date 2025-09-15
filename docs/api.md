# Pipeline API

This document specifies the **YAML schema** used by the BitNet Hybrid Orchestrator to build and run a mixed DAG (sequential + parallel).  
Files typically live in `orchestrator/pipeline.yml` (single-turn) and `orchestrator/pipeline.chat.yml` (chat).

- Minimal, human-readable YAML
- Safe defaults
- Backward-compatible evolution via `version` and `schema`

---

## Quick example (single-turn)

```yaml
version: 0.1.0
schema: pipeline.v1
name: summarize_and_verify

budgets: { latency_ms: 1800, max_concurrency: 2, memory_mb: 1200 }

models:
  reasoner: bitnet-s-1.58b
  guard: tinybert-onnx-int8

policies:
  thresholds:
    toxicity_block: 0.5
    pii_redact: 0.7
    jailbreak_block: 0.6

nodes:
  - id: parse
    agent: bitnet.summarizer
    guard_pre: true
    guard_post: true
    params: { max_sentences: 3 }

  - id: claim1
    agent: bitnet.claimcheck
    deps: [parse]
    params: { claim: "BitNet uses 1.58-bit weights" }

  - id: claim2
    agent: bitnet.claimcheck
    deps: [parse]
    params: { claim: "TinyBERT is effective for classification" }

  - id: reduce
    agent: bitnet.synthesis
    deps: [claim1, claim2]
````

---

## Quick example (chat mode)

```yaml
version: 0.1.0
schema: pipeline.v1
name: chat_orchestrator

budgets: { latency_ms: 2000, max_concurrency: 2, memory_mb: 1200 }

models: { reasoner: bitnet-s-1.58b, guard: tinybert-onnx-int8 }

policies:
  thresholds: { toxicity_block: 0.5, pii_redact: 0.7, jailbreak_block: 0.6 }

conversation:
  kind: transcript          # "transcript" (default) or "none"
  window_messages: 12       # keep last N (user,assistant) pairs
  persist: false            # if true, store per-session history server-side
  redact_pii_in_history: true

nodes:
  - { id: parse,  agent: bitnet.summarizer,  guard_pre: true, guard_post: true, params: { max_sentences: 3 } }
  - { id: claim1, agent: bitnet.claimcheck, deps: [parse], params: { claim: "BitNet uses 1.58-bit weights" } }
  - { id: claim2, agent: bitnet.claimcheck, deps: [parse], params: { claim: "TinyBERT is effective for classification" } }
  - { id: reduce, agent: bitnet.synthesis,  deps: [claim1, claim2] }
```

---

## Top-level fields

| Key            | Type         | Required | Default          | Notes                                                      |
| -------------- | ------------ | -------- | ---------------- | ---------------------------------------------------------- |
| `version`      | string       | ✅        | —                | Pipeline file version (e.g., `0.1.0`).                     |
| `schema`       | string       | ✅        | —                | Current schema id: `pipeline.v1`.                          |
| `name`         | string       | ✅        | —                | Human-friendly name.                                       |
| `description`  | string       | ❌        | `""`             | Optional description.                                      |
| `budgets`      | object       | ❌        | see defaults     | Latency, concurrency, memory hints.                        |
| `models`       | object       | ❌        | `{}`             | Logical labels for reasoner/guard.                         |
| `policies`     | object       | ❌        | see defaults     | Guard thresholds and other policies.                       |
| `conversation` | object       | ❌        | `{ kind: none }` | Chat behavior (transcript window, persistence, redaction). |
| `nodes`        | array\<Node> | ✅        | —                | DAG steps (see **Node**).                                  |

### `budgets`

```yaml
budgets:
  latency_ms: 1800          # soft target per request
  max_concurrency: 2        # scheduler semaphore
  memory_mb: 1200           # informational (not enforced)
```

### `models`

Purely descriptive labels you can map to your runtime:

```yaml
models:
  reasoner: bitnet-s-1.58b
  guard: tinybert-onnx-int8
```

### `policies.thresholds`

Guard thresholds (mirrored by the notebook demo defaults):

```yaml
policies:
  thresholds:
    toxicity_block: 0.5
    pii_redact: 0.7
    jailbreak_block: 0.6
```

### `conversation` (chat)

```yaml
conversation:
  kind: transcript          # "transcript" | "none"
  window_messages: 12       # keep last N (user,assistant) pairs
  persist: false            # only relevant for server deployments
  redact_pii_in_history: true
```

---

## Node schema

Each `nodes[]` entry has:

| Field         | Type   | Required | Default | Description                                     |
| ------------- | ------ | -------- | ------- | ----------------------------------------------- |
| `id`          | string | ✅        | —       | Unique node identifier.                         |
| `agent`       | string | ✅        | —       | Registry key, e.g., `bitnet.summarizer`.        |
| `deps`        | array  | ❌        | `[]`    | Upstream node IDs; empty means a **root** node. |
| `guard_pre`   | bool   | ❌        | `true`  | Run guard on this node’s input.                 |
| `guard_post`  | bool   | ❌        | `true`  | Run guard on this node’s output text.           |
| `timeout_ms`  | int    | ❌        | `1000`  | Per-attempt timeout.                            |
| `max_retries` | int    | ❌        | `0`     | Retries on failure (simple backoff).            |
| `params`      | object | ❌        | `{}`    | Agent-specific parameters.                      |

**Agent contract (Python):**

```python
# Registered under the key from `agent`
async def agent(**kwargs) -> dict:
    # Must return at least {"text": "<primary string payload>"}
    return {"text": "..."}
```

---

## Data flow & merge semantics

* At runtime, the scheduler **topologically** executes nodes whose `deps` are satisfied.
* Each child’s input is a **shallow merge** of all parent results plus the top-level `sources` given at run time:

  1. Start with `sources` (e.g., `{"text": user_input}`).
  2. Merge each parent result dict (later parents overwrite earlier keys if colliding).
  3. Merge `node.params` last (params override prior keys).

---

## Guard & moderation

The guard is invoked at two points per node when enabled:

1. **Pre-guard** — `guard.check(text, mode="input")`

   * May redact PII; may **block** if thresholds exceeded.
2. **Post-guard** — `guard.check(text, mode="output")`

   * May redact or block.

When invoked, the guard adds a **moderation card** to the node result under `"_moderation"`.

### Moderation card structure

```json
{
  "node": "parse:post",
  "mode": "output",
  "guard_version": "v0.2",
  "allowed": true,
  "text": "possibly redacted text",
  "labels": { "toxicity": 0.02, "jailbreak": 0.00, "pii": 1.00 },
  "actions": ["redact"],
  "redactions": [ { "span": [14, 31], "type": "PII.email" } ],
  "why": "ok"
}
```

---

## Result map & reserved keys

The scheduler returns a dict: **`{ node_id: result_dict }`**.
Each `result_dict` can contain:

| Key           | Type   | Meaning                                                   |
| ------------- | ------ | --------------------------------------------------------- |
| `text`        | string | Primary payload for that node.                            |
| `_node`       | string | Node id that produced the result.                         |
| `_error`      | string | If execution failed (timeout, exception, or guard block). |
| `_moderation` | array  | List of moderation cards created by pre/post guards.      |
| other keys…   | any    | Agent-specific outputs (e.g., `evidence`, `scores`).      |

---

## Environment variables (guard)

These control the TinyBERT-style guard behavior at runtime:

```bash
# Provide ONNX model + tokenizer directory to enable learned signals.
TINYBERT_ONNX_PATH=/abs/path/to/tinybert-int8.onnx
TINYBERT_TOKENIZER_DIR=/abs/path/to/tokenizer

# Force regex-only mode (disable ONNX even if paths exist).
GUARD_DISABLE_ONNX=1
```

If unset, the guard runs in **regex-only** mode (PII redaction + jailbreak heuristics).

---

## Validation rules

* `nodes[].id` must be unique.
* Every dependency in `nodes[].deps` must reference an existing node id.
* DAG must be acyclic (scheduler errors as `dag_unresolved_nodes` if cycles prevent readiness).
* Agents referenced by name must be registered in the runtime **Registry**.

---

## Error taxonomy (runtime)

* `timeout:<node_id>:<ms>` — node exceeded its timeout.
* `node_failed:<node_id>:<ExceptionType>:<message>` — agent error after retries.
* `blocked_pre|blocked_post:<reason>` — guard decision (e.g., `jailbreak_block`).
* `dag_unresolved_nodes:[...]` — cyclic deps or permanently blocked parents.

Downstream nodes receive partial merges of parents that did produce results; you can design agents to handle missing inputs.

---

## Programmatic run (shape)

**Run inputs** (provided by your app/adapter):

```python
sources = {"text": "… user or transcript text …"}
results = await scheduler.run_dag(nodes, sources)
final = results.get("reduce", {}).get("text", "")
```

**Chat adapter** concatenates the rolling transcript into `sources.text` before calling the DAG.

---

## Extending the schema

* Add new node types by registering agents; **no schema change** is needed.
* New policy fields should be **optional** with safe defaults.
* If you add required fields, bump `schema` (e.g., `pipeline.v2`) and keep a compatibility path for v1.

---

## Compliance & security

* If you host a UI/API, **AGPL §13** requires exposing the running commit’s source. See **[COMPLIANCE.md](../COMPLIANCE.md)** for copy-paste headers, routes, and UI footers.
* PII handling: the example guard redacts emails/phones by default; extend taxonomy as needed.
* Security reporting: **[SECURITY.md](../SECURITY.md)** (PGP key + workflow).

---

```
::contentReference[oaicite:0]{index=0}
```

```
::contentReference[oaicite:0]{index=0}
```
