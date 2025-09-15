```markdown
---
title: Chat mode
nav_order: 6
description: Multi-turn chat UI that reuses the hybrid pipeline with input/output guard each turn.
---

# Chat mode (multi-turn)

**Chat mode** lets you converse with the orchestrator over multiple turns. Each new message is appended to a rolling **transcript**:

```

User: ...
Assistant: ...
User: <new message>

````

That transcript is fed into the same hybrid DAG:  
`parse → [claim1, claim2] (parallel) → reduce`, with **TinyBERT Guard** on **input** and **output** every turn.

- Guard can **redact PII** before processing.
- Output is moderated again prior to display.
- Works in **Colab** or **locally** with Gradio.

---

## Quickstart

### A) Google Colab (recommended first run)

1. Open the notebook:  
   <https://colab.research.google.com/gist/ShiySabiniano/a34e01bcfc227cddc55a6634f1823539/bitnet_tinybert_orchestrator_colab.ipynb>
2. Run **Cells 1 → 5**.
3. Run **Cell 6B — Chat Demo**. Click the link (`*.gradio.live`) and start chatting.

> The chat UI builds a transcript from history + your new message, then calls the DAG with `sources={"text": transcript}`.

### B) Local (Gradio app)

Install core + UI deps:
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r orchestrator/requirements.txt
pip install -r ui/requirements.txt
````

Launch chat:

```bash
python ui/chat_gradio.py
```

Open the printed local URL (and/or the temporary public link). Type messages; each turn runs the hybrid pipeline with guard on input/output.

---

## How it works

### Transcript assembly

The chat UI formats the rolling history like:

```
User: hello
Assistant: Hi—what can I help with?
User: verify these points about BitNet and TinyBERT
```

This **transcript** becomes the `text` input to the pipeline’s root node (`parse`). The reducer’s output becomes the assistant reply for that turn.

### Guard (safety)

* **Pre-guard** runs on the transcript (redacts PII; can block if thresholds exceeded).
* **Post-guard** runs on the synthesized reply (may redact or block).
* Without an ONNX model, guard is **regex-only** (PII + jailbreak heuristics). With ONNX + tokenizer, it upgrades to **onnx+regex**.

Environment knobs:

```bash
# Optional: provide ONNX model + tokenizer to enable learned signals
TINYBERT_ONNX_PATH=/abs/path/to/tinybert-int8.onnx
TINYBERT_TOKENIZER_DIR=/abs/path/to/tokenizer

# Or force regex-only mode
GUARD_DISABLE_ONNX=1
```

---

## Configuration

You can keep chat fully UI-driven, or describe it in YAML.

**`orchestrator/pipeline.chat.yml`**

```yaml
version: 0.1.0
schema: pipeline.v1
name: chat_orchestrator
budgets: { latency_ms: 2000, max_concurrency: 2, memory_mb: 1200 }
models: { reasoner: bitnet-s-1.58b, guard: tinybert-onnx-int8 }
policies:
  thresholds: { toxicity_block: 0.5, pii_redact: 0.7, jailbreak_block: 0.6 }

conversation:
  kind: transcript            # transcript | none
  window_messages: 12         # keep last N user/assistant pairs
  persist: false              # set true only if you add a server with storage
  redact_pii_in_history: true # store the redacted transcript

nodes:
  - { id: parse,  agent: bitnet.summarizer, guard_pre: true, guard_post: true, params: { max_sentences: 3 } }
  - { id: claim1, agent: bitnet.claimcheck, deps: [parse], params: { claim: "BitNet uses 1.58-bit weights" } }
  - { id: claim2, agent: bitnet.claimcheck, deps: [parse], params: { claim: "TinyBERT is effective for classification" } }
  - { id: reduce, agent: bitnet.synthesis,  deps: [claim1, claim2] }
```

* The **UI** mirrors these thresholds even if you don’t load the YAML.
* The chat adapter truncates history to `conversation.window_messages`.

See **[Pipeline API](./api.md)** for the full schema.

---

## Customizing the chat

* **Claims**: Edit “Claim 1/2” fields in the UI; add more branches by extending the DAG.
* **Summary length**: Adjust “Summary sentences” (affects the `summarizer` agent).
* **Swap in BitNet**: Keep function signatures (`async def summarizer(**kwargs) -> dict`), route to your BitNet runtime.
* **RAG evidence**: Replace the dummy list in `claimcheck` with DuckDB/FAISS retrieval.

---

## Troubleshooting

* **NameError: guard/Registry/Node not defined** → Run cells **1 → 5** first (UI cells reuse those globals).
* **“event loop already running”** in Colab → We call `nest_asyncio.apply()` in the UI cell; re-run the cell once after install.
* **No public link appears** → Use the local URL; in Colab, allow pop-ups and re-run.
* **Blocked output** → Your turn hit a guard threshold. Lower `toxicity_block`/`jailbreak_block` cautiously or revise the prompt.
* **Performance** → Demo is CPU-only. Attach accelerated BitNet backends when ready.

---

## Compliance & Security

If you host the chat over a network, **AGPL §13** requires exposing the **Corresponding Source** for the running commit. Use an HTTP header + `/source` endpoint and/or a UI footer link. Copy-paste snippets: **[COMPLIANCE.md](../COMPLIANCE.md)**.

For vulnerability reporting and PGP details: **[SECURITY.md](../SECURITY.md)**.

---

## See also

* **Quickstart:** [docs/quickstart.md](./quickstart.md)
* **Colab Guide:** [docs/colab.md](./colab.md)
* **Architecture:** [docs/architecture.md](./architecture.md)
* **Pipeline API:** [docs/api.md](./api.md)
* **Local chat app:** `ui/chat_gradio.py`

```
::contentReference[oaicite:0]{index=0}
```

```
::contentReference[oaicite:0]{index=0}
```
