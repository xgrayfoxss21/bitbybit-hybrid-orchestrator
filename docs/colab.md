# Google Colab Guide

Run the **BitNet Hybrid Orchestrator** end-to-end with zero local setup. This notebook includes the orchestrator core, safety guard, demo agents, and optional Web UIs (single-turn and multi-turn chat).

**Open the notebook:**  
➡️ <https://colab.research.google.com/github/xgrayfoxss21/bitbybit-hybrid-orchestrator/blob/main/notebooks/bitbybit-hybrid-orchestrator.ipynb****>

> Tip: Run cells **in order**. Cells **6** and **6B** depend on cells **1 → 5**.

---

## Cell-by-cell map

### **Cell 1 — Environment & deps**
- Installs pinned Python packages (CPU-friendly).
- Prints versions and ONNX Runtime providers.

**You should see** a JSON block of versions and a line like:
```

ONNX Runtime providers: \['CPUExecutionProvider']

````

---

### **Cell 2 — TinyBERT Guard (ONNX optional)**
- Provides input/output moderation + regex PII redaction.
- Works in **regex-only** mode by default. If you supply an ONNX model + tokenizer, it upgrades to **onnx+regex**.

**Env vars (optional):**
```python
import os
os.environ["TINYBERT_ONNX_PATH"] = "/content/tinybert-int8.onnx"
os.environ["TINYBERT_TOKENIZER_DIR"] = "/content/tokenizer"
# To force regex-only:
# os.environ["GUARD_DISABLE_ONNX"] = "1"
````

**You should see**:

```
[guard] mode = regex-only
== Guard sanity check ==
...
redacted_text: Contact me at [REDACTED_EMAIL]. ...
```

---

### **Cell 3 — Orchestrator core**

* DAG `Node` model, `Registry` for agents, and `Scheduler`.
* Supports pre/post guard hooks, timeouts, retries, and parallel branches.

*No output unless errors.*

---

### **Cell 4 — Agents**

* Lightweight, deterministic placeholder agents:

  * `summarizer` (extractive)
  * `claimcheck` (heuristic overlap)
  * `synthesis` (executive brief)

* You’ll later swap these with your BitNet backends (same signatures).

*No output unless errors.*

---

### **Cell 5 — Demo run (programmatic)**

* Registers agents, builds mixed DAG:

  ```
  parse → [claim1, claim2] (parallel) → reduce
  ```
* Runs once and prints per-node outputs + moderation cards (if any).

**You should see** sections: `=== parse ===`, `=== claim1 ===`, `=== claim2 ===`, `=== reduce ===`, and a final guard mode line.

---

### **Cell 6 — Web UI (single-turn)**

* Launches a **Gradio** form: paste text → run pipeline once.
* Returns the **Executive Brief**, compact per-node outputs, and moderation JSON.

**Output**: a local URL and a `*.gradio.live` link while the Colab session is active.

---

### **Cell 6B — Chat UI (multi-turn)**

* Launches a **Gradio Chatbot** that **preserves history**.
* Each new user message builds a transcript:

  ```
  User: ...
  Assistant: ...
  User: <new message>
  ```

  and runs the same DAG with guard on input/output every turn.

**Controls**: summary length, editable claims, “show moderation JSON”.

---

## Common issues & fixes

* **SyntaxError with backticks**
  Colab code cells must contain **only Python**. Remove stray \`\`\` fences.

* **`RuntimeError: This event loop is already running`**
  We call `nest_asyncio.apply()` in the UI cells. If you still see it, re-run the UI cell once more after install.

* **No public link**
  Use the local URL printed by Gradio. If Colab blocks pop-ups, allow them and re-run.

* **Blocked/Redacted output**
  The guard tripped a threshold. Adjust in policy:

  ```python
  # in code, not YAML (demo)
  # thresholds: toxicity_block, jailbreak_block, pii_redact
  ```

* **NameError (guard/Registry/Node/agents)**
  Re-run **Cells 1 → 5** (UI cells rely on globals defined earlier).

---

## Swap in your BitNet backends

Keep the function signatures and replace internals:

```python
async def summarizer(text: str, max_sentences: int = 3, **_) -> dict:
    # call your BitNet summarizer (onnx/cpp/accelerated)
    return {"text": "..."}
```

Do the same for `claimcheck` and `synthesis`. The orchestrator and UIs remain unchanged.

---

## Guard modes & configuration

* **Modes**: `regex-only` (default) or `onnx+regex` (with ONNX model + tokenizer).
* **Env vars** (set before importing guard):

  ```python
  os.environ["TINYBERT_ONNX_PATH"] = "/abs/path/to/model.onnx"
  os.environ["TINYBERT_TOKENIZER_DIR"] = "/abs/path/to/tokenizer"
  # or disable:
  os.environ["GUARD_DISABLE_ONNX"] = "1"
  ```
* **Thresholds** (toxicity, jailbreak, PII redaction) are set in the code demo; mirror them in YAML when you externalize the config.

---

## Compliance & safety notes

* **AGPL §13**: If you host a UI/API, expose a **Source** link or `/source` endpoint pointing to the **exact commit**; set an `X-AGPL-Source` header. See **[COMPLIANCE.md](../COMPLIANCE.md)**.
* Use **dummy PII** (`test@example.com`) in demos; the guard redacts emails/phones by default.
* Security reporting: see **[SECURITY.md](../SECURITY.md)** (PGP key + safe-harbor).

---

## See also

* **Quickstart:** [docs/quickstart.md](./quickstart.md)
* **Chat mode:** [docs/chat.md](./chat.md)
* **Architecture:** [docs/architecture.md](./architecture.md)
* **Pipeline API:** [docs/api.md](./api.md)

```
::contentReference[oaicite:0]{index=0}
```

```
::contentReference[oaicite:0]{index=0}
```
