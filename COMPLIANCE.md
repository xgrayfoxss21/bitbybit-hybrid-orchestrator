# COMPLIANCE.md

**Project:** BitByBit Hybrid Orchestrator
**License:** **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)**
**Repo:** [https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator](https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator)

This document tells you how to run, host, and redistribute this project **in compliance with the AGPL** and how to track third-party licenses, models, and data.

---

## 1) What the AGPL requires (in plain English)

* The AGPL is a copyleft license. If you **modify** this software and make it available to users **over a network** (e.g., a web UI or API), you **must** provide those users access to the **Corresponding Source** of the running version.
* “Corresponding Source” = your complete source code for the modified work, including build scripts and interface code to run it (but **not** the build tools themselves).
* If you distribute binaries or containers, the same obligation applies: provide the Corresponding Source.

👉 Full text: see **[LICENSE](LICENSE)** (AGPL-3.0-or-later). The key part for hosted services is **§13 (“Remote Network Interaction”)**.

---

## 2) Minimum obligations when you host this project

If you host a modified or unmodified version accessible over a network, do **all** of the following:

1. **Link to Source (exact commit) visible to users**

   * Add a footer link or “About/Source” menu entry pointing to the **exact commit** that’s running.
   * Example URL:
     `https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator/tree/<COMMIT_SHA>`

2. **Expose the commit in responses (header or endpoint)**

   * Send an HTTP response header such as:

     ```
     X-AGPL-Source: https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator/tree/<COMMIT_SHA>
     ```
   * Optionally, expose a `GET /source` endpoint that returns the same info in JSON.

3. **Make your modifications available**

   * If you’ve changed anything, ensure that users can access **your fork/branch** (and build instructions) from that source link.

4. **Include license notices**

   * Keep the top-level **LICENSE** file.
   * Preserve SPDX headers in source files (see below).

---

## 3) Implementation recipes (copy-paste)

### 3.1 FastAPI middleware + `/source` route

```python
# app.py
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse

REPO  = "https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator"
COMMIT = os.getenv("APP_COMMIT_SHA", "HEAD")

app = FastAPI()

@app.middleware("http")
async def agpl_header(req, call_next):
    resp = await call_next(req)
    resp.headers["X-AGPL-Source"] = f"{REPO}/tree/{COMMIT}"
    return resp

@app.get("/source")
async def source():
    return JSONResponse({
        "license": "AGPL-3.0-or-later",
        "repo": REPO,
        "commit": COMMIT,
        "url": f"{REPO}/tree/{COMMIT}"
    })
```

Run with:

```bash
export APP_COMMIT_SHA=$(git rev-parse HEAD)
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3.2 Gradio footer (UI cell / script)

```python
# After creating your Gradio Blocks `demo`
REPO  = "https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator"
import os
COMMIT = os.getenv("APP_COMMIT_SHA", "HEAD")
demo.footer = f"Source: [{COMMIT[:7]}]({REPO}/tree/{COMMIT}) • License: AGPL-3.0-or-later"
```

### 3.3 CLI notice

```python
# at startup or --version
import os
REPO="https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator"
COMMIT=os.getenv("APP_COMMIT_SHA","HEAD")
print(f"[AGPL] Source: {REPO}/tree/{COMMIT}")
```

### 3.4 Docker labels + env

```dockerfile
# Dockerfile
ARG COMMIT_SHA
LABEL org.opencontainers.image.source="https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator"
LABEL org.opencontainers.image.revision="${COMMIT_SHA}"
ENV  APP_COMMIT_SHA="${COMMIT_SHA}"
```

Build:

```bash
docker build --build-arg COMMIT_SHA=$(git rev-parse HEAD) -t bitbybit-orch:$(git rev-parse --short HEAD) .
```

### 3.5 GitHub Actions (bake commit SHA)

```yaml
# .github/workflows/ci.yml (snippet)
env:
  APP_COMMIT_SHA: ${{ github.sha }}

steps:
  - name: Echo AGPL Source
    run: echo "X-AGPL-Source=https://github.com/${{ github.repository }}/tree/${{ github.sha }}"
```

### 3.6 Notebooks (Colab header cell)

```python
# Compliance header cell (put near the top)
import os, json
REPO  = "https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator"
COMMIT = os.getenv("APP_COMMIT_SHA","HEAD")
print("AGPL: Source:", f"{REPO}/tree/{COMMIT}")
```

### 3.7 GitHub Pages (Just-the-Docs footer)

Add to `docs/_config.yml`:

```yaml
footer_content: >
  Source: <a href="https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator/tree/{{ site.data.build.commit | default: 'HEAD' }}">commit</a>
  &nbsp;•&nbsp; License: AGPL-3.0-or-later
```

(Optionally populate `site.data.build.commit` at build time.)

---

## 4) SPDX headers

Keep SPDX identifiers in source files (Python example):

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
```

This is already recommended in **README.md** and ensures machine-readable license tagging.

---

## 5) Third-party software and model compliance

Track external components in **THIRD\_PARTY\_LICENSES.md**:

* Libraries (PyPI, system libs) with names, versions, and licenses.
* **Model weights** (TinyBERT, BitNet variants, etc.) with their **original license** (Apache-2.0, MIT, OpenRAIL, custom).
* Datasets or evaluation sets with their licenses/terms.

### Example entry (model)

```md
- TinyBERT ONNX (variant: *tinybert-int8*), source: <link>, license: Apache-2.0
  - Notes: Used for classification-based moderation; stored locally only.
```

> **Important:** Model weights are often licensed separately from your code. Respect those terms. If redistribution is restricted, provide instructions to **download from the original source**, not the weights themselves.

---

## 6) Data protection & privacy practices

This project includes a **Guard** that can redact PII (emails/phones) on input/output. To stay aligned with privacy best practices:

* **Disable telemetry** in environments by default:

  * We set `HF_HUB_DISABLE_TELEMETRY=1` in Colab cell 1.
* **Logging**: avoid storing raw prompts/outputs that contain PII. If logs are needed, store the **redacted** forms the Guard produced.
* **Retention**: delete ephemeral logs/state when a session ends (Colab, Gradio demo).
* **Security reporting**: refer researchers to **[SECURITY.md](SECURITY.md)** (PGP key + safe-harbor).

---

## 7) Releasing & attestation

* **Tag releases** and consider **signed tags** using your maintainer PGP key:

  ```bash
  git tag -s v0.1.0 -m "BitByBit Hybrid Orchestrator v0.1.0"
  git push --tags
  ```
* Public key is at: `security/pgp/xgrayfoxss21.asc`. Contributors can verify signatures with:

  ```bash
  gpg --show-keys --with-fingerprint security/pgp/xgrayfoxss21.asc
  git verify-tag v0.1.0
  ```

---

## 8) Compliance checklist (pre-deploy)

* [ ] UI shows a **Source** link to the **exact commit** of the running code.
* [ ] HTTP responses include `X-AGPL-Source` header.
* [ ] `/source` endpoint (or equivalent) is enabled.
* [ ] **LICENSE** present; SPDX headers intact.
* [ ] **THIRD\_PARTY\_LICENSES.md** updated (libs + models + datasets).
* [ ] If you modified code, your **fork/branch** is public and linked.
* [ ] Telemetry disabled (if applicable); logs redact PII; retention reasonable.
* [ ] **SECURITY.md** is reachable; PGP key published.
* [ ] Container images carry `org.opencontainers.image.source` and `revision` labels.

---

## 9) Placeholders to replace

Search and replace these where you’ve copy-pasted snippets:

* `<COMMIT_SHA>` → the real commit hash (e.g., from `git rev-parse HEAD` or `${{ github.sha }}`)
* Ensure repository URL matches your canonical repo:
  `https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator`

---

## 10) Questions

* Legal nuance about AGPL? Read the full text in **[LICENSE](LICENSE)** and consult counsel if needed.
* Security or privacy issues? See **[SECURITY.md](SECURITY.md)** (PGP key + workflow).
