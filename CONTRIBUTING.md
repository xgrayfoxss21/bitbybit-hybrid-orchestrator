# Contributing to BitByBit Hybrid Orchestrator

First off—thank you! 🎉 Contributions help this project evolve into a robust, edge‑ready orchestration stack.

This guide explains how to propose changes, the coding standards, and the legal bits (AGPL + DCO).

---

## 📜 Code of Conduct

By participating you agree to uphold our [CODE\_OF\_CONDUCT.md](CODE_OF_CONDUCT.md). Be kind, be constructive, and assume good intent.

---

## 🧭 Ways to Contribute

* **Bug reports:** use **Issues → Bug report** template.
* **Feature requests / design proposals:** use **Issues → Feature request** template or start a **Discussion**.
* **Docs:** fixes, clarifications, or new pages in `docs/`.
* **Code:** improvements to the orchestrator, agents, guard, tests, or tooling.
* **Security:** see [SECURITY.md](SECURITY.md) (never file public issues for vulnerabilities).

Before large changes, please open an issue to align early.

---

## 🛠️ Development Setup

**Requirements**

* Python **3.10+**
* Git
* (Optional) Node is *not* required; site uses GitHub Pages.

**Clone & create a virtualenv**

```bash
git clone https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator.git
cd bitbybit-hybrid-orchestrator
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r orchestrator/requirements.txt
```

**Run the demo (local/Colab)**

* Open the Colab notebook (zero setup):

  * [https://colab.research.google.com/drive/1WZw9Ro4L1X9pZP34VEbJ0pTs5IbdPREv?usp=sharing](https://colab.research.google.com/drive/1WZw9Ro4L1X9pZP34VEbJ0pTs5IbdPREv?usp=sharing)

* Or run the Python orchestrator skeleton directly from your scripts / notebook environment.

---

## 🧑‍💻 Coding Standards

* **Python**

  * Follow PEP8 where sensible; prefer **type hints** and **docstrings**.
  * Keep functions small and testable; prefer pure functions for guards and planners.
  * Structure agent outputs as **typed dicts / pydantic models** where practical.
* **Config**

  * Pipelines are **config‑as‑data** in YAML (`orchestrator/pipeline.yml`).
  * Keep defaults minimal; prefer explicit params over hidden globals.
* **Docs**

  * Update `docs/` when behavior, flags, or user workflows change.
* **Dependencies**

  * Be conservative adding deps; verify licenses are compatible and list them in `THIRD_PARTY_LICENSES.md`.

---

## ✅ Commit & PR Process

1. **Fork** → create a feature branch:

   ```bash
   git checkout -b feat/my-awesome-change
   ```
2. **Write tests** when possible (unit / golden outputs).
3. **Conventional Commits** for messages:

   * `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `perf:`, `ci:`
   * Examples:

     * `feat(orchestrator): add memory-aware concurrency cap`
     * `fix(guard): apply PII redaction before logging`
4. **DCO sign‑off** each commit (see below).
5. **PR checklist**

   * [ ] Tests pass locally
   * [ ] Docs updated (`docs/*` and/or README)
   * [ ] `THIRD_PARTY_LICENSES.md` updated if you added deps or model weights
   * [ ] `COMPLIANCE.md` touched if network behavior/headers changed
   * [ ] SPDX header present in new files
   * [ ] Security considerations noted (e.g., guard thresholds, input/output handling)
6. Open a PR and link to the related issue. Keep PRs focused & reviewable.

**CI must pass** before maintainers review.

---

## 🧾 DCO (Developer Certificate of Origin)

We use the **DCO** instead of a CLA. Sign off each commit to confirm you have the right to contribute:

```bash
git commit -s -m "feat: add DAG reduce step"
```

This appends a line like:

```
Signed-off-by: Your Name <your.email@example.com>
```

If you forgot, you can amend:

```bash
git commit --amend -s
git push --force-with-lease
```

---

## 🧩 SPDX License Headers (AGPL)

All source files **must** start with:

```
SPDX-License-Identifier: AGPL-3.0-or-later
```

This is required for compliance and automated license scanning.

---

## 🔐 Security & Responsible Disclosure

* Don’t disclose vulnerabilities publicly—see [SECURITY.md](SECURITY.md).
* If your changes affect **network‑served** behavior, ensure **AGPL §13** compliance is intact:

  * `/source` endpoint / header still returns the correct commit,
  * `COMPLIANCE.md` remains accurate.

---

## 📦 Models & Third‑Party Assets

* **Do not** commit large model weights to the repo.
* Document the **source, license, and version** for:

  * model weights,
  * datasets,
  * third‑party code snippets.
* Add entries to `THIRD_PARTY_LICENSES.md`.
* Ensure licenses are compatible with **AGPL‑3.0‑or‑later** distribution.

---

## 🧪 Testing (Guidelines)

* Unit test planning heuristics, guard decisions (inputs/outputs), and DAG scheduling (parallel vs sequential).
* Add **golden outputs** for stable tasks (e.g., summarizer placeholders).
* Prefer **deterministic** test settings (fixed seeds, no network).

*(We’ll add a test harness shortly—feel free to include one in your PR.)*

---

## 🗂️ Branches & Releases

* Work happens on feature branches → PR → **main**.
* Releases are tagged; update **CHANGELOG.md** using “Keep a Changelog” style.
* Use labels (`feat`, `fix`, `breaking`) to help generate release notes.

---

## 🔏 Signed Commits (optional but encouraged)

If you use GPG:

```bash
git config --global user.signingkey <YOUR_KEY_ID>
git config --global commit.gpgsign true
```

(See **SECURITY.md** for PGP details.)

---

## 🧰 Helpful Scripts

You can add or run local helpers to keep things tidy (suggested):

```bash
# format (if you install black/ruff locally)
python -m pip install black ruff
ruff check .
black .
```

---

## 🧠 Design Principles (tl;dr)

* **Edge‑first:** small, fast, local‑friendly.
* **Safety‑by‑default:** guardrails as pure functions; moderate input and output.
* **Config‑as‑data:** pipelines are YAML‑driven; no magic.
* **Observability:** minimal JSON traces, moderation cards, reproducible runs.
* **Modularity:** agents with a tiny interface (`infer()`, `tool_call()`), easy to swap BitNet backend.

---

## 🙌 Attribution

By contributing, you agree your contributions will be licensed under the project’s license (**AGPL‑3.0‑or‑later**).

Thanks for making BitByBit Hybrid Orchestrator better! 🦊
