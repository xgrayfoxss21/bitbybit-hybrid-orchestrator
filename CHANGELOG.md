# Changelog

All notable changes to **BitByBit Hybrid Orchestrator** will be documented in this file.

The format is based on **Keep a Changelog**, and this project adheres to **Semantic Versioning**.

> Repo: [https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator](https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator)
> License: AGPL-3.0-or-later

---

## \[Unreleased]

### Added

* **Chat mode (multi-turn)**:

  * Colab **Cell 6B — Chat Demo** (Gradio `Chatbot` with rolling transcript).
  * Local app **`ui/chat_gradio.py`** (multi-turn chat; shows moderation JSON + guard mode).
  * Sample chat pipeline **`orchestrator/pipeline.chat.yml`** with conversation window/persist flags.
  * New doc **`docs/chat.md`** (guide, config, troubleshooting).
* **Compliance**:

  * **`COMPLIANCE.md`** with copy-paste snippets (HTTP header, `/source` route, UI footers, Docker labels) updated for the `xgrayfoxss21/bitbybit-hybrid-orchestrator` repo.
* **Security**:

  * Maintainer PGP key path **`security/pgp/xgrayfoxss21.asc`** and updated **SECURITY.md** contact block (**[troubleshooting@foxesden.xyz](mailto:troubleshooting@foxesden.xyz)**).
* **Docs site**:

  * Navigation update and pages for **Quickstart**, **Colab**, **Architecture**, **API**, **Chat**.
* **Community**:

  * GitHub Discussions setup (categories + template stubs).

### Changed

* **README.md**: rebranded to **BitByBit** with new repo links, added **Project Website** badge linking to [https://bit.foxesden.xyz/](https://bit.foxesden.xyz/); updated CI/security badges.
* **docs/architecture.md**: clarified flow + fixed Mermaid diagram; added **Chat mode** sequence.
* **docs/api.md**: added `conversation` schema (windowing, persistence, redaction).
* **docs/\_config.yml**: footer includes commit link (AGPL §13 hint) and site nav polish.
* **Notebook**:

  * Rewrote **Cells 1–6** for clarity and stability; added **guard v0.2** (env knobs, ONNX disable flag).
  * Web UIs: single-turn (Cell 6) and multi-turn chat (Cell 6B).
* **Orchestrator core**:

  * Scheduler: clearer error taxonomy, shallow-merge semantics documented, small retry/backoff tidy.
  * Agents: deterministic placeholders with compact outputs for reducer.

### Fixed

* Mermaid parse error in Architecture diagram.
* Colab `SyntaxError` caused by stray markdown fences inside code cells.
* Notebook event-loop issues by applying `nest_asyncio` in UI cells.
* Minor typos in badges/links and repo paths.

### Security

* Documented PGP fingerprint and public key location.
* Reinforced guidance to redact logs and disable telemetry by default.

### Deprecated / Removed

* None.

### Breaking Changes

* None.

---

## \[0.1.0] - 2025-09-10

### Added

* Initial public skeleton:

  * **Hybrid DAG orchestrator** (Node/Registry/Scheduler) with per-node **pre/post guard** hooks.
  * **TinyBERT Guard v0.1** (regex PII + heuristic jailbreak; optional ONNX scoring).
  * **Demo agents**: `summarizer`, `claimcheck`, `synthesis`.
  * **Colab notebook** with Cells 1–5 and optional single-turn Web UI (Cell 6).
  * **Docs**: README, Quickstart outline, Safety, Roadmap.
  * **Governance**: CONTRIBUTING, CODE\_OF\_CONDUCT, SECURITY, license headers.
  * **Issue templates** for bugs and features.

---

## Links

* Compare **Unreleased**: [https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator/compare/v0.1.0...HEAD](https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator/compare/v0.1.0...HEAD)
* Tag **v0.1.0**: [https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator/releases/tag/v0.1.0](https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator/releases/tag/v0.1.0)

> When you cut a new release, update the links above and replace placeholders with the actual tag/commit.
