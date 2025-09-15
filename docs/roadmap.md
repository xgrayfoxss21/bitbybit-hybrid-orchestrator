---
nav_order: 6
title: Roadmap
---

# Roadmap

This roadmap tracks the evolution of the **BitNet Hybrid Orchestrator** from a documented blueprint to a practical, edge-ready stack.  
Each phase lists **goals**, **exit criteria**, **deliverables**, and **risk notes** so you can make clear go/no-go decisions.

> See also: [Quickstart](./quickstart.md) • [Architecture](./architecture.md) • [Safety](./safety.md)

---

## Phase 0 — Foundation (Blueprint → Working Skeleton)
**Goal:** Run a minimal mixed-DAG with pre/post guards and example agents.

**Deliverables**
- Colab demo: _Summarize → parallel claim checks → synthesis_  
  → [Launch notebook](https://colab.research.google.com/gist/ShiySabiniano/a34e01bcfc227cddc55a6634f1823539/bitnet_tinybert_orchestrator_colab.ipynb)
- Orchestrator core (`Scheduler`, `AgentRegistry`, `Node` model)
- TinyBERT Guard (ONNX if available; regex PII fallback)
- `orchestrator/pipeline.yml` (config-as-data)
- Docs (this site), license, compliance, security

**Exit Criteria**
- End-to-end demo runs locally and in Colab
- Guard redacts emails/phones; moderation card attaches to outputs
- YAML → DAG loader executes with node-level `deps`, `params`, and guards

**Risks / Notes**
- Different TinyBERT ONNX variants have mismatched label orders → verify and map labels

---

## Phase 1 — Core Intelligence (Runnable MVP)
**Goal:** Replace placeholders with a real **BitNet** backend and harden the DAG.

**Deliverables**
- BitNet adapter (e.g., `bitnet.cpp` or ONNX EP wrapper)
- Device profiles (phone/SBC/VPS) with memory-aware concurrency caps
- Node controls: `timeout_ms`, `max_retries`, `cost_est`, `mem_est`
- Minimal test suite for scheduler + guard (golden tests for demo pipeline)

**Exit Criteria**
- p50 E2E under device-appropriate targets (documented per profile)
- No OOM with `max_concurrency` set by profile
- Tests reliable (deterministic) on CI

**Risks / Notes**
- BitNet builds vary per platform; document build flags and fallbacks (CPU first)

---

## Phase 2 — Retrieval & Observability
**Goal:** Add a local, privacy-friendly RAG and minimal tracing.

**Deliverables**
- DuckDB + FAISS option for small KBs (ingest CLI + query helpers)
- Structured traces (JSON) with PII-redaction in logs
- Provenance capture: show which leaf results feed the reducer
- `policy_tags` surfaced from guard to orchestrator decisions

**Exit Criteria**
- Claim checks read from a local KB and return evidence
- Traces show per-node timings, guard decisions, and provenance
- Redaction verified in logs with synthetic PII samples

**Risks / Notes**
- Keep indexes small for edge devices; document size/latency trade-offs

---

## Phase 3 — Learning Loop (Optional)
**Goal:** Create a lightweight improvement loop without sending data off-device by default.

**Deliverables**
- Local evaluation harness (prompt set + labels) for guard thresholds & agent outputs
- Metrics: precision/recall for PII, toxicity, jailbreak; task-specific accuracy
- Optional fine-tune hooks (TinyBERT) or calibration files (no weights committed)

**Exit Criteria**
- Threshold tuning reproducible with a small eval set
- Report that shows guard TPR/FPR deltas before/after tuning

**Risks / Notes**
- Respect licenses for any training data; prefer synthetic or user-provided sets

---

## Phase 4 — Autonomous Logic (Planner Enhancements)
**Goal:** Make the orchestrator more self-directed while staying resource-aware.

**Deliverables**
- Heuristics: critical-path bias, parallelize independent leaves, serialize under pressure
- Retry/fallback trees per node (safe templates on blocked outputs)
- Goal memory (lightweight): carry short-term goals across a single user session

**Exit Criteria**
- Planner improves p95 latency on multi-leaf graphs without increasing OOMs
- Safe fallbacks demonstrated when post-guard blocks

**Risks / Notes**
- Avoid unbounded recursion or uncontrolled task generation on edge devices

---

## Phase 5 — User Interface (Optional)
**Goal:** Provide a minimal operator view; keep it light.

**Deliverables**
- Tiny web dashboard or TUI displaying: live DAG status, traces, moderation cards
- `/source` endpoint + `X-AGPL-Source` header (see COMPLIANCE)

**Exit Criteria**
- Operator can see per-node state, timings, and guard decisions for a run
- AGPL §13 compliance verifiable in the UI and API headers

**Risks / Notes**
- UI must not leak raw PII—show redacted spans only

---

## Feature Gates (Definition of Done)

| Gate | Checks |
|---|---|
| **Boot** | All deps install; demo runs on CPU (no GPU assumed) |
| **Guard** | PII redaction works; jailbreak/toxicity thresholds enforce policy; moderation card attached |
| **DAG** | Parallel leaves + sequential chain both execute; retries/timeouts honored |
| **BitNet** | Real backend swaps in; deterministic tests pass |
| **RAG** | Local KB queried; provenance recorded |
| **Observability** | JSON traces redact PII; performance metrics captured |
| **Compliance** | `/source` and `X-AGPL-Source` expose exact commit; LICENSE & THIRD_PARTY_LICENSES updated |

---

## KPIs

- **Latency:** p50/p95 E2E per device profile
- **Reliability:** success rate; retry count; blocked outputs recovered by safe templates
- **Safety:** PII redaction recall; jailbreak/toxicity FPR at set thresholds
- **Resource:** peak memory; `max_concurrency` scaling
- **RAG:** hit rate / evidence quality (manual spot checks acceptable at small scale)

---

## Backlog (Nice-to-Have)

- Hardware-backed subkeys (YubiKey/Nitrokey) guide for maintainers
- Model caching & lazy loading; adapter pooling
- CLI: `orchestrator run --pipeline orchestrator/pipeline.yml --input file.txt`
- Expand PII patterns (postal addresses, IBAN) behind a toggle
- Localized safe-templates

---

## Risks & Mitigations

- **Model/license drift:** Track versions & licenses in `THIRD_PARTY_LICENSES.md`; pin hashes where possible.  
- **Guard mismatch across variants:** Add label-mapping layer + calibration tests.  
- **Edge OOM:** Memory-aware concurrency + early serialization; document max input sizes.  
- **PII in logs:** Keep `tracing.redact_pii_in_traces: true`; test with seeded PII strings.

---

## Links

- 📚 [Architecture](./architecture.md)  
- 🛡️ [Safety & Guardrails](./safety.md)  
- 🧩 [Pipeline schema & examples](./api.md)  
- 🔐 [Security policy](../SECURITY.md) • [AGPL §13 Compliance](../COMPLIANCE.md)  
- 🎒 Colab: [Run the demo](https://colab.research.google.com/gist/ShiySabiniano/a34e01bcfc227cddc55a6634f1823539/bitnet_tinybert_orchestrator_colab.ipynb)
