---
nav_order: 4
title: Safety & Guardrails
---

# Safety & Guardrails

The orchestrator ships with a **TinyBERT-style Guard** that protects **inputs** and **outputs** every time the pipeline runs. It can also wrap **individual nodes** for extra protection (e.g., before tool/exec/network steps).

- **Pre-guard (input):** scrubs/redacts the inbound text and can **block** unsafe prompts.
- **Per-node gate (optional):** run the guard before/after specific nodes.
- **Post-guard (output):** moderates responses, **redacts PII**, and can **block** risky content.
- **Moderation card:** every guard call can attach a compact JSON record of its decision.

See also: **[SECURITY.md](../SECURITY.md)** and **[COMPLIANCE.md](../COMPLIANCE.md)**.

---

## Guard modes

The guard supports two execution modes:

1. **`regex-only` (default):**  
   *PII redaction* (email/phone), *basic jailbreak heuristics,* and light toxicity checks using rule-based signals.

2. **`onnx+regex` (optional):**  
   Adds TinyBERT **ONNX** sequence-classifier signals on top of the regex/heuristics. Useful when you can ship a small, quantized model.

> If ONNX artifacts aren’t configured or fail to load, the guard **automatically** runs in `regex-only`.

---

## Configuration (YAML)

Tune thresholds and redactions in your pipeline file:

```yaml
# orchestrator/pipeline.yml (excerpt)
policies:
  thresholds:
    toxicity_block: 0.50      # block if predicted toxicity > 0.50
    pii_redact: 0.70          # redact spans when PII score/prob ≥ 0.70
    jailbreak_block: 0.60     # block if jailbreak score > 0.60

  # Optional: explicit redaction toggles (defaults shown)
  redactions:
    email: true
    phone: true

  # Optional: output policy
  output:
    append_moderation_card: true
    safe_templates:
      fallback_enabled: true
      blocked_message: "Response withheld due to safety policy."

# For chat pipelines:
conversation:
  redact_pii_in_history: true   # store redacted transcript in history
