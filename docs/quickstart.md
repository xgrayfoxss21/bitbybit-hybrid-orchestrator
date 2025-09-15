---
title: Quickstart
nav_order: 2
---

# Quickstart

This guide shows you how to run the **BitNet Hybrid Orchestrator** in **Colab** (zero setup) or **locally** with Python. It also covers the optional **Web UIs** (single-turn and multi-turn chat) and how to use the optional **TinyBERT ONNX** guard.

---

## TL;DR

- **Colab (recommended first run):**  
  👉 https://colab.research.google.com/gist/ShiySabiniano/a34e01bcfc227cddc55a6634f1823539/bitnet_tinybert_orchestrator_colab.ipynb  
  Run Cells **1 → 5**.  
  - **Cell 6:** single-turn web form.  
  - **Cell 6B:** **chat mode** (multi-turn, history-preserving).

- **Local (Python 3.10+):**
  ```bash
  git clone https://github.com/ShiySabiniano/bitnet-hybrid-orchestrator.git
  cd bitnet-hybrid-orchestrator
  python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
  pip install -r orchestrator/requirements.txt
