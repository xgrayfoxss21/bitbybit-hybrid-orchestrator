#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ui/gradio_demo.py

Single-turn Web UI for the BitByBit Hybrid Orchestrator.

- Runs the mixed DAG once per submission:
    parse → [claim1, claim2] (parallel) → reduce
- Applies the TinyBERT-style guard on input and output.
- Reuses orchestrator primitives and demo agents from `orchestrator/cli.py`.

Usage:
  pip install -r orchestrator/requirements.txt
  pip install -r ui/requirements.txt
  python ui/gradio_demo.py

Compliance (AGPL §13):
- Footer shows the running commit with a link to source.
- Attempts to add an X-AGPL-Source header and expose GET /source via the
  FastAPI app Gradio runs under (best-effort; no-op if unavailable).
"""

import os
import json
import asyncio
from typing import Dict, Any, List

import gradio as gr

# Allow run_until_complete inside notebooks/Gradio
try:
    import nest_asyncio  # type: ignore
    nest_asyncio.apply()
except Exception:
    pass

# --- Import orchestrator primitives (Registry, Scheduler, Node, guard, agents) ---
try:
    from orchestrator.cli import (
        Registry,
        Scheduler,
        Node,
        guard,          # TinyBERT-style guard instance
        summarizer,     # demo agent
        claimcheck,     # demo agent
        synthesis,      # demo agent
    )
except ImportError as e:
    raise SystemExit(
        "ERROR: Could not import orchestrator primitives from 'orchestrator/cli.py'.\n"
        "Ensure your repo contains that file (see README layout) and that PYTHONPATH "
        "includes the repo root. Original error:\n" + repr(e)
    )


def _build_nodes(max_sentences: int, claim1: str, claim2: str) -> List[Node]:
    """Construct the DAG used by the demo."""
    return [
        Node(
            id="parse",
            agent="bitnet.summarizer",
            deps=[],
            guard_pre=True,
            guard_post=True,
            timeout_ms=900,
            max_retries=0,
            params={"max_sentences": int(max_sentences)},
        ),
        Node(
            id="claim1",
            agent="bitnet.claimcheck",
            deps=["parse"],
            guard_pre=False,
            guard_post=True,
            timeout_ms=600,
            max_retries=1,
            params={"claim": claim1, "kb": []},
        ),
        Node(
            id="claim2",
            agent="bitnet.claimcheck",
            deps=["parse"],
            guard_pre=False,
            guard_post=True,
            timeout_ms=600,
            max_retries=1,
            params={"claim": claim2, "kb": []},
        ),
        Node(
            id="reduce",
            agent="bitnet.synthesis",
            deps=["claim1", "claim2"],
            guard_pre=False,
            guard_post=True,
            timeout_ms=800,
            max_retries=0,
            params={},
        ),
    ]


def _run_once(user_text: str, max_sentences: int, claim1: str, claim2: str) -> Dict[str, Any]:
    """Register demo agents, build nodes, and execute the DAG once."""
    reg = Registry()
    reg.register("bitnet.summarizer", summarizer)
    reg.register("bitnet.claimcheck", claimcheck)
    reg.register("bitnet.synthesis", synthesis)

    nodes = _build_nodes(max_sentences, claim1, claim2)
    sched = Scheduler(registry=reg, guard=guard, max_concurrency=2)

    async def _go():
        return await sched.run_dag(nodes, {"text": user_text})

    loop = asyncio.get_even_
