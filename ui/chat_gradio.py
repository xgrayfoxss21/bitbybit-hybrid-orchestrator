#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ui/chat_gradio.py

Multi-turn chat UI for the BitByBit Hybrid Orchestrator.
- Keeps a rolling transcript (User/Assistant) and feeds it into the same hybrid DAG:
    parse → [claim1, claim2] (parallel) → reduce
- Runs the TinyBERT-style guard on input and output each turn.
- Reuses the tiny orchestrator pieces from `orchestrator/cli.py`.

Usage:
  pip install -r orchestrator/requirements.txt
  pip install -r ui/requirements.txt
  python ui/chat_gradio.py

Compliance (AGPL §13):
- Footer shows the running commit with a link to source.
- Attempts to add an X-AGPL-Source header and expose GET /source via the
  FastAPI app Gradio runs under (best-effort; no-op if unavailable).
"""

import os
import json
import asyncio
from typing import List, Tuple, Dict, Any, Optional

import gradio as gr

# Allow run_until_complete inside notebooks/Gradio
try:
    import nest_asyncio  # type: ignore
    nest_asyncio.apply()
except Exception:
    pass

# --- Import orchestrator primitives (Registry, Scheduler, Node, guard, agents) ---
try:
    # These should exist in your repo per README layout
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
    # Helpful error if the orchestrator file isn't present
    raise SystemExit(
        "ERROR: Could not import orchestrator primitives from 'orchestrator/cli.py'.\n"
        "Make sure your repo includes that file (see README layout) and your PYTHONPATH "
        "includes the repo root. Original error:\n" + repr(e)
    )


# ---------------------------
# Helpers
# ---------------------------
def _format_transcript(history: List[Tuple[str, str]], user_msg: str) -> str:
    """
    Convert chat history + new user message to a transcript string consumed by the pipeline.
    History is a list of [user, assistant] pairs from gr.Chatbot.
    """
    lines: List[str] = []
    for u, a in history:
        if u:
            lines.append(f"User: {u}")
        if a:
            lines.append(f"Assistant: {a}")
    lines.append(f"User: {user_msg}")
    return "\n".join(lines).strip()


def _build_nodes(claim1: str, claim2: str, max_sentences: int) -> List[Node]:
    """
    Construct the DAG used by the demo:
      parse → [claim1, claim2] → reduce
    """
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


def _run(transcript: str, claim1: str, claim2: str, max_sentences: int) -> Dict[str, Dict[str, Any]]:
    """
    Register demo agents, build nodes, and execute the DAG.
    Returns the full results map {node_id: result_dict}.
    """
    reg = Registry()
    reg.register("bitnet.summarizer", summarizer)
    reg.register("bitnet.claimcheck", claimcheck)
    reg.register("bitnet.synthesis", synthesis)

    nodes = _build_nodes(claim1, claim2, max_sentences)
    sched = Scheduler(registry=reg, guard=guard, max_concurrency=2)

    async def _go():
        return await sched.run_dag(nodes, {"text": transcript})

    # Use the current loop (nest_asyncio makes this safe in Gradio)
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_go())


def chat_predict(
    history: List[Tuple[str, str]],
    user_msg: str,
    max_sentences: int,
    claim1: str,
    claim2: str,
    show_mod: bool,
):
    """
    Gradio event handler for sending a chat turn.
    """
    # Build transcript from history + this user message
    transcript = _format_transcript(history, user_msg)

    # Execute DAG
    results = _run(transcript, claim1, claim2, max_sentences)

    # Compose assistant reply from the reducer
    reply = (results.get("reduce", {}) or {}).get("text", "").strip() or "(no response)"

    # Optional: moderation JSON + guard mode (return as dict for gr.JSON)
    debug_payload: Optional[Dict[str, Any]] = None
    if show_mod:
        modcards = {}
        for nid, r in results.items():
            if isinstance(r, dict) and "_moderation" in r:
                modcards[nid] = [
                    {
                        "node": m.get("node"),
                        "actions": m.get("actions"),
                        "labels": {k: round(float(v), 2) for k, v in (m.get("labels", {}) or {}).items()},
                        "why": m.get("why"),
                    }
                    for m in (r.get("_moderation") or [])
                ]
        debug_payload = {"guard_mode": getattr(guard, "mode", "n/a"), "modcards": modcards}

    # Update chat history (clear input box)
    history = history + [[user_msg, reply]]
    guard_mode = getattr(guard, "mode", "n/a")
    return history, "", (debug_payload or {}), guard_mode


def main():
    # Rebranded repo + commit for AGPL §13 surfaces
    repo = "https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator"
    commit = os.getenv("APP_COMMIT_SHA", "HEAD")

    # Optional runtime toggles (no secrets here)
    share = str(os.getenv("GRADIO_SHARE", "0")).lower() in {"1", "true", "yes", "on"}
    server_name = os.getenv("SERVER_NAME")  # e.g., "0.0.0.0"
    server_port = int(os.getenv("SERVER_PORT", "7861"))
    # Basic auth: GRADIO_AUTH="user:pass"  (for quick demos; use a proper proxy for prod)
    auth_env = os.getenv("GRADIO_AUTH", "")
    auth = None
    if ":" in auth_env and len(auth_env.split(":", 1)[0]) > 0:
        user, pw = auth_env.split(":", 1)
        auth = (user, pw)

    with gr.Blocks(title="BitByBit Hybrid Orchestrator — Chat") as app:
        gr.Markdown("## BitByBit Hybrid Orchestrator — Chat")
        gr.Markdown(
            "Multi-turn chat that runs the hybrid pipeline each turn.\n\n"
            "**Flow:** `parse → [claim1, claim2] (parallel) → reduce` with guard on input & output."
        )

        with gr.Row():
            maxs = gr.Slider(1, 8, value=3, step=1, label="Summary sentences")
            c1 = gr.Textbox(label="Claim 1", value="BitNet uses 1.58-bit weights")
            c2 = gr.Textbox(label="Claim 2", value="TinyBERT is effective for classification")
            show_mod = gr.Checkbox(label="Show moderation JSON", value=False)

        chat = gr.Chatbot(height=460, label="Chat with the orchestrator")
        user_in = gr.Textbox(placeholder="Type your message…", label="Message")

        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")

        dbg = gr.JSON(label="Debug (moderation + guard mode)")
        mode_out = gr.Textbox(label="Guard mode", interactive=False)

        # Wire interactions
        send.click(
            chat_predict,
            inputs=[chat, user_in, maxs, c1, c2, show_mod],
            outputs=[chat, user_in, dbg, mode_out],
        )
        user_in.submit(
            chat_predict,
            inputs=[chat, user_in, maxs, c1, c2, show_mod],
            outputs=[chat, user_in, dbg, mode_out],
        )

        def _clear():
            return [], "", {}, getattr(guard, "mode", "n/a")

        clear.click(_clear, outputs=[chat, user_in, dbg, mode_out])

        # AGPL §13 hint in footer
        try:
            app.footer = (
                f"Source: [{commit[:7]}]({repo}/tree/{commit}) • License: AGPL-3.0-or-later "
                f"• Project Website: https://bit.foxesden.xyz/"
            )
        except Exception:
            pass

        # Best-effort: add X-AGPL-Source header + /source endpoint on the FastAPI app Gradio uses
        try:
            # Different gradio versions expose the FastAPI app differently
            fastapi_app = getattr(app, "server_app", None) or getattr(app, "app", None)
            if fastapi_app is not None:
                from fastapi.responses import JSONResponse
                from starlette.middleware.base import BaseHTTPMiddleware

                class AgplHeader(BaseHTTPMiddleware):
                    async def dispatch(self, request, call_next):
                        resp = await call_next(request)
                        resp.headers["X-AGPL-Source"] = f"{repo}/tree/{commit}"
                        return resp

                # add middleware
                fastapi_app.add_middleware(AgplHeader)  # type: ignore[attr-defined]

                # /source endpoint
                @fastapi_app.get("/source")  # type: ignore[attr-defined]
                def source():
                    return JSONResponse(
                        {
                            "license": "AGPL-3.0-or-later",
                            "repo": repo,
                            "commit": commit,
                            "url": f"{repo}/tree/{commit}",
                        }
                    )
        except Exception:
            # Fail closed (no crash) if Gradio internals change
            pass

    # modest queue; tune as needed
    app = app.queue(concurrency_count=1, max_size=16)

    # Launch with optional knobs; avoid share=True in production
    launch_kwargs: Dict[str, Any] = {}
    if server_name:
        launch_kwargs["server_name"] = server_name
    if server_port:
        launch_kwargs["server_port"] = server_port
    if auth:
        launch_kwargs["auth"] = auth
    if share:
        launch_kwargs["share"] = True

    app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
