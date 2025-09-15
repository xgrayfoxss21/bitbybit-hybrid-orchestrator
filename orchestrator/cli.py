#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Minimal CLI runner for BitNet Hybrid Orchestrator demo (enhanced).
- Loads inline text, file, or stdin
- Runs a small mixed DAG with a simple PII guard
- Prints results in human-readable or JSON format
- Supports concurrency, retries, and basic timing

Examples
--------
python -m orchestrator.cli --input "Contact me at test@example.com and (555) 123-4567"
python -m orchestrator.cli --input @examples/sample.txt --format json --save out.json
python -m orchestrator.cli --stdin --claim "BitNet uses 1.58-bit weights" --claim "TinyBERT is effective for classification"

Notes
-----
This CLI is intended for local demos and developer testing. For networked
deployments, also read COMPLIANCE.md (AGPL §13) for source exposure surfaces.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Optional

# ---- Tiny guard (regex PII; mirrors notebook fallback) ----
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{3}\)?[\s\-\.]?)\d{3}[\s\-\.]?\d{4}\b")

class Guard:\n    def __init__(self, redact: bool = True):
        self.redact = redact

    def check(self, text: str, mode: str = "input") -> Dict[str, Any]:
        if not isinstance(text, str):
            return {"allowed": True, "text": text, "labels": {"pii": 0.0}, "actions": []}
        redactions: List[Dict[str, str]] = []
        out = text
        if self.redact:
            new = EMAIL.sub("[REDACTED_EMAIL]", out)
            if new != out:
                redactions.append({"type": "PII.email"})
            out = new
            new = PHONE.sub("[REDACTED_PHONE]", out)
            if new != out:
                redactions.append({"type": "PII.phone"})
            out = new
        return {
            "allowed": True,
            "text": out,
            "labels": {"pii": 1.0 if redactions else 0.0},
            "actions": ["redact"] if redactions else [],
            "redactions": redactions,
            "mode": mode,
        }

guard = Guard()

# ---- Orchestrator core (trimmed but more robust) ----
@dataclass
class Node:
    id: str
    agent: str
    deps: List[str] = field(default_factory=list)
    guard_pre: bool = True
    guard_post: bool = True
    timeout_ms: int = 2000
    max_retries: int = 0
    params: Dict[str, Any] = field(default_factory=dict)

class Registry:
    def __init__(self):
        self._a: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._a[name] = fn

    async def run(self, name: str, **kw) -> Dict[str, Any]:
        fn = self._a[name]
        return await fn(**kw)

class Scheduler:
    def __init__(self, registry: Registry, guard: Guard, max_concurrency: int = 2):
        self.r = registry
        self.g = guard
        self.sema = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def _run_once(self, n: Node, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Guard (pre)
        if n.guard_pre:
            g = self.g.check(payload.get("text", ""), "input")
            if not g.get("allowed", True):
                raise RuntimeError("blocked_pre")
            payload = dict(payload, text=g.get("text", payload.get("text")))

        # Execute with timeout
        start = time.perf_counter()
        res: Dict[str, Any]
        try:
            res = await asyncio.wait_for(self.r.run(n.agent, **n.params, **payload), timeout=n.timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            raise RuntimeError(f"timeout({n.timeout_ms}ms)")

        # Guard (post)
        if n.guard_post:
            g2 = self.g.check(res.get("text", ""), "output")
            if not g2.get("allowed", True):
                raise RuntimeError("blocked_post")
            res["text"] = g2.get("text", res.get("text", ""))

        res["_node"] = n.id
        res["_duration_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
        return res

    async def _run(self, n: Node, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with self.sema:
            last_exc: Optional[Exception] = None
            for attempt in range(n.max_retries + 1):
                try:
                    return await self._run_once(n, payload)
                except Exception as e:  # noqa: BLE001 - we want to capture and report
                    last_exc = e
                    await asyncio.sleep(min(0.1 * (attempt + 1), 1.0))
            # Retries exhausted
            return {"_node": n.id, "error": str(last_exc or "unknown_error"), "text": ""}

    async def run_dag(self, nodes: List[Node], sources: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        id2: Dict[str, Node] = {n.id: n for n in nodes}
        deps: Dict[str, set] = {n.id: set(n.deps) for n in nodes}
        ready: List[str] = [n.id for n in nodes if not n.deps]

        inbuf: Dict[str, Dict[str, Any]] = {rid: dict(sources) for rid in ready}
        running: Dict[str, asyncio.Task] = {}

        async def launch(i: str) -> None:
            running[i] = asyncio.create_task(self._run(id2[i], inbuf[i]))

        for i in ready:
            await launch(i)

        results: Dict[str, Dict[str, Any]] = {}
        while running:
            done, _ = await asyncio.wait(running.values(), return_when=asyncio.FIRST_COMPLETED)
            for i, t in list(running.items()):
                if t in done:
                    try:
                        results[i] = await t
                    except Exception as e:  # defensive
                        results[i] = {"_node": i, "error": str(e), "text": ""}
                    running.pop(i, None)
                    # Propagate outputs to children
                    for c, ds in deps.items():
                        if i in ds:
                            ds.remove(i)
                            inbuf.setdefault(c, {}).update(results[i])
                    # Launch newly unblocked children
                    for c, ds in list(deps.items()):
                        if not ds and c not in results and c not in running:
                            await launch(c)
        return results

# ---- Demo agents (same as notebook, trimmed) ----

def _sents(x: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", x) if s.strip()]

async def summarizer(text: str, max_sentences: int = 3, **_):
    s = _sents(text)
    keep = s[: max(1, max_sentences)]
    return {"text": " ".join(keep)}

def _tok(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())

async def claimcheck(text: str, claim: str, **_):
    verdict = "supported" if all(t in _tok(text) for t in _tok(claim)) else "uncertain"
    return {"text": f"Claim: {claim} → {verdict}"}

async def synthesis(text: str, pieces: List[str] | None = None, **_):
    pieces = pieces or []
    bullets = [p.splitlines()[0] for p in pieces if isinstance(p, str) and p.strip()]
    return {"text": "Executive Brief:\n- " + "\n- ".join(bullets)}

# ---- CLI helpers ----

def _read_input_value(arg_input: Optional[str], use_stdin: bool) -> str:
    if use_stdin:
        data = sys.stdin.read()
        if not data:
            raise SystemExit("--stdin specified but no data on STDIN")
        return data
    if arg_input and arg_input.startswith("@"):
        path = arg_input[1:]
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if arg_input:
        return arg_input
    # Default sample
    return (
        "Contact me at test@example.com. BitNet b1.58 enables efficient ~1.58-bit weights. "
        "TinyBERT helps safety. Call (555) 123-4567."
    )


def build_nodes(default_timeout_ms: int, default_retries: int, claims: List[str]) -> List[Node]:
    if not claims:
        claims = [
            "BitNet uses 1.58-bit weights",
            "TinyBERT is effective for classification",
        ]
    nodes: List[Node] = [
        Node("parse", "bitnet.summarizer", [], True, True, default_timeout_ms, default_retries, {"max_sentences": 3}),
    ]

    # Claim nodes depend on parse
    for idx, c in enumerate(claims, start=1):
        nodes.append(
            Node(
                f"claim{idx}",
                "bitnet.claimcheck",
                ["parse"],
                False,
                True,
                default_timeout_ms,
                max(1, default_retries),
                {"claim": c},
            )
        )

    # Reduce depends on all claim nodes
    claim_ids = [n.id for n in nodes if n.id.startswith("claim")]
    nodes.append(Node("reduce", "bitnet.synthesis", claim_ids, False, True, default_timeout_ms, default_retries, {}))
    return nodes


def print_text_results(res: Dict[str, Dict[str, Any]], claim_count: int) -> None:
    order = ["parse"] + [f"claim{i}" for i in range(1, claim_count + 1)] + ["reduce"]
    for nid in order:
        r = res.get(nid, {})
        header = f"=== {nid} ==="
        print(f"\n{header}\n{r.get('text', '')}")
        if "_duration_ms" in r:
            print(f"(took {r['_duration_ms']} ms)")
        if r.get("error"):
            print(f"[error] {r['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BitNet Hybrid Orchestrator - Minimal CLI (enhanced)")
    parser.add_argument("--input", help="Inline text or @path/to/file.txt", required=False)
    parser.add_argument("--stdin", help="Read input from STDIN", action="store_true")
    parser.add_argument("--claim", dest="claims", action="append", default=[], help="Add a claim check (repeatable)")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--max-concurrency", type=int, default=2, help="Max concurrent node execution")
    parser.add_argument("--timeout-ms", type=int, default=2000, help="Default per-node timeout in ms")
    parser.add_argument("--retries", type=int, default=0, help="Default per-node max retries")
    parser.add_argument("--no-guard-pre", action="store_true", help="Disable pre-guard globally")
    parser.add_argument("--no-guard-post", action="store_true", help="Disable post-guard globally")
    parser.add_argument("--save", help="Write full results JSON to a file")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--source-url", action="store_true", help="Print AGPL source URL and exit")

    args = parser.parse_args()

    if args.version:
        print("BitNet Hybrid Orchestrator CLI alpha v0.002")
        return

    if args.source_url:
        repo = os.getenv("APP_SOURCE_REPO", "https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator")
        commit = os.getenv("APP_COMMIT_SHA", "HEAD")
        print(f"{repo}/tree/{commit}")
        return

    # Source text
    try:
        text = _read_input_value(args.input, args.stdin)
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)

    # Registry
    reg = Registry()
    for name, fn in [
        ("bitnet.summarizer", summarizer),
        ("bitnet.claimcheck", claimcheck),
        ("bitnet.synthesis", synthesis),
    ]:
        reg.register(name, fn)

    # Build DAG
    nodes = build_nodes(args.timeout_ms, args.retries, args.claims)

    # Optionally toggle guards
    if args.no_guard_pre or args.no_guard_post:
        for n in nodes:
            if args.no_guard_pre:
                n.guard_pre = False
            if args.no_guard_post:
                n.guard_post = False

    async def run():
        sched = Scheduler(reg, guard, max_concurrency=args.max_concurrency)
        res = await sched.run_dag(nodes, {"text": text})

        # Collect pieces for synthesis (first line of each claim)
        claim_ids = [nid for nid in res if nid.startswith("claim")]
        pieces = [res[c]["text"] for c in sorted(claim_ids)]
        # If reducer didn't receive pieces via propagation (it should), ensure they exist
        if "reduce" in res and "text" in res["reduce"] and pieces:
            pass  # output already contains synthesized text
        elif "reduce" in res:
            # Run a quick local synthesis if needed
            res["reduce"] = await synthesis(text=text, pieces=pieces)
            res["reduce"]["_node"] = "reduce"

        if args.format == "json":
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print_text_results(res, claim_count=len([n for n in nodes if n.id.startswith("claim")]))

        if args.save:
            try:
                with open(args.save, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
                print(f"\n[written] {args.save}")
            except Exception as e:
                print(f"[warn] failed to write {args.save}: {e}", file=sys.stderr)

        # Exit code: 0 if no node has error, 2 if partial errors, 1 if reduce missing
        errors = [k for k, v in res.items() if v.get("error")]
        if "reduce" not in res:
            raise SystemExit(1)
        raise SystemExit(0 if not errors else 2)

    try:
        asyncio.run(run())
    except RuntimeError:
        # Handle nested loop environments (e.g., notebooks)
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())


if __name__ == "__main__":
    main()
