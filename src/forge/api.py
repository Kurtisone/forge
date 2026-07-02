"""
Forge HTTP API — exposes the runtime as a REST service.

Endpoints:
  GET  /            → UI (HTML)
  GET  /health      → provider / model info
  POST /chat        → single conversation turn
  POST /review      → file content review
  GET  /traces      → recent execution traces

Run:
  uvicorn forge.api:app --host 0.0.0.0 --port 8000

The LLM calls are blocking (HTTP to llama.cpp / Ollama). They run
in a thread-pool executor so FastAPI's event loop is never blocked.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from forge import trace
from forge.config import FORGE_PROVIDER, LLM_MODEL
from forge.orchestrator import Orchestrator

app = FastAPI(title="Forge", version="3.1.0", docs_url="/docs")
_executor = ThreadPoolExecutor(max_workers=2)
_orchestrator = Orchestrator()


# ─── Models ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: Optional[list[dict]] = None   # reserved for future multi-session use


class ReviewRequest(BaseModel):
    content: str                            # file content (not a path)
    filename: str = "untitled"
    question: str = "Que peut-on améliorer ?"


class ChatResponse(BaseModel):
    output: str
    tool: str
    ok: bool
    steps: int
    error: Optional[str] = None


class ReviewResponse(BaseModel):
    output: str
    ok: bool
    error: Optional[str] = None


# ─── Helpers ───────────────────────────────────────────────────────

async def _run_in_thread(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


class RunRequest(BaseModel):
    graph: str                          # registered graph name: "review"
    input: str                          # user_input passed to Graph.run()
    context: Optional[dict] = None      # initial_context for the graph


class RunResponse(BaseModel):
    output: str
    ok: bool
    steps: int
    graph: str
    error: Optional[str] = None


# ─── Graph registry ────────────────────────────────────────────────

def _graph_registry() -> dict:
    """Return all available graph builders, keyed by name."""
    from forge.graphs.default import build as default_build
    from forge.graphs.review import build as review_build
    return {
        "default": default_build,
        "review":  review_build,
    }


# ─── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "provider": FORGE_PROVIDER,
        "model": LLM_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    result = await _run_in_thread(_orchestrator.run, req.message)
    return ChatResponse(
        output=result.output,
        tool=result.tool,
        ok=result.ok,
        steps=result.steps,
        error=result.error,
    )


@app.post("/review", response_model=ReviewResponse)
async def review(req: ReviewRequest):
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content cannot be empty")

    import os
    import tempfile

    from forge.graphs.review import run as review_run

    # Write the content to a temp file so the review graph can read it
    suffix = Path(req.filename).suffix or ".txt"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as f:
        f.write(req.content)
        tmp_path = f.name

    try:
        output = await _run_in_thread(review_run, tmp_path, req.question)
    finally:
        os.unlink(tmp_path)

    return ReviewResponse(output=output, ok=bool(output))


@app.get("/traces")
async def get_traces(n: int = 10):
    return {"traces": trace.read_last(n)}


@app.get("/tools")
async def list_tools():
    """Return the list of currently enabled tools and available graphs."""
    from forge.tools.registry import available_tools
    return {
        "tools": available_tools(),
        "graphs": list(_graph_registry().keys()),
    }


@app.post("/run", response_model=RunResponse)
async def run_graph(req: RunRequest):
    """Run any registered graph by name with an optional initial context."""
    registry = _graph_registry()
    if req.graph not in registry:
        raise HTTPException(
            status_code=404,
            detail=f"graph {req.graph!r} not found. Available: {sorted(registry)}"
        )
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input cannot be empty")

    def _execute():
        g = registry[req.graph]()
        return g.run(req.input, initial_context=req.context or {})

    state = await _run_in_thread(_execute)
    return RunResponse(
        output=state.final_output or "",
        ok=state.ok,
        steps=state.steps_taken,
        graph=req.graph,
        error=state.error,
    )


# ─── UI ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    static = Path(__file__).parent / "static" / "index.html"
    if static.exists():
        return HTMLResponse(static.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Forge UI not found</h1><p>Run from the installed package.</p>")
