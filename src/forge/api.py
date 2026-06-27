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
from fastapi.responses import HTMLResponse, JSONResponse
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

    import tempfile, os
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


# ─── UI ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    static = Path(__file__).parent / "static" / "index.html"
    if static.exists():
        return HTMLResponse(static.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Forge UI not found</h1><p>Run from the installed package.</p>")
