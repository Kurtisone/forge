"""
Forge HTTP API — exposes the runtime as a REST service.

Endpoints:
  GET  /            → UI (HTML)
  GET  /health      → provider / model info
  POST /chat        → single conversation turn
  POST /review      → file content review
  GET  /traces      → recent execution traces
  GET  /history     → full rolling history (single-thread UI, v3.9)
  GET  /drawer      → pinned messages ("tiroir", v3.9)
  POST /drawer/pin  → pin a message by id
  POST /drawer/unpin → unpin a message by id
  POST /compact     → force a compaction pass now (v3.9)

Auth: set API_TOKEN in the environment to require
`Authorization: Bearer <token>` on every endpoint except / and
/health. Unset (default) means the API stays open, unchanged from
before this was added.

Rate limiting: in-memory sliding window, per client IP, on every
endpoint except / and /health. RATE_LIMIT_REQUESTS per
RATE_LIMIT_WINDOW_SECONDS (default: 30 per 60s). Set
RATE_LIMIT_ENABLED=false to disable.

Run:
  uvicorn forge.api:app --host 0.0.0.0 --port 8000

The LLM calls are blocking (HTTP to llama.cpp / Ollama). They run
in a thread-pool executor so FastAPI's event loop is never blocked.
"""

import asyncio
import hmac
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from forge import rag, ratelimit, trace
from forge.config import API_TOKEN, FORGE_PROVIDER, LLAMA_CPP_URL, LLM_MODEL
from forge.orchestrator import Orchestrator

app = FastAPI(title="Forge", version="3.3.0", docs_url="/docs")
_executor = ThreadPoolExecutor(max_workers=2)
_orchestrator = Orchestrator()


# ─── Auth ──────────────────────────────────────────────────────────
# Optional: only enforced when API_TOKEN is set in the environment.
# Unset (the default) means the API is open, exactly as before this
# was added -- nothing changes for anyone not opting in.


async def require_token(authorization: str | None = Header(None)) -> None:
    if not API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    # constant-time comparison: this guards a real secret, not just a UX check
    if not hmac.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=401, detail="invalid bearer token")


# ─── Rate limiting ─────────────────────────────────────────────────
# In-memory, per-client-IP sliding window (forge/ratelimit.py). Set
# RATE_LIMIT_ENABLED=false to disable entirely -- e.g. for local dev,
# or if you're fronting this with a proxy that already rate-limits.


async def rate_limit(request: Request) -> None:
    client_key = request.client.host if request.client else "unknown"
    allowed, retry_after = ratelimit.check(client_key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded, try again shortly",
            headers={"Retry-After": str(retry_after)},
        )


# ─── Models ────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None  # reserved for future multi-session use


class ReviewRequest(BaseModel):
    content: str  # file content (not a path)
    filename: str = "untitled"
    question: str = "Que peut-on améliorer ?"
    test_path: str | None = None  # optional, run these tests before reviewing


class ChatResponse(BaseModel):
    output: str
    tool: str
    ok: bool
    steps: int
    error: str | None = None


class ReviewResponse(BaseModel):
    output: str
    ok: bool
    error: str | None = None


# ─── Helpers ───────────────────────────────────────────────────────


async def _run_in_thread(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


class RunRequest(BaseModel):
    graph: str  # registered graph name: "review"
    input: str  # user_input passed to Graph.run()
    context: dict | None = None  # initial_context for the graph


class RememberRequest(BaseModel):
    kind: Literal["decision", "todo", "fact"]
    content: str
    project: str | None = None


class RememberResponse(BaseModel):
    id: int


class PinRequest(BaseModel):
    message_id: int


class PinnedMessage(BaseModel):
    id: int
    role: str
    content: str
    pinned: bool


class CompactResponse(BaseModel):
    removed: int


class SearchResult(BaseModel):
    id: int
    kind: str
    content: str
    project: str | None
    status: str
    created_at: str
    distance: float


class RunResponse(BaseModel):
    output: str
    ok: bool
    steps: int
    graph: str
    error: str | None = None


# ─── Graph registry ────────────────────────────────────────────────


def _graph_registry() -> dict:
    """Return all available graph builders, keyed by name."""
    from forge.graphs.default import build as default_build
    from forge.graphs.review import build as review_build

    return {
        "default": default_build,
        "review": review_build,
    }


# ─── Endpoints ─────────────────────────────────────────────────────


@app.get("/health")
async def health():
    model = LLM_MODEL
    if FORGE_PROVIDER == "llama_cpp":
        from forge.providers.llama_cpp import get_loaded_model

        loaded = await _run_in_thread(get_loaded_model, LLAMA_CPP_URL)
        if loaded:
            model = loaded

    return {
        "status": "ok",
        "provider": FORGE_PROVIDER,
        "model": model,
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_token), Depends(rate_limit)],
)
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


@app.post(
    "/review",
    response_model=ReviewResponse,
    dependencies=[Depends(require_token), Depends(rate_limit)],
)
async def review(req: ReviewRequest):
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content cannot be empty")

    import os
    import tempfile

    from forge.graphs.review import run as review_run

    # Write the content to a temp file so the review graph can read it.
    # Note: test_path (if given) is resolved relative to WORKSPACE_DIR
    # by the test tool, NOT relative to this temp file -- running
    # tests against submitted content only makes sense when that
    # content already corresponds to a file inside the workspace
    # (e.g. reviewing a workspace file's current content with its
    # existing test suite), not for arbitrary pasted snippets.
    suffix = Path(req.filename).suffix or ".txt"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as f:
        f.write(req.content)
        tmp_path = f.name

    try:
        output = await _run_in_thread(review_run, tmp_path, req.question, req.test_path)
    finally:
        os.unlink(tmp_path)

    return ReviewResponse(output=output, ok=bool(output))


@app.get("/traces", dependencies=[Depends(require_token), Depends(rate_limit)])
async def get_traces(n: int = 10):
    return {"traces": trace.read_last(n)}


@app.get("/tools", dependencies=[Depends(require_token), Depends(rate_limit)])
async def list_tools():
    """Return the list of currently enabled tools and available graphs."""
    from forge.tools.registry import available_tools

    return {
        "tools": available_tools(),
        "graphs": list(_graph_registry().keys()),
    }


@app.post(
    "/run",
    response_model=RunResponse,
    dependencies=[Depends(require_token), Depends(rate_limit)],
)
async def run_graph(req: RunRequest):
    """Run any registered graph by name with an optional initial context."""
    registry = _graph_registry()
    if req.graph not in registry:
        raise HTTPException(
            status_code=404,
            detail=f"graph {req.graph!r} not found. Available: {sorted(registry)}",
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


# ─── Vector memory / RAG (v3.7) ────────────────────────────────────


@app.post(
    "/remember",
    response_model=RememberResponse,
    dependencies=[Depends(require_token), Depends(rate_limit)],
)
async def remember(req: RememberRequest):
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content cannot be empty")

    def _execute():
        conn = rag.get_connection()
        try:
            return rag.remember(
                conn, kind=req.kind, content=req.content, project=req.project
            )
        finally:
            conn.close()

    try:
        entry_id = await _run_in_thread(_execute)
    except rag.EmbeddingError as e:
        raise HTTPException(
            status_code=502, detail=f"embedding server unreachable: {e}"
        ) from e
    return RememberResponse(id=entry_id)


@app.get(
    "/search",
    response_model=list[SearchResult],
    dependencies=[Depends(require_token), Depends(rate_limit)],
)
async def search(
    q: str = Query(..., description="query text"),
    top_k: int = Query(5, ge=1, le=50),
    kind: Literal["decision", "todo", "fact"] | None = Query(None),
    project: str | None = Query(None),
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="q cannot be empty")

    def _execute():
        conn = rag.get_connection()
        try:
            return rag.search(conn, query=q, top_k=top_k, kind=kind, project=project)
        finally:
            conn.close()

    try:
        results = await _run_in_thread(_execute)
    except rag.EmbeddingError as e:
        raise HTTPException(
            status_code=502, detail=f"embedding server unreachable: {e}"
        ) from e
    return results


# ─── Drawer / compaction (v3.9) ────────────────────────────────────


@app.get(
    "/history",
    response_model=list[PinnedMessage],
    dependencies=[Depends(require_token), Depends(rate_limit)],
)
async def get_history():
    """
    Full rolling history with stable ids -- what the single-thread web
    UI renders on load/after each turn, and what pin/unpin target.
    """
    from forge import memory

    return memory.get_history()


@app.get(
    "/drawer",
    response_model=list[PinnedMessage],
    dependencies=[Depends(require_token), Depends(rate_limit)],
)
async def get_drawer():
    """List currently pinned messages -- the 'tiroir'."""
    from forge import memory

    return memory.get_pinned()


@app.post(
    "/drawer/pin",
    dependencies=[Depends(require_token), Depends(rate_limit)],
)
async def pin(req: PinRequest):
    from forge import memory

    if not memory.pin_message(req.message_id):
        raise HTTPException(status_code=404, detail="message not found")
    return {"ok": True}


@app.post(
    "/drawer/unpin",
    dependencies=[Depends(require_token), Depends(rate_limit)],
)
async def unpin(req: PinRequest):
    from forge import memory

    if not memory.unpin_message(req.message_id):
        raise HTTPException(status_code=404, detail="message not found")
    return {"ok": True}


@app.post(
    "/compact",
    response_model=CompactResponse,
    dependencies=[Depends(require_token), Depends(rate_limit)],
)
async def compact():
    """Force a compaction pass now, regardless of COMPACTION_THRESHOLD."""
    from forge import memory

    removed = await _run_in_thread(memory.compact_now)
    return CompactResponse(removed=removed)


# ─── UI ────────────────────────────────────────────────────────────


# Third layer under the two fixes in static/index.html (quote escaping
# + link-scheme validation): defense in depth, not a replacement for
# them. 'unsafe-inline' is unavoidable for now -- the UI is a single
# self-contained file with inline <script>/<style> and onclick=
# attributes, deliberately (no CDN, must work offline). So this does
# NOT stop an inline XSS from running. What it does stop is the part
# that actually hurts: connect-src 'self' means injected script can't
# POST the localStorage API token to an attacker's host, and
# default-src 'self' blocks pulling a payload from anywhere external.
# Splitting the JS into its own static file would let 'unsafe-inline'
# go away entirely -- worth doing, but a bigger change than this fix.
_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)
_UI_HEADERS = {
    "Content-Security-Policy": _CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


@app.get("/", response_class=HTMLResponse)
async def ui():
    static = Path(__file__).parent / "static" / "index.html"
    if static.exists():
        return HTMLResponse(static.read_text(encoding="utf-8"), headers=_UI_HEADERS)
    return HTMLResponse(
        "<h1>Forge UI not found</h1><p>Run from the installed package.</p>",
        headers=_UI_HEADERS,
    )
