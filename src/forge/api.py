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
/health. Leaving it unset no longer silently opens the API -- the app
refuses to start unless API_ALLOW_UNAUTHENTICATED=true says the open
posture is intentional. See check_auth_configuration() below.

Rate limiting: in-memory sliding window, per client IP, on every
endpoint except /. RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW_SECONDS
(default: 30 per 60s). Set RATE_LIMIT_ENABLED=false to disable.
/health stays unauthenticated (healthchecks and the UI status line
have no token yet) but is metered like the rest: each hit costs an
outbound call to llama.cpp.

Interactive docs (/docs, /redoc, /openapi.json) are off unless
API_DOCS_ENABLED=true. FastAPI mounts them itself, so they cannot be
put behind require_token.

Run:
  uvicorn forge.api:app --host 0.0.0.0 --port 8000

The LLM calls are blocking (HTTP to llama.cpp / Ollama). They run
in a thread-pool executor so FastAPI's event loop is never blocked.
"""

import asyncio
import hmac
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from forge import jobs, memory, rag, ratelimit, trace
from forge.config import (
    API_ALLOW_UNAUTHENTICATED,
    API_DOCS_ENABLED,
    API_TOKEN,
    FORGE_PROVIDER,
    LLAMA_CPP_URL,
    LLM_MODEL,
    MEMORY_ENABLED,
)
from forge.logger import log
from forge.orchestrator import Orchestrator
from forge.router import build_router_prompt
from forge.tokens import estimate_tokens


class InsecureConfiguration(RuntimeError):
    """Raised at startup when the API would come up unauthenticated
    without anyone having asked for that in writing."""


def check_auth_configuration() -> None:
    """
    Refuse to start with no API_TOKEN unless API_ALLOW_UNAUTHENTICATED
    is explicitly set.

    /chat dispatches whatever is in ENABLED_TOOLS -- shell, files, test,
    sysadmin -- and the container's CMD binds 0.0.0.0. So "no token" is
    not a mild default: it's arbitrary tool dispatch for anyone who can
    reach the port. Failing loudly at startup is the only version of
    this warning that can't be scrolled past.

    Reads the module globals rather than the config constants directly
    so tests (and anything embedding the app) can patch them at the
    same boundary the auth dependency already uses.
    """
    if API_TOKEN or API_ALLOW_UNAUTHENTICATED:
        return
    raise InsecureConfiguration(
        "refusing to start: API_TOKEN is unset, so /chat, /run and every "
        "other endpoint would accept unauthenticated requests -- including "
        "tool dispatch. Set API_TOKEN in .env.local, or set "
        "API_ALLOW_UNAUTHENTICATED=true if this instance really is "
        "local-only and you want it open on purpose."
    )


def log_effective_settings() -> None:
    """
    Print the settings whose wrong value has no symptom other than
    latency, once, at startup.

    LLAMA_CPP_CACHE_PROMPT sat at false in the container for weeks
    while config.py's default said true, costing roughly 75% of every
    run's wall time. Nothing was broken, nothing was logged, and the
    hunt went through the model architecture and three llama-server
    flags before reaching the .env. The compaction thresholds are in
    the same class: a value nobody remembers setting produces passes
    that look inexplicable in the log and one slow turn afterwards.

    Reads the modules' own attributes rather than re-importing the
    constants, so what is printed is what the code will actually use.
    """
    from forge import compaction
    from forge.providers import llama_cpp

    log.event(
        "config.effective",
        provider=FORGE_PROVIDER,
        cache_prompt=llama_cpp.LLAMA_CPP_CACHE_PROMPT,
        use_grammar=llama_cpp.LLAMA_CPP_USE_GRAMMAR,
        memory_enabled=MEMORY_ENABLED,
        compaction_enabled=compaction.COMPACTION_ENABLED,
        compaction_threshold=compaction.COMPACTION_THRESHOLD,
        compaction_token_threshold=compaction.COMPACTION_TOKEN_THRESHOLD,
        compaction_token_target=compaction.COMPACTION_TOKEN_TARGET,
        compaction_keep_recent=compaction.COMPACTION_KEEP_RECENT,
        compaction_strategy=compaction.COMPACTION_STRATEGY,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    check_auth_configuration()
    log_effective_settings()
    # A job left RUNNING on disk is a lie as soon as this process is
    # new: whatever was executing it died with the previous one.
    jobs.reconcile()
    yield


# docs_url/redoc_url/openapi_url are None unless API_DOCS_ENABLED says
# otherwise (audit M-3). These three routes are mounted by FastAPI
# itself, so they can't take Depends(require_token) the way this app's
# own routes do -- there is no version of them that is behind the
# token. Off by default; API_DOCS_ENABLED=true turns them back on for
# development.
app = FastAPI(
    title="Forge",
    version="3.3.0",
    docs_url="/docs" if API_DOCS_ENABLED else None,
    redoc_url="/redoc" if API_DOCS_ENABLED else None,
    openapi_url="/openapi.json" if API_DOCS_ENABLED else None,
    lifespan=lifespan,
)
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
    # Absent, not zeroed, when no accounting scope was open -- same
    # convention as the trace's "llm" block. A client must be able to
    # tell "not reported" from "reported as nothing", which matters
    # here because the header gauge has to decide between showing a
    # number and showing nothing at all.
    usage: dict | None = None


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


# Rate-limited but deliberately still unauthenticated (audit M-3):
# it's what a container healthcheck and the UI's own status line call
# before there's a token to send. The limit is the point -- every hit
# makes an outbound HTTP call to llama.cpp to read the loaded model
# name, so an unmetered /health turns one cheap request from an
# anonymous caller into load on the inference server. That's an
# amplifier, not just a chatty endpoint.
@app.get("/health", dependencies=[Depends(rate_limit)])
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


async def _context_limit() -> int | None:
    """
    The context window, asked of the server that owns it. None for any
    provider that cannot say -- the gauge then shows a used figure with
    no denominator rather than inventing one.
    """
    if FORGE_PROVIDER != "llama_cpp":
        return None

    from forge.providers.llama_cpp import get_context_size

    return await _run_in_thread(get_context_size, LLAMA_CPP_URL)


async def _context_usage(usage: dict | None) -> dict | None:
    """
    Add the two fields the header gauge needs to the run's own totals.

    `estimated` is not decoration. Everything in *usage* comes from the
    backend's own counts, but the gauge also has to render before any
    call has happened (GET /context on page load), where the only
    number available is tokens.estimate_tokens. Showing an estimate as
    if it were a measurement is precisely the confusion this whole lot
    exists to remove, so the flag travels with the numbers.
    """
    if usage is None:
        return None
    return {
        **usage,
        "context_limit": await _context_limit(),
        "estimated": False,
    }


@app.get("/context", dependencies=[Depends(require_token), Depends(rate_limit)])
async def get_context():
    """
    What the next turn will cost, before it is sent.

    /chat can only report a window that has already been used, so on a
    freshly loaded page there is nothing to show. This is the answer to
    "am I near the limit?", which is a question worth asking BEFORE
    writing rather than after.

    Necessarily an estimate: the exact count only exists once
    llama-server has seen the prompt. It is built from the same
    rendered history block the router would send, not from the stored
    messages -- assistant entries are truncated to 120 chars on the way
    into the prompt, so the stored size runs more than double the real
    one and would make this gauge alarmist for no reason.
    """
    history = memory.get_history() if MEMORY_ENABLED else []
    # Builds the prompt but never sends it. Worth being explicit about:
    # an LLM call here would be a passive UI poll writing into
    # llama-server's pinned slot, evicting the KV prefix the next real
    # turn depends on -- a silent factor-of-twenty on latency, for a
    # gauge.
    prompt = build_router_prompt("", history=history)
    return {
        "prompt_tokens": estimate_tokens(prompt),
        "history_messages": len(history),
        "context_limit": await _context_limit(),
        "estimated": True,
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
        usage=await _context_usage(result.usage),
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


@app.get("/jobs", dependencies=[Depends(require_token), Depends(rate_limit)])
async def list_jobs():
    """
    Every delegation job and its state.

    Forge's own interface is the conversation thread -- the zero-tab
    rule -- so this is not where a job is meant to be read day to day.
    It exists so that "what is this job actually doing" has an answer
    that does not require reading data/jobs.json over SSH from a
    phone.
    """
    return {"jobs": [job.to_dict() for job in jobs.all_jobs()]}


@app.get("/tools", dependencies=[Depends(require_token), Depends(rate_limit)])
async def list_tools():
    """
    Return the currently enabled tools and available graphs.

    "tools" reports what is enabled AND permitted right now, matching
    what the router is actually offered -- a caller checking whether a
    capability is usable should get the same answer the router gets.
    "denied" names what the active policy is subtracting, so a tool
    missing from the list is explained rather than merely absent.
    """
    from forge.kernel.registry import allowed_names, capability_names

    allowed = allowed_names()
    return {
        "tools": allowed,
        "denied": [name for name in capability_names() if name not in allowed],
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
