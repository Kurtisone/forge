"""
Runtime configuration, read once from the environment.

This is the only file allowed to call os.getenv(). Everything else
imports values from here, so there is exactly one place to look when
something is misconfigured.
"""

import os


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# --- Provider selection -----------------------------------------------
FORGE_PROVIDER = os.getenv("FORGE_PROVIDER", "llama_cpp")
LLM_MODEL = os.getenv("LLM_MODEL", "default")

# --- Ollama -------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")

# --- llama.cpp ------------------------------------------------------------
LLAMA_CPP_URL = os.getenv("LLAMA_CPP_URL", "http://127.0.0.1:8080")
# At ~7 t/s on a Steam Deck, 512 tokens ≈ 73s, well within the default
# 120s timeout. Raise LLAMA_CPP_N_PREDICT if you want longer answers,
# but always keep it below LLAMA_CPP_TIMEOUT * tokens_per_second.
LLAMA_CPP_TIMEOUT = int(os.getenv("LLAMA_CPP_TIMEOUT", "120"))
LLAMA_CPP_N_PREDICT = int(os.getenv("LLAMA_CPP_N_PREDICT", "512"))
# Grammar-constrained decoding (see router/grammar.py): forces the
# model's raw output to match the router's exact JSON schema at the
# sampling level, instead of relying on prompt instructions + post-hoc
# parsing alone. Disable if your llama.cpp server version doesn't
# support the "grammar" completion field, or to rule it out while
# debugging.
LLAMA_CPP_USE_GRAMMAR = _bool("LLAMA_CPP_USE_GRAMMAR", "true")
# --- Prompt cache (v3.8) -------------------------------------------------
# Forge only ever drives a single conversation against this server, so
# there's no per-session slot pool to manage -- pin every request to the
# same slot so llama-server can reuse its KV cache across turns instead
# of treating each call as a fresh, unrelated prompt.
LLAMA_CPP_ID_SLOT = int(os.getenv("LLAMA_CPP_ID_SLOT", "0"))
LLAMA_CPP_CACHE_PROMPT = _bool("LLAMA_CPP_CACHE_PROMPT", "true")

# --- OpenRouter -----------------------------------------------------------
# These were referenced by providers/llm_provider.py but never defined,
# which meant FORGE_PROVIDER=openrouter could never actually work.
OPENROUTER_URL = os.getenv(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# --- Runtime safety ---------------------------------------------------------
# Hard ceiling on how many router->tool steps a single run() call may
# take. Default is 1 (single-shot). The orchestrator only goes beyond
# one step if the router's own JSON explicitly sets "done": false --
# see Orchestrator.run() -- so this ceiling exists to make sure that,
# whatever a model decides, a run can't loop forever.
MAX_STEPS = int(os.getenv("MAX_STEPS", "1"))

# --- Memory --------------------------------------------------------------
MEMORY_ENABLED = _bool("MEMORY_ENABLED", "true")
MEMORY_FILE = os.getenv("MEMORY_FILE", "data/memory.json")
# Was 20 (10 exchanges). Raised because the sliding-window eviction
# here fights KV-cache reuse in llama_cpp.py (v3.8): once the window
# is full, every new turn drops the oldest entry, shifting the whole
# history block's text and invalidating the cached prefix for that
# entire block on the server side -- confirmed in real testing
# (identical repeated messages still showed near-zero cache reuse
# once the window was full, on a server otherwise confirmed to be
# caching correctly on a non-hybrid-attention model). A higher cap
# doesn't remove the problem, it just makes eviction rare instead of
# constant; long-term recall is RAG's job (rag.py / v3.7), this is
# only the short-term conversational window shown to the router.
MEMORY_MAX_HISTORY = int(os.getenv("MEMORY_MAX_HISTORY", "100"))

# --- Context compaction / drawer (v3.9) ------------------------------------
# v3.8 raised MEMORY_MAX_HISTORY so FIFO eviction became rare instead
# of constant, but rare isn't never -- something still has to give
# once history keeps growing. Compaction replaces the oldest
# non-pinned messages with a single summary once COMPACTION_THRESHOLD
# is crossed, instead of just dropping them. MEMORY_MAX_HISTORY stays
# in place below as a hard-cap safety net in case compaction is
# disabled or its strategy fails outright (e.g. embedding server
# down) -- history must never grow unbounded either way.
COMPACTION_ENABLED = _bool("COMPACTION_ENABLED", "true")
# Trigger compaction once history reaches this many messages. Kept
# below MEMORY_MAX_HISTORY on purpose, to leave headroom instead of
# racing the hard cap.
COMPACTION_THRESHOLD = int(os.getenv("COMPACTION_THRESHOLD", "80"))
# How many of the most recent (non-pinned) messages are always left
# untouched by compaction.
COMPACTION_KEEP_RECENT = int(os.getenv("COMPACTION_KEEP_RECENT", "20"))
# "rag_pointer" (default): no LLM call -- push the compacted block into
# vector memory verbatim (rag.py) and replace it inline with a short
# pointer, searchable via !recall. "llm_summary": one LLM call per
# compaction, condenses the block into prose kept inline -- more
# faithful, costs tokens/latency. Both strategies share the same
# signature in forge/compaction.py, so switching is a one-line config
# change, not a rewrite.
COMPACTION_STRATEGY = os.getenv("COMPACTION_STRATEGY", "rag_pointer")

# --- Files tool workspace ---------------------------------------------------
# The files tool (forge.tools.files) confines all read/write/list
# operations to this directory. Paths outside it are rejected before
# any filesystem operation is attempted.
# Mount a volume here when running in a container:
#   podman run -v $(pwd):/workspace ...  and set WORKSPACE_DIR=/workspace
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "data/workspace")

# --- Shell tool -------------------------------------------------------------
SHELL_TIMEOUT = int(os.getenv("SHELL_TIMEOUT", "30"))
_default_shell_cmds = "ls,cat,head,tail,wc,grep,find,python3,pip,pytest"
SHELL_ALLOWED_COMMANDS: set[str] = {
    c.strip()
    for c in os.getenv("SHELL_ALLOWED_COMMANDS", _default_shell_cmds).split(",")
    if c.strip()
}

# --- Test/lint tool ----------------------------------------------------------
# Separate from SHELL_ALLOWED_COMMANDS on purpose: the test tool has its own
# narrower allowlist so "run the tests" / "lint this" stay first-class router
# intents with a purpose-built safety boundary, independent of whatever the
# general shell tool happens to allow.
TEST_TIMEOUT = int(os.getenv("TEST_TIMEOUT", "60"))
_default_test_cmds = "pytest,ruff"
TEST_ALLOWED_COMMANDS: set[str] = {
    c.strip()
    for c in os.getenv("TEST_ALLOWED_COMMANDS", _default_test_cmds).split(",")
    if c.strip()
}

# --- Tool allowlist ---------------------------------------------------------# A module exposing run() in src/forge/tools/ is NOT dispatchable just
# because it exists. It must also be explicitly listed here. This is
# the guard that matters once files.py / git.py / shell.py stop being
# empty stubs: implementing run() in shell.py must not silently make
# shell execution reachable from router output -- it has to be opted
# into on purpose, here.
ENABLED_TOOLS = {
    name.strip()
    for name in os.getenv("ENABLED_TOOLS", "chat,code").split(",")
    if name.strip()
}

# --- Execution trace --------------------------------------------------------
# When enabled, each run() appends a JSONL record to TRACE_FILE.
# One JSON object per line — inspect with:
#   cat data/traces.jsonl | python -m json.tool
#   tail -n1 data/traces.jsonl | jq .
#   !trace  (inside Forge REPL)
TRACE_ENABLED = _bool("TRACE_ENABLED", "true")
TRACE_FILE = os.getenv("TRACE_FILE", "data/traces.jsonl")

# --- Debug ------------------------------------------------------------------
SHOW_DEBUG = _bool("SHOW_DEBUG")

# --- API auth -----------------------------------------------------------
# Optional bearer token for the HTTP API (api.py). Empty by default:
# the API stays open, exactly like before this was added. Set this
# before exposing forge-core on anything beyond localhost/trusted LAN
# -- /chat, /review, /run, /traces and /tools currently have zero
# protection otherwise.
API_TOKEN = os.getenv("API_TOKEN", "")

# --- API rate limiting ----------------------------------------------------
# In-memory sliding window, per client IP, single-process only (see
# forge/ratelimit.py). Defaults are generous for interactive/UI use
# and mainly matter if the API is hammered or scripted against.
RATE_LIMIT_ENABLED = _bool("RATE_LIMIT_ENABLED", "true")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# --- Vector memory / RAG (v3.7) --------------------------------------------
# Separate llama.cpp instance (forge-embedding container), embedding-only,
# distinct from LLAMA_CPP_URL which stays dedicated to chat/tool-dispatch.
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://127.0.0.1:8082/embedding")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "30"))
RAG_DB_FILE = os.getenv("RAG_DB_FILE", "data/forge_rag.db")
