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
MEMORY_MAX_HISTORY = int(os.getenv("MEMORY_MAX_HISTORY", "20"))

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
