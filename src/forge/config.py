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

# --- OpenRouter -----------------------------------------------------------
# These were referenced by providers/llm_provider.py but never defined,
# which meant FORGE_PROVIDER=openrouter could never actually work.
OPENROUTER_URL = os.getenv(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# --- Runtime safety ---------------------------------------------------------
# Hard ceiling on how many router->tool steps a single run_agent() call
# may take. Forge today is single-shot (1 step), but the orchestrator
# enforces this ceiling unconditionally so future multi-step tools
# (roadmap: memory, filesystem, shell) cannot turn into infinite loops
# just because nobody remembered to add a guard.
MAX_STEPS = int(os.getenv("MAX_STEPS", "1"))

# --- Memory --------------------------------------------------------------
MEMORY_ENABLED = _bool("MEMORY_ENABLED", "true")
MEMORY_FILE = os.getenv("MEMORY_FILE", "data/memory.json")
MEMORY_MAX_HISTORY = int(os.getenv("MEMORY_MAX_HISTORY", "20"))

# --- Debug ------------------------------------------------------------------
SHOW_DEBUG = _bool("SHOW_DEBUG")
