import os

# ----------------------------
# PROVIDER
# ----------------------------
FORGE_PROVIDER = os.getenv("FORGE_PROVIDER", "local")

# ----------------------------
# MODEL
# ----------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")

# ----------------------------
# LOCAL OLLAMA
# ----------------------------
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate"
)

# ----------------------------
# LLAMA.CPP
# ----------------------------
LLAMA_CPP_URL = os.getenv(
    "LLAMA_CPP_URL",
    "http://127.0.0.1:8080"
)

# ----------------------------
# OPENROUTER
# ----------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = os.getenv(
    "OPENROUTER_URL",
    "https://openrouter.ai/api/v1/chat/completions"
)

# ----------------------------
# REASONING
# ----------------------------
SHOW_REASONING = os.getenv("SHOW_REASONING", "false").lower() == "true"
