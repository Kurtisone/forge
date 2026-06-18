import os

FORGE_PROVIDER = os.getenv("FORGE_PROVIDER", "local")

# Local (Ollama)
LLM_URL = os.getenv("FORGE_LLM_URL", "http://127.0.0.1:11434/api/generate")
LLM_MODEL = os.getenv("FORGE_MODEL", "llama3.1")

# Cloud (OpenRouter style)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
