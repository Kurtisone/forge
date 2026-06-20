import os

FORGE_PROVIDER = os.getenv("FORGE_PROVIDER", "llama_cpp")
LLM_MODEL = os.getenv("LLM_MODEL", "default")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
LLAMA_CPP_URL = os.getenv("LLAMA_CPP_URL", "http://127.0.0.1:8080")

SHOW_DEBUG = os.getenv("SHOW_DEBUG", "false").lower() == "true"
