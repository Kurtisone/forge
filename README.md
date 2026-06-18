# Forge

Forge is a lightweight LLM agent framework designed for experimentation with local and cloud-based AI providers.

It supports:
- OpenRouter (cloud LLMs)
- Ollama (local LLMs)
- Modular provider architecture
- Simple CLI agent loop
- Extensible tool system (WIP)

---

## Why this project exists

This project was built to explore:
- How to design a minimal agent architecture
- How to abstract LLM providers cleanly
- How to switch between local and cloud inference
- How to structure an extensible tool-based AI system

---

## Architecture
User
↓
CLI (main.py)
↓
Agent (agent.py)
↓
LLM Router (llm.py)
↓
Provider Layer (llm_provider.py)
↓
OpenRouter / Ollama

---

## Setup

### Build container

```bash
podman build -t forge .
```

### Run
```bash
podman run -it --env-file .env forge
```

## Environment variables

Create a .env file:

FORGE_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key_here
FORGE_MODEL=meta-llama/llama-3.2-3b-instruct:free

## Status

This project is an early MVP and actively evolving.
Planned features:

Memory system (short-term + persistent)
Tool execution (shell, git, filesystem)
Multi-agent orchestration
