# Forge

Lightweight LLM agent framework for local and cloud models (Ollama / OpenRouter)

---

## Overview

Forge is a minimal experimental framework for building LLM-based agents with a clear separation between:

- Agent logic
- LLM provider abstraction
- Local and cloud inference backends
- Tool system (work in progress)

---

## Features

- Multi-provider support (OpenRouter / Ollama)
- Modular LLM abstraction layer
- Simple CLI agent loop
- Extensible tool system (filesystem, shell, git)
- Container-ready (Podman / Docker)

---

## Architecture

User → CLI (main.py) → Agent (agent.py) → LLM Router (llm.py) → Provider Layer (llm_provider.py) → OpenRouter / Ollama

---

## Installation

Build container:

podman build -t forge .

Run:

podman run -it --env-file .env forge

---

## Configuration

Create a .env file:

FORGE_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key_here
FORGE_MODEL=meta-llama/llama-3.2-3b-instruct:free

---

## Usage

Forge > Hello

---

## Design goals

- Clean LLM abstraction
- Local/cloud switching
- Minimal agent architecture
- Extensible tool system

---

## Roadmap

- Memory system
- Tool execution (shell, git, filesystem)
- Streaming responses
- Multi-agent support
- Config file support

---

## Status

Early MVP for experimentation and learning

---

## License

MIT
