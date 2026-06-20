# Forge

Forge is a lightweight LLM-based agent runtime built around a router + tool execution model.
Instead of relying on a monolithic prompt or complex reasoning loops, Forge delegates actions to explicit tools selected by a structured LLM router.

---

### Core Concept

Forge works in 3 steps:
```
User Input
   ↓
LLM Router (JSON output)
   ↓
Tool Dispatcher
   ├── chat
   └── code
```
The model does not “decide freely” — it must output a strict JSON instruction describing which tool to use.

---

### Router Format

The LLM must return:
```
{
  "tool": "chat" | "code",
  "content": "string"
}
```
Example

Input:
```
write a python script to print hello world
```
Output:
```
{
  "tool": "code",
  "content": "print('Hello World')"
}
```
---

### Available Tools

>chat

Used for conversational responses.
```
chat(content: str) -> str
```

>code

Used to generate executable Python code.
```
generate_code(content: str) -> str
```
---

### Architecture
```
src/forge/
│
├── agent.py          # core orchestrator
├── llm.py            # LLM provider wrapper
├── config.py         # runtime configuration
│
├── tools/
│   ├── router.py     # prompt + JSON parser
│   ├── chat.py       # chat tool
│   └── code.py       # code tool
│
└── providers/
    └── llm_provider.py
```
---

### Usage
```
podman run -it --rm \
  --env-file .env.local \
  -v $(pwd)/data:/app/data \
  forge-core
```
---

### Configuration

Environment variables:

|Variable       | Description	                    |Default                              |
|---------------|:---------------------------------:|:-----------------------------------:|
|FORGE_PROVIDER	| LLM backend (ollama, llama_cpp)	| llama_cpp                           |
|LLM_MODEL	    | Model name	                    | default                             |
|OLLAMA_URL	    | Ollama endpoint	                | http://127.0.0.1:11434/api/generate |
|LLAMA_CPP_URL	| llama.cpp endpoint	            | http://127.0.0.1:8080               |
|SHOW_DEBUG	    | Debug output	                    | false                               |

---

### Design Philosophy

Forge is built on three principles:

Deterministic routing over free-form reasoning
Explicit tool execution instead of implicit behavior
Minimal agent core, maximal extensibility

The goal is to make LLM behavior predictable, debuggable, and composable.

---

### Roadmap

Planned extensions:

- memory tool (persistent context)
- filesystem tool (read/write local files)
- shell tool (sandboxed execution)
- multi-agent routing layer
- plugin-based tool registry

---

### Status

Forge v2 is an experimental runtime, not a production framework.
Expect frequent breaking changes.

---

### License

MIT
