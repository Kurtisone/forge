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

Forge v2.2 ("Clean Runtime") enforces a strict separation between the
three layers: the **LLM** (router prompt + providers), **tools**
(dispatch + handlers), and **logs** (the only place anything is
printed). The orchestrator is the single point where they meet, and
it is the only place a loop guard (`MAX_STEPS`) can ever apply.

```
src/forge/
│
├── orchestrator.py    # the only orchestrator (agent.py is a compat alias)
├── llm.py             # LLM dispatch — never called from anywhere else
├── config.py          # the only module that reads os.getenv()
├── logger.py           # the only module allowed to print/log
├── errors.py           # typed exception hierarchy (ForgeError, ...)
├── types.py             # RouterDecision / ToolResult / AgentResult dataclasses
│
├── router/
│   ├── prompt.py        # router prompt template — isolated, nothing else builds prompts
│   └── parser.py         # raw LLM text -> RouterDecision
│
├── tools/
│   ├── registry.py        # discovery; failures are logged, never swallowed
│   ├── chat.py
│   ├── code.py
│   └── files.py / git.py / shell.py   # roadmap stubs, no run() yet
│
└── providers/
    ├── llama_cpp.py
    ├── ollama.py
    └── openrouter.py
```

Data flow:
```
user_input
   ↓
Orchestrator._route()      -> RouterDecision   (LLM layer)
   ↓
Orchestrator._dispatch()   -> ToolResult        (tools layer)
   ↓
AgentResult                                      (returned to caller)
```
Every step along the way emits a structured event through `log.event()`
(visible only when `SHOW_DEBUG=true`), and is bounded by `MAX_STEPS`
with cycle detection so the same `(tool, content)` pair can never be
dispatched twice in a single run.

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
|FORGE_PROVIDER	| LLM backend (ollama, llama_cpp, openrouter)	| llama_cpp                           |
|LLM_MODEL	    | Model name	                    | default                             |
|OLLAMA_URL	    | Ollama endpoint	                | http://127.0.0.1:11434/api/generate |
|LLAMA_CPP_URL	| llama.cpp endpoint	            | http://127.0.0.1:8080               |
|OPENROUTER_URL	| OpenRouter endpoint	            | https://openrouter.ai/api/v1/chat/completions |
|OPENROUTER_API_KEY	| OpenRouter API key	        | (empty)                             |
|MAX_STEPS	    | Hard ceiling on router→tool steps per run (loop guard) | 1                |
|SHOW_DEBUG	    | Emit structured debug trace (router prompt, raw output, tool dispatch, timings) | false |

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
