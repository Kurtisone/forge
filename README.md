# Forge

Forge is a lightweight LLM-based agent runtime built around a router + tool execution model.
Instead of relying on a monolithic prompt or complex reasoning loops, Forge delegates actions
to explicit tools selected by a structured LLM router.

---

### Core Concept

```
User Input
   ↓
LLM Router  (structured JSON decision)
   ↓
Tool Dispatcher
   ├── chat    (conversational response)
   └── code    (code generation)
```

The model does not "decide freely" — it must output a strict JSON instruction
(`{"tool": "...", "content": "..."}`) describing which tool to invoke.
The router is resilient: it handles JSON, XML tool-call format (Qwen HERETIC),
markdown code fences, and plain text as fallbacks, in that order.

---

### Architecture

Forge v2.3 enforces a strict separation between three layers:
the **LLM** (router prompt + providers), **tools** (dispatch + handlers),
and **logs** (the only module allowed to print anything).
The orchestrator is the single point where they meet.

```
src/forge/
│
├── orchestrator.py    # single orchestrator — MAX_STEPS loop guard + cycle detection
├── llm.py             # LLM dispatch — called from nowhere else
├── config.py          # sole reader of os.getenv()
├── logger.py          # sole logger; SHOW_DEBUG gates structured trace events
├── errors.py          # typed exception hierarchy (ForgeError, ProviderError, …)
├── types.py           # RouterDecision / ToolResult / AgentResult dataclasses
│
├── router/
│   ├── prompt.py      # router prompt template — isolated; nothing else builds prompts
│   └── parser.py      # raw LLM output → RouterDecision (4-step cascade)
│
├── tools/
│   ├── registry.py    # discovery + ENABLED_TOOLS allowlist; failures logged, never swallowed
│   ├── chat.py
│   ├── code.py
│   └── files.py / git.py / shell.py   # roadmap stubs — having run() is not enough;
│                                       # must also appear in ENABLED_TOOLS to be reachable
│
├── memory.py          # JSON-backed rolling history + key/value facts
│
└── providers/
    ├── llama_cpp.py
    ├── ollama.py
    └── openrouter.py
```

Data flow per turn:
```
user_input
   ↓
Orchestrator._route()      →  RouterDecision   (LLM layer)
   ↓
Orchestrator._dispatch()   →  ToolResult       (tools layer)
   ↓
AgentResult                                     (returned to caller)
```

Every step emits a structured event via `log.event()` (visible only when `SHOW_DEBUG=true`),
and is bounded by `MAX_STEPS` with cycle detection so the same `(tool, content)` pair
can never be dispatched twice in a single run.

---

### Usage

```bash
podman build -t forge-core .
podman run -it --rm \
  --env-file .env.local \
  -v $(pwd)/data:/app/data \
  forge-core
```

Multi-line / code paste: wrap in ` ``` ` fences, or paste `question + code` together
(the REPL detects buffered stdin and combines them into a single turn automatically).
Type `!help` inside Forge for REPL commands.

---

### Configuration

| Variable | Description | Default |
|---|---|---|
| `FORGE_PROVIDER` | LLM backend: `llama_cpp`, `ollama`, `openrouter` | `llama_cpp` |
| `LLM_MODEL` | Model name | `default` |
| `OLLAMA_URL` | Ollama endpoint | `http://127.0.0.1:11434/api/generate` |
| `LLAMA_CPP_URL` | llama.cpp endpoint | `http://127.0.0.1:8080` |
| `LLAMA_CPP_N_PREDICT` | Max tokens per llama.cpp response | `512` |
| `LLAMA_CPP_TIMEOUT` | HTTP timeout for llama.cpp requests (seconds) | `120` |
| `OPENROUTER_URL` | OpenRouter endpoint | `https://openrouter.ai/api/v1/chat/completions` |
| `OPENROUTER_API_KEY` | OpenRouter API key | *(empty)* |
| `MAX_STEPS` | Hard ceiling on router→tool steps per run | `1` |
| `ENABLED_TOOLS` | Comma-separated allowlist of dispatchable tools | `chat,code` |
| `MEMORY_ENABLED` | Persist and recall conversation history | `true` |
| `MEMORY_FILE` | Path to the JSON memory file | `data/memory.json` |
| `MEMORY_MAX_HISTORY` | Number of past messages kept in the prompt | `20` |
| `SHOW_DEBUG` | Emit full structured trace (prompt, raw output, dispatch, timings) | `false` |

---

### Memory

Forge keeps a rolling window of the last `MEMORY_MAX_HISTORY` messages in `MEMORY_FILE`
and injects it as context into the router prompt on every turn.

Storage is plain JSON — no schema, no migrations, `cat data/memory.json` to inspect it.
The module also exposes a `facts` key/value store (`add_fact` / `get_facts`) not yet
wired into the router prompt — reserved for a future "remember this" tool.

Only successful turns are persisted; error replies are never written to memory.
Large pastes are truncated before saving to avoid bloating future prompts.
SQLite becomes relevant if/when Forge ever needs concurrent writers or indexed queries — not before.

---

### Design Philosophy

- **Deterministic routing over free-form reasoning** — the model picks a tool from a fixed set,
  not an open-ended plan.
- **Explicit tool execution** — a tool is only reachable if it has `run()` *and* is listed in
  `ENABLED_TOOLS`. Code existing is not enough.
- **Typed boundaries** — `RouterDecision`, `ToolResult`, `AgentResult` dataclasses at every
  interface; raw dicts never cross module boundaries.
- **Best-effort memory** — a memory read/write failure logs a warning and is ignored;
  it never breaks a conversation turn.
- **Local-first** — llama.cpp and Ollama are first-class backends; no cloud dependency required.

---

### Roadmap

| Version | Focus |
|---|---|
| **v2.3** *(current)* | Robustness: ToolResult contract, ENABLED_TOOLS allowlist, parser cascade (XML/fences/think blocks), memory hardening, REPL paste detection |
| **v2.4** | Structured execution trace — per-turn step log with replay; `AgentState` carrying input/memory/steps/results as a single object through the run |
| **v3.0** | Graph execution engine — `Node.execute(state) → state`, conditional edges, tool chaining; Forge becomes a local-first mini LangGraph without the cloud dependency |

Tools planned: `files` (read/write local files), `shell` (sandboxed execution), `git` (repo ops).
Each requires explicit `ENABLED_TOOLS` opt-in and sandboxing before activation.

---

### Status

Forge is an experimental local runtime, not a production framework.
The public API (orchestrator, tool registry, providers) is stabilising from v2.3 onward.

---

### License

MIT
