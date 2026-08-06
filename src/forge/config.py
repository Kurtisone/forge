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
# --- Prompt cache (v3.8) -------------------------------------------------
# Forge only ever drives a single conversation against this server, so
# there's no per-session slot pool to manage -- pin every request to the
# same slot so llama-server can reuse its KV cache across turns instead
# of treating each call as a fresh, unrelated prompt.
LLAMA_CPP_ID_SLOT = int(os.getenv("LLAMA_CPP_ID_SLOT", "0"))
LLAMA_CPP_CACHE_PROMPT = _bool("LLAMA_CPP_CACHE_PROMPT", "true")

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
# Raising MAX_STEPS above 1 is what opens the indirect prompt-injection
# surface described in audit E-2: from the second step onward, the
# previous step's tool output is part of the prompt that decides the
# next tool call, so the text of a web page or a log influences that
# decision. The guard in orchestrator.py answers that deterministically
# -- once a step has pulled in data Forge doesn't control (web_fetch,
# web_search, research, sysadmin), no later step in the same run may
# dispatch a mutating tool (shell, test, files:write).
#
# Set this to true only if you have a real multi-step flow that needs
# to write after fetching, and you trust every source it reads. It
# costs you the one guarantee that doesn't depend on the model
# behaving: prompt wording has failed three times on this project, a
# refusal in orchestrator.py cannot be talked out of. A blocked step
# is not a dead end either -- the same request asked again as a fresh
# turn starts with a clean slate.
ALLOW_MUTATION_AFTER_EXTERNAL_DATA = _bool(
    "ALLOW_MUTATION_AFTER_EXTERNAL_DATA", "false"
)

# --- Memory --------------------------------------------------------------
MEMORY_ENABLED = _bool("MEMORY_ENABLED", "true")
MEMORY_FILE = os.getenv("MEMORY_FILE", "data/memory.json")
# Was 20 (10 exchanges). Raised because the sliding-window eviction
# here fights KV-cache reuse in llama_cpp.py (v3.8): once the window
# is full, every new turn drops the oldest entry, shifting the whole
# history block's text and invalidating the cached prefix for that
# entire block on the server side -- confirmed in real testing
# (identical repeated messages still showed near-zero cache reuse
# once the window was full, on a server otherwise confirmed to be
# caching correctly on a non-hybrid-attention model). A higher cap
# doesn't remove the problem, it just makes eviction rare instead of
# constant; long-term recall is RAG's job (rag.py / v3.7), this is
# only the short-term conversational window shown to the router.
MEMORY_MAX_HISTORY = int(os.getenv("MEMORY_MAX_HISTORY", "100"))
# How far below the cap to trim when the hard cap does fire. Trimming
# to exactly MEMORY_MAX_HISTORY means the very next turn is over the
# cap again, so the oldest message is evicted every single turn -- the
# prompt prefix then shifts every turn and llama-server re-processes
# the whole thing, which is the FIFO problem v3.8 diagnosed, arriving
# through the back door. Cutting deeper trades a little more lost
# context for one expensive re-prefill every SLACK/2 exchanges instead
# of one per turn.
MEMORY_HARD_CAP_SLACK = int(os.getenv("MEMORY_HARD_CAP_SLACK", "20"))

# --- Context compaction / drawer (v3.9) ------------------------------------
# v3.8 raised MEMORY_MAX_HISTORY so FIFO eviction became rare instead
# of constant, but rare isn't never -- something still has to give
# once history keeps growing. Compaction replaces the oldest
# non-pinned messages with a single summary once COMPACTION_THRESHOLD
# is crossed, instead of just dropping them. MEMORY_MAX_HISTORY stays
# in place below as a hard-cap safety net in case compaction is
# disabled or its strategy fails outright (e.g. embedding server
# down) -- history must never grow unbounded either way.
COMPACTION_ENABLED = _bool("COMPACTION_ENABLED", "true")
# Trigger compaction once history reaches this many messages. Kept
# below MEMORY_MAX_HISTORY on purpose, to leave headroom instead of
# racing the hard cap.
COMPACTION_THRESHOLD = int(os.getenv("COMPACTION_THRESHOLD", "80"))
# Second trigger, on the size of the history as the ROUTER PROMPT
# renders it -- not as memory.json stores it. The two differ by more
# than a factor of two, because router/prompt.py truncates assistant
# entries to 120 chars on the way in.
#
# Both triggers are kept, and neither replaces the other. A message
# count cannot see a big paste: user entries are capped at 4000 chars,
# so one exchange costs ~90 rendered tokens in ordinary chat and ~1150
# at the ceiling -- a factor of thirteen. At that ceiling the context
# window is gone in about 24 messages, and a threshold of 80 would
# never fire in time. Conversely a token budget cannot see a flood of
# tiny entries, which costs little in the prompt but plenty in
# _format_history and in every persist.
#
# 6000 was chosen so the message trigger stays the common path: at
# ordinary rates this is ~133 messages, well past 80. It only takes
# over for the paste case, where it fires around 10 messages. The
# prompt at that point is roughly 8800 of 16384 tokens -- deliberately
# early, since compaction is cheap and a blown context window is not.
COMPACTION_TOKEN_THRESHOLD = int(os.getenv("COMPACTION_TOKEN_THRESHOLD", "6000"))
# What a pass aims to get BACK DOWN to, not just under.
#
# MEMORY_MAX_HISTORY below records what happens when eviction lands
# exactly on its limit: one message evicted per turn, which is the
# sliding window v3.8 removed, and which destroys KV-cache reuse. The
# same trap is worse here, because COMPACTION_KEEP_RECENT is a count of
# MESSAGES while the budget is in TOKENS -- how much a pass frees is
# not fixed. Aiming well below the threshold buys turns before the next
# pass instead of hovering at the line.
COMPACTION_TOKEN_TARGET = int(os.getenv("COMPACTION_TOKEN_TARGET", "3000"))
# How many of the most recent (non-pinned) messages are always left
# untouched by compaction. A FLOOR, not an exact count: a token-budget
# pass keeps compacting past this only when the budget still is not
# met, and never leaves fewer than this many behind.
COMPACTION_KEEP_RECENT = int(os.getenv("COMPACTION_KEEP_RECENT", "20"))
# "rag_pointer" (default): no LLM call -- push the compacted block into
# vector memory verbatim (rag.py) and replace it inline with a short
# pointer, searchable via !recall. "llm_summary": one LLM call per
# compaction, condenses the block into prose kept inline -- more
# faithful, costs tokens/latency. Both strategies share the same
# signature in forge/compaction.py, so switching is a one-line config
# change, not a rewrite.
COMPACTION_STRATEGY = os.getenv("COMPACTION_STRATEGY", "rag_pointer")

# --- Files tool workspace ---------------------------------------------------
# The files tool (forge.tools.files) confines all read/write/list
# operations to this directory. Paths outside it are rejected before
# any filesystem operation is attempted.
# Mount a volume here when running in a container:
#   podman run -v $(pwd):/workspace ...  and set WORKSPACE_DIR=/workspace
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "data/workspace")

# --- Shell tool -------------------------------------------------------------
SHELL_TIMEOUT = int(os.getenv("SHELL_TIMEOUT", "30"))
# The default no longer includes python3, pip or find. Each of those is
# an interpreter in its own right, so allowlisting it allowlists
# everything:
#   python3 -c "import os; os.system(...)"
#   pip install <url>            (runs setup.py, plus network egress)
#   find . -exec <anything> {} \;
# The allowlist only checks parts[0], never the arguments -- an
# allowlist containing an interpreter is not an allowlist.
#
# _SHELL_ALLOWLIST_DEFEATING below is the same idea generalised: adding
# any of those to SHELL_ALLOWED_COMMANDS is a decision to disable this
# protection, which is legitimate on a trusted local box and should be
# a deliberate act, not a default. tools/shell.py logs a warning at
# import when one is present, so the choice is visible in the logs.
_default_shell_cmds = "ls,cat,head,tail,wc,grep"
SHELL_ALLOWED_COMMANDS: set[str] = {
    c.strip()
    for c in os.getenv("SHELL_ALLOWED_COMMANDS", _default_shell_cmds).split(",")
    if c.strip()
}
# Not exhaustive and can't be: this is a "you are switching the
# allowlist off" tripwire, not a blocklist. Anything that can execute
# an arbitrary argument belongs here.
_SHELL_ALLOWLIST_DEFEATING: set[str] = {
    "awk",
    "bash",
    "env",
    "find",
    "gawk",
    "git",
    "less",
    "man",
    "more",
    "nano",
    "nc",
    "perl",
    "pip",
    "pip3",
    "python",
    "python3",
    "ruby",
    "sed",
    "sh",
    "ssh",
    "tar",
    "vi",
    "vim",
    "xargs",
    "zsh",
}

# --- Test/lint tool ----------------------------------------------------------
# Separate from SHELL_ALLOWED_COMMANDS on purpose: the test tool has its own
# narrower allowlist so "run the tests" / "lint this" stay first-class router
# intents with a purpose-built safety boundary, independent of whatever the
# general shell tool happens to allow.
TEST_TIMEOUT = int(os.getenv("TEST_TIMEOUT", "60"))
_default_test_cmds = "pytest,ruff"
TEST_ALLOWED_COMMANDS: set[str] = {
    c.strip()
    for c in os.getenv("TEST_ALLOWED_COMMANDS", _default_test_cmds).split(",")
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

# --- Delegation jobs --------------------------------------------------------
# Deliberately NOT a key inside memory.json: compaction rewrites that
# file in full, and the job runner writes to this one from another
# thread. Two whole-file writers on one file means the loser's work
# disappears, and here the loser would be queued work. See jobs.py.
JOBS_FILE = os.getenv("JOBS_FILE", "data/jobs.json")

# Wall-clock ceiling for one job, in seconds. Generous by default: the
# work being delegated is the kind that takes minutes, and a bound
# that fires on normal work is a bound that gets removed. It exists so
# that an executor which hangs cannot leave a job RUNNING forever --
# a state nothing else would ever move it out of.
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "1800"))

# Who carries out a job. "handoff" writes the spec into the workspace
# and stops -- the honest default while no implementer is reachable
# from the container. "echo" does nothing at all and exists for tests.
# A real executor is one class implementing executors.Executor; the
# rest of the delegation machinery does not change when one appears.
DELEGATE_EXECUTOR = os.getenv("DELEGATE_EXECUTOR", "handoff")

# Whether the delegate graph spends an LLM call drafting the spec
# before interviewing. OFF by default, on the evidence rather than on
# principle: across the first real delegations the draft cost 6-14 s
# and the only field it ever contributed was a workspace the user had
# already typed in the request -- everything else was either invented
# (and dropped by spec.ground) or left for the interview anyway. The
# interview turns it replaces cost 0 ms.
#
# The path is kept rather than deleted because the reason it fails is
# the model, not the design: a 9B under a grammar fills required keys
# instead of leaving them empty. Point call_llm at something stronger
# and this becomes worth its call again -- turn it back on and measure
# rather than rewriting it.
DELEGATE_DRAFT = os.getenv("DELEGATE_DRAFT", "false").lower() in ("1", "true", "yes")

# How long the "echo" executor pretends to work, in seconds. Exists so
# that cancelling a RUNNING job and restarting mid-run can be exercised
# for real: handoff finishes in milliseconds, so both paths had unit
# tests and no way to reproduce them against a live Forge.
DELEGATE_ECHO_SECONDS = float(os.getenv("DELEGATE_ECHO_SECONDS", "0"))

# How long the "echo" executor pretends to work, in seconds. Zero by
# default. It exists because cancelling a RUNNING job and restarting
# mid-run are the two behaviours that cannot be exercised by hand
# otherwise: handoff writes a file and returns in milliseconds, so
# there is never a window to cancel inside. Unit tests cover both, and
# a behaviour only ever verified by its own test is one nobody has
# actually seen work.
DELEGATE_ECHO_SECONDS = float(os.getenv("DELEGATE_ECHO_SECONDS", "0"))

# --- Debug ------------------------------------------------------------------
SHOW_DEBUG = _bool("SHOW_DEBUG")

# --- API auth -----------------------------------------------------------
# Optional bearer token for the HTTP API (api.py). Empty by default:
# the API stays open, exactly like before this was added. Set this
# before exposing forge-core on anything beyond localhost/trusted LAN
# -- /chat, /review, /run, /traces and /tools currently have zero
# protection otherwise.
API_TOKEN = os.getenv("API_TOKEN", "")
# Forge refuses to start with no API_TOKEN unless this is set to true.
# The old behaviour was "open by default, documented as risky" -- but a
# documented unsafe default is still an unsafe default, and this one is
# the kind you notice the day you add a published port to a compose
# file, not before. Flipping it means the risky configuration has to be
# written down in .env.local, where it's visible, instead of being what
# happens when you write nothing at all. Local-only development is a
# perfectly good reason to set it; forgetting isn't.
API_ALLOW_UNAUTHENTICATED = _bool("API_ALLOW_UNAUTHENTICATED", "false")

# Interactive docs (/docs, /redoc, /openapi.json). FastAPI mounts all
# three by default and none of them can carry a Depends(require_token)
# -- they're wired up by the framework, not by this app's routes. So
# on an instance reachable by anything other than you, they publish a
# complete, machine-readable map of every endpoint, its parameters and
# its schemas to an unauthenticated caller. That's not a
# vulnerability by itself; it's the reconnaissance step made free, and
# it's free for an attacker who has learned nothing else about the
# instance. Off by default, on where it's useful (audit M-3): set
# API_DOCS_ENABLED=true while developing against the API.
API_DOCS_ENABLED = _bool("API_DOCS_ENABLED", "false")

# --- API rate limiting ----------------------------------------------------
# In-memory sliding window, per client IP, single-process only (see
# forge/ratelimit.py). Defaults are generous for interactive/UI use
# and mainly matter if the API is hammered or scripted against.
RATE_LIMIT_ENABLED = _bool("RATE_LIMIT_ENABLED", "true")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# --- Vector memory / RAG (v3.7) --------------------------------------------
# Separate llama.cpp instance (forge-embedding container), embedding-only,
# distinct from LLAMA_CPP_URL which stays dedicated to chat/tool-dispatch.
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://127.0.0.1:8082/embedding")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "30"))
# An embedding server has to hold the whole input in one physical
# batch (pooling is non-causal, unlike generation), so llama-server
# rejects anything longer than --ubatch-size with a 400, not a
# truncated result. Its default is 512 tokens. rag.py splits longer
# text into chunks of at most this many characters, embeds each, and
# averages -- ~1500 chars stays under 512 tokens even for dense French
# or code. Raise it if you also raise -b/-ub on the embedding server.
EMBEDDING_MAX_CHARS = int(os.getenv("EMBEDDING_MAX_CHARS", "1500"))
# Ceiling on chunks per call, so one oversized input can't turn into
# hundreds of sequential HTTP requests. Beyond this the tail is
# dropped, with a warning.
EMBEDDING_MAX_CHUNKS = int(os.getenv("EMBEDDING_MAX_CHUNKS", "16"))
RAG_DB_FILE = os.getenv("RAG_DB_FILE", "data/forge_rag.db")
# Per-hit ceiling on what the memory tool feeds back into the router
# prompt. A "fact" entry is one line, but a "history_summary" written
# by compaction holds a whole evicted conversation block, so five hits
# could be several thousand characters of mostly-irrelevant transcript
# -- burying the one line that answered the question and paying for it
# twice, in prompt tokens and in prefill time.
MEMORY_RECALL_MAX_CHARS = int(os.getenv("MEMORY_RECALL_MAX_CHARS", "500"))

# --- Web fetch tool ----------------------------------------------------------
# WEB_FETCH_ALLOWED_DOMAINS is empty by default (any public domain is
# fetchable) -- the SSRF guard in tools/web_fetch.py (blocking private/
# loopback/link-local resolved IPs) is NOT configurable and always applies,
# regardless of this allowlist. This matters specifically because Forge
# itself sits on a home network (NiPoGi behind WireGuard) that a
# router-hallucinated URL must never be able to reach.
#
# Known limitation, not fixed by raising this value alone: heavy
# corporate portals (observed live: boursorama.com) often wrap large
# navigation megamenus in generic <div>s instead of semantic
# <nav>/<header>/<footer>/<aside>, which tools/web_fetch.py's
# extractor skips by tag name only -- real content on such sites can
# still be pushed far down past a lot of noise. A real "main content"
# heuristic (à la Readability/trafilatura) would need a new
# dependency, which this tool deliberately avoids -- see
# tools/web_fetch.py's module docstring. web_fetch stays best-effort
# on non-semantic sites; it's reliable on simpler/standards-compliant
# pages (docs, articles, wikis).
WEB_FETCH_TIMEOUT = int(os.getenv("WEB_FETCH_TIMEOUT", "15"))
WEB_FETCH_MAX_BYTES = int(os.getenv("WEB_FETCH_MAX_BYTES", str(2 * 1024 * 1024)))
WEB_FETCH_ALLOWED_DOMAINS: set[str] = {
    d.strip().lower()
    for d in os.getenv("WEB_FETCH_ALLOWED_DOMAINS", "").split(",")
    if d.strip()
}

# --- Web search tool (SearXNG) ------------------------------------------------
# Requires a self-hosted SearXNG instance -- not a cloud search API, in
# keeping with Forge's self-hosting posture. SearXNG must have "json" added
# to search.formats in its own settings.yml (disabled by default upstream,
# to discourage scraping public instances -- safe to enable on a private,
# self-hosted one that only Forge talks to). Distinct from web_fetch: this
# queries a search index for ranked results, it does not fetch a page whose
# URL is already known.
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888")
SEARXNG_TIMEOUT = int(os.getenv("SEARXNG_TIMEOUT", "10"))
SEARXNG_MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", "5"))

# --- Research graph (search -> fetch top N -> synthesize) --------------------
# Deliberately a deterministic sequence, not a router-driven multi-step
# chain: see graphs/research.py's module docstring for why -- the router
# repeatedly failed to reliably follow a "search then decide what's next"
# instruction with this model, even with explicit worked examples, so the
# decision was removed from the router's hands entirely for this flow.
RESEARCH_FETCH_TOP_N = int(os.getenv("RESEARCH_FETCH_TOP_N", "3"))
RESEARCH_FETCH_CHARS_PER_RESULT = int(
    os.getenv("RESEARCH_FETCH_CHARS_PER_RESULT", "1500")
)

# --- Recall graph (recall -> synthesize) -------------------------------------
# Same reasoning and same fix as the research graph above, applied to
# memory: chaining memory:recall into a synthesis step via the
# router's "done": false steering hint reliably failed live (see
# graphs/recall.py's module docstring) -- the router repeated the
# identical recall call instead of phrasing an answer from it. A
# recall answer is one or two facts restated as a sentence, not a
# multi-source summary, so its cap is far smaller than research's.
RECALL_MAX_ANSWER_CHARS = int(os.getenv("RECALL_MAX_ANSWER_CHARS", "800"))

# Recall answering a French question in English is the failure this
# guards. The prompt names the detected language (forge/lang.py); this
# knob controls the half that doesn't trust the prompt -- checking the
# answer and, when it is demonstrably in the wrong language, spending
# ONE more call to ask again. Priced deliberately: the retry only ever
# fires on a run that was already wrong, and never fires at all when
# the language of either text is uncertain. Set false to keep the
# naming and drop the retry.
RECALL_ENFORCE_LANGUAGE = os.getenv("RECALL_ENFORCE_LANGUAGE", "true").lower() == "true"

# --- Sysadmin graph (discover -> collect -> synthesize) ---------------------
# Deliberately read-only, always: no command in graphs/sysadmin.py can
# mutate anything (no systemctl restart/stop, no podman stop/rm) -- this
# mirrors the tools/git.py decision to keep that tool strictly read-only
# too, with any write action requiring an explicit human-confirmed flow
# outside the router's reach. Not configurable here on purpose, unlike
# SHELL_ALLOWED_COMMANDS: sysadmin's command set is fixed in code, not
# environment-extensible, so enabling this tool can never accidentally
# grant more than "read logs, discover what's running."
SYSADMIN_DISCOVERY_TIMEOUT = int(os.getenv("SYSADMIN_DISCOVERY_TIMEOUT", "10"))
SYSADMIN_COLLECT_TIMEOUT = int(os.getenv("SYSADMIN_COLLECT_TIMEOUT", "15"))
# 200 was the original default and blew the 4096-token context on its
# own (200 journalctl lines ~= 4000+ tokens before the rest of the
# prompt) -- see the SYSADMIN_LOG_CHARS_BUDGET note below for the
# actual fix; this default is now just a sane collection size, not
# the thing keeping the prompt in budget.
SYSADMIN_MAX_LOG_LINES = int(os.getenv("SYSADMIN_MAX_LOG_LINES", "40"))
# Hard character cap on the log block actually inserted into the LLM
# prompt, independent of line count -- same reasoning as
# RESEARCH_FETCH_CHARS_PER_RESULT in graphs/research.py. Truncates
# keeping the END of the log (most recent events), not the start:
# journalctl/podman logs already return the tail via -n/--tail, so
# the most relevant lines are at the end of that output.
SYSADMIN_LOG_CHARS_BUDGET = int(os.getenv("SYSADMIN_LOG_CHARS_BUDGET", "2000"))

# --- Read-only access to the HOST's journalctl/systemctl/podman ------------
# Forge's own container has no business holding real systemd/podman
# privilege (mutation) -- see deploy/README.md for the full design.
# Empty (default) means "talk to whatever is reachable normally",
# which inside Forge's own minimal container image is nothing at all
# (no journalctl/systemctl/podman binaries, no bus, no socket) --
# each of these three only does anything once the matching deploy/
# artifact is wired up:
#
# - SYSADMIN_JOURNAL_DIR: host's /var/log/journal bind-mounted
#   read-only into the container (e.g. "/host-journal"). journalctl
#   reads journal files directly -- no daemon, no socket -- so this
#   one needs nothing beyond the RO bind mount + the binary itself.
# - SYSADMIN_DBUS_ADDRESS: address of the FILTERED bus exposed by
#   deploy/forge-dbus-proxy.sh (xdg-dbus-proxy), never the host's real
#   system bus directly -- the proxy allows only read-only systemd
#   method calls (ListUnits/GetUnit), denies everything else
#   (StartUnit/StopUnit/...) at the bus level itself, before Forge's
#   own code is ever in a position to decide anything.
# - SYSADMIN_PODMAN_URL: address of deploy/podman_ro_proxy.py, never
#   the host's real podman.sock directly -- that proxy allows only
#   GET /containers/json and GET /containers/{id}/logs, rejecting
#   every other verb/path (start/stop/rm/exec/...) before it reaches
#   the real socket.
SYSADMIN_JOURNAL_DIR = os.getenv("SYSADMIN_JOURNAL_DIR", "")
SYSADMIN_DBUS_ADDRESS = os.getenv("SYSADMIN_DBUS_ADDRESS", "")
SYSADMIN_PODMAN_URL = os.getenv("SYSADMIN_PODMAN_URL", "")

# --- Policy Engine ----------------------------------------------------------
# Transversal gate consulted before a capability runs (ARCHITECTURE.md,
# "Policy Engine"). Each flag denies a whole class of capability by the
# static requirements it declares, regardless of which tool it is.
#
# All three default to true, so an untouched deployment behaves exactly
# as before. They exist to make a degraded context expressible as
# configuration rather than as a code path: NiPoGi offline or a metered
# connection -> POLICY_ALLOW_NETWORK=false and research / web_fetch /
# web_search stop being reachable while chat, code, memory and review
# keep working. A machine where Forge should look but never touch ->
# POLICY_ALLOW_WORKSPACE_WRITES=false.
#
# This is a deny gate, not a permission grant: it can only subtract from
# what ENABLED_TOOLS already allows. Turning a flag on never makes a
# tool reachable that was not already opted in.
POLICY_ALLOW_NETWORK = _bool("POLICY_ALLOW_NETWORK", "true")
POLICY_ALLOW_WORKSPACE_WRITES = _bool("POLICY_ALLOW_WORKSPACE_WRITES", "true")
POLICY_ALLOW_SUBPROCESS = _bool("POLICY_ALLOW_SUBPROCESS", "true")
