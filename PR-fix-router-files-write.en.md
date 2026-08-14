# fix/router-files-write — `files:write` through the router

Two preexisting bugs on `main`, found during the security lot 3 pre-merge
testing (so not lot 3 regressions). Together they made `files:write` from
chat fail for essentially **any real file content**.

Base: `1cdc26d`. **551 tests green** (baseline 514), `ruff check` +
`ruff format --check` clean, applied and verified with `git am` on a fresh
clone of `main`.

## Diagnosis

**1. The JSON object scanner counted braces without context.**
`_all_json_objects` incremented on `{` and decremented on `}` including
inside string literals. But a router decision's `content` carries a
payload whose own `content` is file text — and real code has braces that
don't balance on their own (a truncated Go/C/Rust function, a `}` in a
comment, a Python dict). The depth either hit zero too early (unparseable
candidate) or never (scan abandoned), the object was dropped, and the
output fell through to the plain-text fallback.

`hello.py` worked — which is what masked the bug for so long — because
`print('...')` contains no brace at all.

**2. The model does not hold double escaping.**
Nesting a JSON payload inside a JSON string requires a second level of
escaping: `\\n` where a plain string needs `\n`. The 9B writes `\n`. The
outer parse then yields inner text carrying a raw newline, and the tool's
own `json.loads` dies on *invalid control character*.

## The fixes

**`fix(router)` — string-aware scanner.** Brace counting now skips string
literals, escapes included. Fixed along the way: an unclosed `{` used to
`break` and make *every later object* unreachable (its closing brace only
ever brought the depth from 2 to 1, never to 0); the search now resumes
one character further on.

**`feat(router)` + `fix(router)` — `content` MUST be an object for
JSON-payload tools.** First attempt: `content ::= string | object` for
every tool. Not enough — both branches stayed reachable, and against a
9B's prior for the escaped-string shape, the worked examples lost.
Confirmed on the first real `files:write`, which came back as an escaped
string and died on the unescaped quotes in `import "fmt"`.

The actual fix: the grammar conditions the shape of `content` on the tool,
which GBNF can express because `tool` is pinned before `content`. `root`
splits into `payload-call` (object content, for
files/memory/review/sysadmin) and `text-call` (string content, for
chat/code and the rest). A branch with no tools is omitted (an empty
alternation is unsatisfiable).

This is *why* the constraint has to live in the grammar. In the string
shape, the file body needs double escaping that the grammar cannot
check — to `schar`, the whole payload is just characters. In the object
shape, the body is an ordinary JSON string, so `schar` (which excludes a
bare `"` and control characters) enforces its escaping during sampling.
The failure stops being caught after the fact and becomes impossible to
produce.

**`feat(router)` — re-encoding mechanics.** The parser re-encodes an
object with `json.dumps`, so `RouterDecision.content` stays a string and
**no `run(content: str)` contract changes**: tools still parse JSON text,
produced by Forge instead of the model. The object rule pulls in full JSON
values, so a numeric payload field doesn't have to be stringified to
satisfy sampling. The parser still *accepts* the string shape — providers
without GBNF have no other option.

**`feat(router)` — the prompt teaches the object shape.** A small local
model imitates its worked examples far more than it reads descriptions: as
long as they showed an escaped string, that is what it produced. All four
JSON-payload tools (`files`, `memory`, `review`, `sysadmin`) switch
together — two competing shapes side by side in the same prompt is how
routing gets unstable on a 9B (same lesson as the `review`/`files`
ambiguity in v3.10). The `hello.py` example becomes multi-line: it only
ever exercised the one case that was never the problem.

**`fix(router)` — hyphenated rule names.** The tool-conditioned grammar
was structurally valid, passed every test, and was rejected outright by
llama-server: `expecting newline or end at _call`. llama.cpp lexes rule
names with `is_word_char()`, which accepts `[a-zA-Z0-9-]` and **not**
underscore — so `payload_call` reads as the rule `payload` followed by
garbage. The result was a 400 on every completion: the router did not
degrade, it died (16 ms, no routing at all). The real gap was in the
tests, which only asserted the generated *text* — precisely the blind
spot. `is_word_char()` is now reimplemented in the test suite and every
rule name checked against it, verified by reverting the rename and
watching the test fail.

**`fix(tools)` — shared lenient JSON loader.** The string shape does not
go away (grammar disabled, a provider without GBNF, model drift).
`forge/tool_payload.loads_payload()` retries with `strict=False`,
**deliberately as a second attempt**: strict parsing succeeding is the
signal that the model is producing correct JSON, and defaulting to lenient
would hide the next escaping regression instead of surfacing it. The
recovery logs a warning and an event. Shared across all four tools rather
than fixed in `files.py` alone — this repo has twice been bitten by fixing
one copy of a shared behaviour and letting its twin diverge (`review` vs
`research`, see `text_cleaning.py`).

**`feat(files)` — echo the content back when a write creates a file.**
Modifying a file returned a real diff; creating one returned a byte count
and nothing else, so after "create hello.go" the content was never shown
anywhere and had to be opened by hand. A new file now comes back in a
fenced block (language hinted from the extension, capped at 4 KB with a
truncation notice). Beyond the display win, this puts the content in the
conversation, where a follow-up turn has something to refer back to.

**`feat(files)` — one-step `edit` action.** "Replace X with Y" was the
last file operation that still needed the router to chain: read with
`done:false`, then write the whole file back. Observed live, the run
stopped after the read and answered with the file's ORIGINAL content —
exactly the symptom the read example had been added to fix in v3.9. This
is the same non-chaining that already forced deterministic handling twice
(`web_search` in v3.10, `memory:recall` in `fix/memory-recall`); betting
on `done:false` a third time was not reasonable.

`{"action":"edit","path":...,"find":...,"replace":...}` does the
replacement in one dispatch. Beyond removing the chaining, it removes the
file content's round trip through the model entirely: the model supplies
two short strings and never has to reproduce a file it just read — which
is where a 9B quietly "fixes" things along the way (the v3.9 hallucination
bug). A `find` with no match is an error, not a silent no-op, so the model
can fall back to read-then-write for a change that isn't literal. `edit`
is confined to `WORKSPACE_DIR` like read and write, with its own test: a
new action is a new chance to reintroduce the v3.10 escape.

## Out of scope, fixed in passing

`memory.run()` called `.get()` straight off the parse, so a valid but
non-object JSON payload (`"recall"`, a list) raised `AttributeError` out
of a tool whose contract is to return its errors as text. `files`,
`review` and `sysadmin` already guarded this.

## Verification

End to end outside the tests, router → `files.run()` → file on disk:

| content | before | after |
|---|---|---|
| balanced Go | object found, write failed | written |
| truncated JS (`() => {`) | 0 objects, text fallback | written |
| C with an extra `}` | 0 objects, text fallback | written |
| Python dict | object found, write failed | written |
| under-escaped string shape | *invalid control character* | written (with a warning) |
| genuinely malformed payload | error | error (unchanged) |

The tests assert the invariant rather than the escaping they kept breaking
on: **every worked example in the prompt must parse into the tool it
names**, and no JSON-payload example may re-encode its payload as a
string. A malformed future example now breaks a test instead of teaching
the model to fail.

## Real-world validation

Three iterations were needed, each exposing a different cause:

| run | result | cause |
|---|---|---|
| `#76cccf2f` | failure | `content ::= string \| object` left the string shape reachable; the model took it and died on `import "fmt"` |
| `#4862ec0e` | failure | llama-server rejected the grammar (underscore in rule names) — 400 on every completion |
| `#be63eb8d` | **success** | `hello.go` written, `import "fmt"` intact |
| `#127a948d` | **success** | `test.go` created, content shown in a fenced block |
| `#fa53a1e6` | **success** | replacement routed to `files:edit` in one step, correct diff |

The router picks `edit` over a `read`, and extracts the literal string
from a natural-language sentence.

## Known limitation, accepted

`edit` covers **literal** replacement, which is the common case. A
non-literal change — "add a function", "restructure this file" — still
depends on `read→write` chaining, i.e. on the behaviour that failed in run
`#00b1c5f8`. That path and the steering hint after a `files:read` are kept
for those cases.

The logical next step if it becomes a problem: a deterministic `edit`
graph modelled on `research` and `recall` — read, have the LLM produce
only the changed portion, write. That's a separate piece of work, not a
patch.
