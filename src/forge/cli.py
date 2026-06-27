"""
Forge CLI — command-line entry points beyond the REPL.

Available commands:
  forge review <file> [question]   — review a file with the LLM
  forge replay <run_id>            — replay a past execution trace

Run from container:
  podman run --rm --env-file .env.local \\
    -v $(pwd):/workspace forge-core \\
    python -m forge.cli review src/forge/main.py

Or directly:
  PYTHONPATH=src python -m forge.cli review src/forge/main.py
"""

import sys


def _cmd_review(args: list[str]) -> int:
    if not args:
        print("Usage: forge review <file> [question]", file=sys.stderr)
        return 1

    file_path = args[0]
    question = " ".join(args[1:]) if len(args) > 1 else "Que peut-on améliorer ?"

    from forge.graphs.review import run
    print(f"Reviewing {file_path!r}…\n")
    result = run(file_path, question=question)
    print(result)
    return 0


def _cmd_replay(args: list[str]) -> int:
    if not args:
        print("Usage: forge replay <run_id>", file=sys.stderr)
        return 1

    run_id = args[0]
    from forge import trace
    traces = trace.read_last(100)
    matches = [t for t in traces if t.get("run_id", "").startswith(run_id)]

    if not matches:
        print(f"No trace found for run_id starting with {run_id!r}", file=sys.stderr)
        return 1

    t = matches[-1]
    print(f"Run ID  : {t.get('run_id')}")
    print(f"Time    : {t.get('timestamp')}")
    print(f"Input   : {t.get('user_input_preview')!r}")
    print(f"Status  : {'✓ ok' if t.get('ok') else '✗ failed'}")
    print(f"Total   : {t.get('total_ms')}ms")
    print()
    for step in t.get("steps", []):
        ok = "✓" if step.get("tool_ok") else "✗"
        print(f"  [{ok}] {step.get('router_tool')}  {step.get('duration_ms')}ms")
        if step.get("router_content_preview"):
            print(f"       → {step.get('router_content_preview')!r}")
        if step.get("tool_error"):
            print(f"       ! {step.get('tool_error')}")
    if t.get("error"):
        print(f"\nError: {t.get('error')}")
    return 0


_COMMANDS = {
    "review": _cmd_review,
    "replay": _cmd_replay,
}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    cmd = args[0]
    if cmd not in _COMMANDS:
        print(f"Unknown command: {cmd!r}", file=sys.stderr)
        print(f"Available: {', '.join(_COMMANDS)}", file=sys.stderr)
        sys.exit(1)

    sys.exit(_COMMANDS[cmd](args[1:]))


if __name__ == "__main__":
    main()
