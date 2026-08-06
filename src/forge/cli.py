"""
Forge CLI — command-line entry points beyond the REPL.

Available commands:
  forge review <file> [question] [--tests <path>]   — review a file
                                                        with the LLM,
                                                        optionally
                                                        running its
                                                        tests first
  forge replay <run_id>                              — replay a past
                                                        execution trace
  forge capabilities                                 — list what Forge
                                                        can currently do
                                                        and what each
                                                        capability needs

Run from container:
  podman run --rm --env-file .env.local \\
    -v $(pwd):/workspace forge-core \\
    python -m forge.cli review src/forge/main.py

Or directly:
  PYTHONPATH=src python -m forge.cli review src/forge/main.py
  PYTHONPATH=src python -m forge.cli review src/forge/graph.py --tests tests/test_graph.py
"""

import sys


def _cmd_review(args: list[str]) -> int:
    if not args:
        print("Usage: forge review <file> [question] [--tests <path>]", file=sys.stderr)
        return 1

    test_path = None
    if "--tests" in args:
        idx = args.index("--tests")
        if idx + 1 >= len(args):
            print("Usage: --tests requires a path argument", file=sys.stderr)
            return 1
        test_path = args[idx + 1]
        args = args[:idx] + args[idx + 2 :]

    file_path = args[0]
    question = " ".join(args[1:]) if len(args) > 1 else "Que peut-on améliorer ?"

    from forge.graphs.review import run

    print(f"Reviewing {file_path!r}…\n")
    result = run(file_path, question=question, test_path=test_path)
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


def _cmd_capabilities(args: list[str]) -> int:
    from forge.kernel import registry as capabilities
    from forge.tools.registry import load_tools

    load_tools()

    names = capabilities.capability_names()
    if not names:
        print("No capabilities registered — check ENABLED_TOOLS in .env.local.")
        return 0

    width = max(max(len(n) for n in names), len("CAPABILITY"))
    print(f"{len(names)} capabilit{'y' if len(names) == 1 else 'ies'} registered\n")
    print(f"   {'CAPABILITY'.ljust(width)}  {'PROVIDER'.ljust(width)}  REQUIRES")
    for name in names:
        for cap in capabilities.candidates(name):
            mark = " " if cap.declared else "!"
            print(
                f" {mark} {cap.name.ljust(width)}  "
                f"{cap.provider.ljust(width)}  {cap.requirements.summary()}"
            )

    missing = capabilities.undeclared()
    if missing:
        print(
            f"\n! {len(missing)} provider(s) declare no REQUIREMENTS and are shown "
            "with the most demanding profile."
        )
    return 0


_COMMANDS = {
    "review": _cmd_review,
    "replay": _cmd_replay,
    "capabilities": _cmd_capabilities,
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
