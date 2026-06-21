from forge.config import SHOW_DEBUG
from forge.errors import ForgeError
from forge.logger import log
from forge.orchestrator import Orchestrator


def main():
    orchestrator = Orchestrator()

    print("Forge ready" + (" [debug]" if SHOW_DEBUG else "") + ". Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Forge > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("exit", "quit"):
            break

        if not user_input.strip():
            continue

        try:
            result = orchestrator.run(user_input)
        except ForgeError as e:
            log.error("unhandled runtime error: %s", e)
            print(f"\n[error] {e}\n")
            continue

        print("\n" + str(result.output) + "\n")


if __name__ == "__main__":
    main()
