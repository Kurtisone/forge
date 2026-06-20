from forge.agent import run_agent

def main():
    print("Forge V1 ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("Forge > ")

        if user_input.lower() in ["exit", "quit"]:
            break

        response = run_agent(user_input)

        print("\n" + str(response) + "\n")


if __name__ == "__main__":
    main()
