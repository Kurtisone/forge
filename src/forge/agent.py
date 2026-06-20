from forge.llm import call_llm
from forge.tools.router import build_router_prompt, parse_router_output
from forge.tools.code import generate_code
from forge.tools.chat import chat


def run_agent(user_input: str):

    raw = call_llm(build_router_prompt(user_input))
    data = parse_router_output(raw)

    tool = data["tool"]
    content = data["content"]

    if tool == "code":
        return generate_code(content)

    return chat(content)
