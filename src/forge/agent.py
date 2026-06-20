from forge.llm import call_llm
from forge.tools.router import build_router_prompt, parse_router_output
from forge.tools.registry import load_tools, get_tool

load_tools()

def run_agent(user_input: str):
    raw = call_llm(build_router_prompt(user_input))
    data = parse_router_output(raw)

    tool = data.get("tool", "chat")
    content = data.get("content")

    if not content:
        content = user_input

    handler = get_tool(tool)

    if not handler:
         handler = get_tool("chat")

    return handler(content)
