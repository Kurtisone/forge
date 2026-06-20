import re

def parse_router_output(text: str):
    tool_match = re.search(r"<tool>(.*?)</tool>", text, re.DOTALL)
    content_match = re.search(r"<content>(.*?)</content>", text, re.DOTALL)

    if not tool_match or not content_match:
        return {
            "tool": "chat",
            "content": text.strip()
        }

    return {
        "tool": tool_match.group(1).strip(),
        "content": content_match.group(1).strip()
    }

def build_router_prompt(user_input: str):
    return f"""
You are a strict tool router.

Return ONLY in this format:

<tool>chat|code</tool>
<content>
...
</content>

RULES:
- NEVER add explanations
- NEVER add JSON
- NEVER add markdown
- ALWAYS include both tags
- tool must be either chat or code

EXAMPLES:

User: Write hello world in Python
<tool>code</tool>
<content>
print("Hello World")
</content>

User: Hello
<tool>chat</tool>
<content>
Hello!
</content>

User: {user_input}
"""
