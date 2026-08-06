from forge.kernel.capability import LOCAL_READONLY

# Pure string formatting, no side effect of any kind.
REQUIREMENTS = LOCAL_READONLY


def run(content: str):
    return f"```python\n{content}\n```"
