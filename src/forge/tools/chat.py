from forge.kernel.capability import LOCAL_READONLY

# Pass-through: the router already produced the text.
REQUIREMENTS = LOCAL_READONLY


def run(content: str):
    return content
