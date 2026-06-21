import requests

from forge.errors import ProviderError


def call(url: str, model: str, prompt: str) -> str:
    try:
        r = requests.post(
            f"{url}/completion",
            json={
                "prompt": prompt,
                "n_predict": 200,
                "temperature": 0.0,
                # The prompt only ever asks for a single JSON object - no
                # <tool> tags exist anywhere in it, so those old stop
                # sequences never matched and the model kept generating
                # fake "User: ..." turns until it hit n_predict. Stop as
                # soon as the model starts hallucinating a new turn.
                "stop": ["\nUser:", "\nUser :", "User:", "\n\n"],
            },
            timeout=120,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise ProviderError(f"llama_cpp request failed: {e}") from e

    data = r.json()
    content = data.get("content") or data.get("completion")
    if not content:
        raise ProviderError(f"llama_cpp returned no content: {data}")
    return content
