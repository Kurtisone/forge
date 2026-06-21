import requests

from forge.errors import ProviderError


def call(url: str, model: str, prompt: str) -> str:
    try:
        r = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise ProviderError(f"ollama request failed: {e}") from e

    data = r.json()
    content = data.get("response") or data.get("content")
    if not content:
        raise ProviderError(f"ollama returned no content: {data}")
    return content
