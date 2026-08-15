import requests

from forge.errors import ProviderError
from forge.providers import error_body
from forge.types import Completion, Usage


def call(url: str, model: str, prompt: str) -> Completion:
    try:
        r = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
    except requests.RequestException as e:
        raise ProviderError(f"ollama request failed: {e}") from e

    try:
        r.raise_for_status()
    except requests.RequestException as e:
        raise ProviderError(f"ollama request failed: {e}{error_body(r)}") from e

    data = r.json()
    content = data.get("response") or data.get("content")
    if not content:
        raise ProviderError(f"ollama returned no content: {data}")

    return Completion(
        text=content,
        usage=Usage(
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        ),
    )
