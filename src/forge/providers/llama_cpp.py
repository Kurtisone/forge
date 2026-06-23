import requests

from forge.config import LLAMA_CPP_N_PREDICT, LLAMA_CPP_TIMEOUT
from forge.errors import ProviderError


def call(url: str, model: str, prompt: str) -> str:
    try:
        r = requests.post(
            f"{url}/completion",
            json={
                "prompt": prompt,
                "temperature": 0.0,
                "n_predict": LLAMA_CPP_N_PREDICT,
                "stop": [
                    # Prevent the model from hallucinating a new dialogue turn
                    "\nUser:", "\nUser :", "User:",
                    # Qwen HERETIC XML tool-call format: stop after the full
                    # tool_call block (parser extracts <content> from it)
                    "</tool_call>",
                    # NOTE: "\n\n" intentionally absent — it would cut any
                    # multi-line code response mid-generation before the
                    # JSON or XML closing tag is reached.
                ],
            },
            timeout=LLAMA_CPP_TIMEOUT,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise ProviderError(f"llama_cpp request failed: {e}") from e

    data = r.json()
    content = data.get("content") or data.get("completion")
    if not content:
        raise ProviderError(f"llama_cpp returned no content: {data}")
    return content
