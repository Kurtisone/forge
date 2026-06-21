import requests

from forge.errors import ProviderError


def call(url: str, api_key: str, model: str, prompt: str) -> str:
    if not api_key:
        raise ProviderError("OPENROUTER_API_KEY is not set")

    try:
        r = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-OpenRouter-Title": "Forge",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 500,
            },
            timeout=120,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise ProviderError(f"openrouter request failed: {e}") from e

    data = r.json()

    if "error" in data:
        raise ProviderError(f"openrouter error: {data['error']}")
    if "choices" not in data:
        raise ProviderError(f"openrouter bad response: {data}")

    return data["choices"][0]["message"]["content"]
