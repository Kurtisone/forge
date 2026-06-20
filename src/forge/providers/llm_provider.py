import requests
from forge.config import (
    FORGE_PROVIDER,
    OLLAMA_URL,
    LLAMA_CPP_URL,
    LLM_MODEL
)

def call_llama_cpp(url: str, model: str, prompt: str):
    try:
        r = requests.post(
            f"{url}/completion",
            json={
                "prompt": prompt,
                "n_predict": 512,
                "temperature": 0.0,
                "stop": ["</tool>", "<tool>"]
            },
            timeout=120
        )
        r.raise_for_status()
        data = r.json()
        return data.get("content") or data.get("completion") or ""
    except Exception as e:
        return f"[LLAMA_CPP_ERROR] {str(e)}"

def call_ollama(url: str, model: str, prompt: str):
    r = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    data = r.json()
    return data.get("response") or data.get("content")


def call_openrouter(url: str, api_key: str, model: str, prompt: str):
    r = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-OpenRouter-Title": "Forge"
        },
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }
    )

    data = r.json()

    if "error" in data:
        raise Exception(data["error"])

    if "choices" not in data:
        raise Exception(f"Bad response: {data}")

    return data["choices"][0]["message"]["content"]
