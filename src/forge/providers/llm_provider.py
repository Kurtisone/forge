import requests
from forge.config import LLM_URL

def call_ollama(prompt, model):
    r = requests.post(
        LLM_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return r.json()["response"]


def call_openrouter(url, api_key, model, prompt):
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
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
