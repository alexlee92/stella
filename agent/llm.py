# llm.py
import requests

from agent.config import OLLAMA_URL, MODEL


def ask_llm(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}
    )

    return response.json()["response"]
