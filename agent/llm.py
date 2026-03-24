import backoff
import os
from typing import Tuple
import requests
import litellm
from dotenv import load_dotenv
import json

load_dotenv()

MAX_TOKENS = 4096

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CODER_MODEL = os.getenv("OLLAMA_MODEL", "ollama_chat/qwen2.5-coder:7b")
OLLAMA_SMALL_MODEL = "ollama_chat/qwen2.5-coder:3b"
DEFAULT_MODEL = OLLAMA_CODER_MODEL

litellm.drop_params = True


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError, KeyError),
    max_time=600,
    max_value=60,
)
def get_response_from_llm(
    msg: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history=None,
) -> Tuple[str, list, dict]:
    if msg_history is None:
        msg_history = []

    # Convert text to content, compatible with LITELLM API (without mutating original)
    msg_history = [
        {k: v for k, v in msg.items() if k != "text"} | {"content": msg["text"]}
        if "text" in msg else msg
        for msg in msg_history
    ]

    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    completion_kwargs = {
        "model": model,
        "messages": new_msg_history,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Set api_base for Ollama models
    if model.startswith("ollama_chat/") or model.startswith("ollama/"):
        completion_kwargs["api_base"] = OLLAMA_HOST

    response = litellm.completion(**completion_kwargs)
    response_text = response['choices'][0]['message']['content']  # pyright: ignore
    new_msg_history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})

    # Convert content to text, compatible with MetaGen API
    new_msg_history = [
        {**msg, "text": msg.pop("content")} if "content" in msg else msg
        for msg in new_msg_history
    ]

    return response_text, new_msg_history, {}


if __name__ == "__main__":
    msg = 'Write a Python function that checks if a number is prime.'
    models = [
        ("OLLAMA_CODER_MODEL", OLLAMA_CODER_MODEL),
        ("OLLAMA_SMALL_MODEL", OLLAMA_SMALL_MODEL),
    ]
    for name, model in models:
        print(f"\n{'='*50}")
        print(f"Testing {name}: {model}")
        print('='*50)
        try:
            output_msg, msg_history, info = get_response_from_llm(msg, model=model)
            print(f"OK: {output_msg[:200]}...")
        except Exception as e:
            print(f"FAIL: {str(e)[:200]}")
