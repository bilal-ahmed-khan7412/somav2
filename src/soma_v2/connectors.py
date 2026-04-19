import httpx
import json
import logging
import asyncio
from typing import Callable, Any, Awaitable

logger = logging.getLogger("SOMA_V2.CONNECTORS")

async def ollama_callback(model: str, task_type: str, prompt: str) -> str:
    """Standard Ollama connector using httpx."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model.split("/")[-1],
        "prompt": prompt,
        "stream": False,
        "format": "json" if "plan" in task_type.lower() else ""
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
    except Exception as e:
        logger.error(f"OllamaConnector: error calling {model} - {e}")
        raise

async def openai_callback(model: str, task_type: str, prompt: str) -> str:
    """Standard OpenAI connector (expects OPENAI_API_KEY in env)."""
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model.split("/")[-1],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenAIConnector: error calling {model} - {e}")
        raise

def get_llm_callback(model_str: str) -> Callable[[str, str], Awaitable[str]]:
    """
    Factory to return an async callback for a given model string.
    Supported prefixes: 'ollama/', 'openai/'
    """
    if model_str.startswith("ollama/"):
        return lambda t, p: ollama_callback(model_str, t, p)
    elif model_str.startswith("openai/"):
        return lambda t, p: openai_callback(model_str, t, p)
    else:
        # Default to local Ollama if no prefix
        return lambda t, p: ollama_callback(f"ollama/{model_str}", t, p)
