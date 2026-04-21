import httpx
import json
import logging
import asyncio
from typing import Callable, Any, Awaitable

logger = logging.getLogger("SOMA_V2.CONNECTORS")

# Global semaphore to limit concurrent LLM requests
LLM_SEMAPHORE = asyncio.Semaphore(5)

async def ollama_callback(model: str, task_type: str, prompt: str) -> str:
    """Standard Ollama connector using httpx."""
    async with LLM_SEMAPHORE:
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

async def google_callback(model: str, task_type: str, prompt: str) -> str:
    """Google Gemini connector."""
    async with LLM_SEMAPHORE:
        import os
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        # model string like google/gemini-1.5-flash
        model_name = model.split("/")[-1]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json" if "plan" in task_type.lower() else "text/plain"
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                # Gemini response structure: candidates[0].content.parts[0].text
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"GoogleConnector: error calling {model} - {e}")
            raise

async def generic_openai_callback(url: str, api_key_env: str, model: str, task_type: str, prompt: str) -> str:
    """Generic OpenAI-compatible connector (Groq, Deepseek, OpenAI)."""
    async with LLM_SEMAPHORE:
        import os
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} not found in environment")
        
        headers = {"Authorization": f"Bearer {api_key}"}
        
        effective_prompt = prompt
        json_mode = "plan" in task_type.lower()
        
        if json_mode and "json" not in prompt.lower():
            effective_prompt += "\n\n(Respond in valid JSON format)"
                
        payload = {
            "model": model.split("/")[-1],
            "messages": [{"role": "user", "content": effective_prompt}],
            "temperature": 0.1,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"GenericOpenAIConnector ({api_key_env}): error calling {model} - {e}")
            raise

async def openai_callback(model: str, task_type: str, prompt: str) -> str:
    """Standard OpenAI connector."""
    return await generic_openai_callback(
        "https://api.openai.com/v1/chat/completions",
        "OPENAI_API_KEY", model, task_type, prompt
    )

def get_llm_callback(model_str: str) -> Callable[[str, str], Awaitable[str]]:
    """
    Factory to return an async callback for a given model string.
    Supported prefixes: 'ollama/', 'openai/', 'google/', 'groq/', 'deepseek/'
    """
    if model_str.startswith("ollama/"):
        return lambda t, p: ollama_callback(model_str, t, p)
    elif model_str.startswith("openai/"):
        return lambda t, p: openai_callback(model_str, t, p)
    elif model_str.startswith("google/"):
        return lambda t, p: google_callback(model_str, t, p)
    elif model_str.startswith("groq/"):
        return lambda t, p: generic_openai_callback(
            "https://api.groq.com/openai/v1/chat/completions", 
            "GROQ_API_KEY", model_str, t, p
        )
    elif model_str.startswith("deepseek/"):
        return lambda t, p: generic_openai_callback(
            "https://api.deepseek.com/chat/completions", 
            "DEEPSEEK_API_KEY", model_str, t, p
        )
    else:
        # Default to local Ollama if no prefix
        return lambda t, p: ollama_callback(f"ollama/{model_str}", t, p)
