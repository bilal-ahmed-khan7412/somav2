"""
SOMA V2 — LLM Connectors
=========================
Provides a unified async LLM callback using the official openai Python SDK.

Supported model prefixes:
  openai/gpt-4o-mini      → OpenAI (requires OPENAI_API_KEY)
  groq/llama3-8b-8192    → Groq  (requires GROQ_API_KEY)
  deepseek/deepseek-chat → DeepSeek (requires DEEPSEEK_API_KEY)
  ollama/qwen2.5:3b      → local Ollama via httpx (no key needed)

All async callbacks share the same signature:
  async (task_type: str, prompt: str) -> str

JSON mode is enabled for planning calls (task_type contains "plan").
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Awaitable, Callable

import httpx

logger = logging.getLogger("SOMA_V2.CONNECTORS")

# Global semaphore — prevents flooding any single LLM backend
LLM_SEMAPHORE = asyncio.Semaphore(5)

# ------------------------------------------------------------------
# OpenAI-SDK-based connector (OpenAI, Groq, DeepSeek)
# ------------------------------------------------------------------

def _make_openai_callback(
    model: str,
    api_key_env: str,
    base_url: str | None = None,
) -> Callable[[str, str], Awaitable[str]]:
    """
    Returns an async callback backed by the official openai SDK.

    Works for any OpenAI-compatible API (Groq, DeepSeek) by supplying
    a custom base_url.  JSON mode is requested when task_type contains
    the word 'plan'.
    """
    from openai import AsyncOpenAI

    async def _call(task_type: str, prompt: str) -> str:
        async with LLM_SEMAPHORE:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(
                    f"Environment variable '{api_key_env}' is not set. "
                    f"Export it before running SOMA."
                )

            kwargs: dict = {}
            if base_url:
                kwargs["base_url"] = base_url

            client = AsyncOpenAI(api_key=api_key, **kwargs)

            use_json_mode = "plan" in task_type.lower()
            messages = [{"role": "user", "content": prompt}]

            # Append a nudge when JSON is required — some models need it
            if use_json_mode and "json" not in prompt.lower():
                messages.append(
                    {"role": "user", "content": "Respond ONLY with valid JSON."}
                )

            create_kwargs: dict = {
                "model": model.split("/")[-1],
                "messages": messages,
                "temperature": 0.1,
            }
            if use_json_mode:
                create_kwargs["response_format"] = {"type": "json_object"}

            try:
                response = await client.chat.completions.create(**create_kwargs)
                content = response.choices[0].message.content or ""
                logger.debug(
                    f"LLM [{model}] task_type={task_type} "
                    f"tokens_used={response.usage.total_tokens if response.usage else 'n/a'}"
                )
                return content
            except Exception as exc:
                logger.error(f"LLM call failed [{model}]: {exc}")
                raise

    return _call


# ------------------------------------------------------------------
# Ollama connector (local, no SDK, plain httpx)
# ------------------------------------------------------------------

async def _ollama_call(model: str, task_type: str, prompt: str) -> str:
    """Calls a locally-running Ollama instance via httpx."""
    async with LLM_SEMAPHORE:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model.split("/")[-1],
            "prompt": prompt,
            "stream": False,
            "format": "json" if "plan" in task_type.lower() else "",
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "")
        except Exception as exc:
            logger.error(f"Ollama [{model}] error: {exc}")
            raise


# ------------------------------------------------------------------
# Public factory
# ------------------------------------------------------------------

def get_llm_callback(model_str: str) -> Callable[[str, str], Awaitable[str]]:
    """
    Returns an async LLM callback for the given model string.

    Model string format: '<provider>/<model-name>'
      openai/gpt-4o-mini       → OpenAI ChatCompletion
      openai/gpt-4o            → OpenAI ChatCompletion
      groq/llama3-8b-8192      → Groq OpenAI-compatible API
      deepseek/deepseek-chat   → DeepSeek OpenAI-compatible API
      ollama/qwen2.5:3b        → local Ollama (httpx)
      <anything else>          → treated as Ollama local model
    """
    if model_str.startswith("openai/"):
        return _make_openai_callback(model_str, api_key_env="OPENAI_API_KEY")

    if model_str.startswith("groq/"):
        return _make_openai_callback(
            model_str,
            api_key_env="GROQ_API_KEY",
            base_url="https://api.groq.com/openai/v1",
        )

    if model_str.startswith("deepseek/"):
        return _make_openai_callback(
            model_str,
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
        )

    if model_str.startswith("ollama/") or "/" not in model_str:
        _m = model_str if model_str.startswith("ollama/") else f"ollama/{model_str}"
        return lambda t, p: _ollama_call(_m, t, p)

    raise ValueError(
        f"Unsupported model prefix in '{model_str}'. "
        "Use openai/, groq/, deepseek/, or ollama/."
    )
