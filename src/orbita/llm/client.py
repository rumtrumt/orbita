"""Клиент OpenAI-совместимого API.

Этот протокол понимают Ollama, llama.cpp, vLLM, OpenAI, OpenRouter и Groq, поэтому смена
провайдера — это смена двух переменных окружения, а не кода.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

import httpx

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class LlmError(Exception):
    """Модель недоступна или вернула не то, что от неё просили."""


class LlmClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}"}
        # Системный прокси не должен перехватывать обращения к локальной Ollama.
        local = urlparse(base_url).hostname in _LOCAL_HOSTS
        self._http = httpx.Client(timeout=timeout, trust_env=not local)

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], name: str = "result"
    ) -> dict[str, Any]:
        """Запрос со структурированным ответом по JSON-схеме."""
        # Qwen3 иначе тратит секунды на рассуждения, которые здесь не нужны.
        suffix = " /no_think" if "qwen3" in self.model.lower() else ""
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user + suffix},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
        }
        try:
            response = self._http.post(self._url, json=payload, headers=self._headers)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
            raise LlmError(f"Запрос к LLM не выполнен: {error}") from error
        try:
            return dict(json.loads(_THINK_BLOCK.sub("", content).strip()))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise LlmError(f"LLM вернула не JSON: {content[:200]!r}") from error
