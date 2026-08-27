from __future__ import annotations

import time

from openai import OpenAI


class LLMClient:
    """Thin wrapper over the OpenAI-compatible chat completions API.

    DeepSeek exposes an OpenAI-compatible endpoint, so we use the official
    `openai` client — a model vendor's API client (explicitly allowed), not an
    agent framework. Retries with simple exponential backoff cover transient
    network / rate-limit / 5xx failures.
    """

    def __init__(self, api_key: str, base_url: str, model: str, temperature: float, max_retries: int):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """Return the assistant message object (may carry tool_calls)."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools or None,
                    temperature=self.temperature,
                )
                return resp.choices[0].message
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM 调用失败（重试 {self.max_retries} 次后）: {last_exc}")
