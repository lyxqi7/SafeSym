from __future__ import annotations

import os
import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image


class RetryableLlmResponseError(RuntimeError):
    pass


@dataclass(frozen=True)
class LlmConfig:
    model: str = "gpt-5.4-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 60.0


class ChatClient:
    """Small OpenAI-compatible wrapper with credentials read from env vars."""

    def __init__(self, config: LlmConfig | None = None):
        from openai import OpenAI

        self.config = config or LlmConfig()
        api_key = self.config.api_key or os.environ.get(self.config.api_key_env)
        base_url = self.config.base_url if self.config.base_url is not None else os.environ.get(self.config.base_url_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {self.config.api_key_env}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, system_prompt: str, user_content: Any) -> str:
        attempt = 1
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    timeout=self.config.timeout_seconds,
                )
                content = response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise RetryableLlmResponseError("LLM response message content is empty.")
                return content.strip()
            except Exception as exc:
                if not is_retryable_llm_error(exc):
                    raise
                print(
                    "[LLM retry] "
                    f"model={self.config.model} attempt={attempt} "
                    f"error={compact_retry_error(exc)}",
                    flush=True,
                )
                attempt += 1


def is_retryable_llm_error(exc: Exception) -> bool:
    if isinstance(exc, RetryableLlmResponseError):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429}:
        return True
    if isinstance(status_code, int) and status_code >= 500:
        return True

    name = type(exc).__name__.lower()
    retryable_names = [
        "ratelimit",
        "timeout",
        "connection",
        "apierror",
        "internalserver",
        "serviceunavailable",
    ]
    if any(token in name for token in retryable_names):
        return True

    message = str(exc).lower()
    retryable_messages = [
        "rate limit",
        "429",
        "llm returned empty response",
        "response message content is empty",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "upstream",
        "负载",
        "稍后再试",
    ]
    return any(token in message for token in retryable_messages)


def compact_retry_error(exc: Exception, limit: int = 240) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def multimodal_user_content(text: str, image: Image.Image | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if image is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(image)}"},
            }
        )
    return content
