from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import LlmConfig


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 60.0

    def to_llm_config(self) -> LlmConfig:
        return LlmConfig(
            model=self.model,
            api_key_env=self.api_key_env,
            base_url_env=self.base_url_env,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_seconds=self.timeout_seconds,
        )


def model_spec_from_dict(data: dict[str, Any]) -> ModelSpec:
    model = str(data["model"])
    return ModelSpec(
        name=str(data.get("name") or model),
        model=model,
        api_key_env=str(data.get("api_key_env") or "OPENAI_API_KEY"),
        base_url_env=str(data.get("base_url_env") or "OPENAI_BASE_URL"),
        api_key=data.get("api_key"),
        base_url=data.get("base_url"),
        timeout_seconds=float(data.get("timeout_seconds", 60.0)),
    )


def load_model_specs(path: Path | str, names: list[str] | None = None) -> list[ModelSpec]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("models", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError("Model config must be a JSON object with a 'models' list or a JSON list.")

    specs = [model_spec_from_dict(entry) for entry in entries]
    if not names:
        return specs

    wanted = set(names)
    selected = [spec for spec in specs if spec.name in wanted or spec.model in wanted]
    missing = wanted - {spec.name for spec in selected} - {spec.model for spec in selected}
    if missing:
        raise ValueError(f"Models not found in config: {sorted(missing)}")
    return selected


def default_model_spec(model: str) -> ModelSpec:
    return ModelSpec(name=model, model=model)
