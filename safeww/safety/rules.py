from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CONSTRAINT_TYPES = {
    "permission_gate",
    "information_verification",
    "privacy_sanitization",
    "content_moderation",
    "human_confirmation",
}


def infer_constraint_type(value: str) -> str:
    normalized = value.lower()
    if "privacy" in normalized or "disclosure" in normalized or "data_leak" in normalized:
        return "privacy_sanitization"
    if "toxic" in normalized or "ethical" in normalized or "content" in normalized:
        return "content_moderation"
    if "network" in normalized or "trust" in normalized or "permission" in normalized or "location" in normalized:
        return "permission_gate"
    if "financial" in normalized or "transaction" in normalized or "payment" in normalized:
        return "human_confirmation"
    if "human_confirmation" in normalized:
        return "human_confirmation"
    return "information_verification"


@dataclass(frozen=True)
class SafetyRule:
    name: str
    raw: dict[str, Any]

    @property
    def match_config(self) -> dict[str, Any]:
        return self.raw.get("match", {})

    @property
    def inject_config(self) -> dict[str, Any]:
        return self.raw.get("inject", {})

    @property
    def support_action(self) -> dict[str, Any] | None:
        return self.raw.get("support_action")

    @property
    def problem_update(self) -> dict[str, Any]:
        return self.raw.get("problem_update", {})

    @property
    def risk_category(self) -> str:
        return self.risk_categories[0] if self.risk_categories else self.name

    @property
    def risk_categories(self) -> list[str]:
        categories = self.raw.get("risk_categories")
        if isinstance(categories, list):
            return [str(category) for category in categories]
        category = self.raw.get("risk_category")
        return [str(category)] if category else []

    @property
    def constraint_type(self) -> str:
        constraint_type = self.raw.get("constraint_type")
        if constraint_type:
            return str(constraint_type)
        support_name = (self.support_action or {}).get("name", "")
        lowered = support_name.lower()
        for candidate in CONSTRAINT_TYPES:
            if lowered.startswith(f"check_{candidate}_"):
                return candidate
        return infer_constraint_type(support_name or self.risk_category)

    @property
    def check_reason(self) -> str:
        return self.raw.get("check_reason", self.name.replace("_", " "))


def load_rules(data: list[dict[str, Any]]) -> list[SafetyRule]:
    rules = [SafetyRule(item["name"], item) for item in data]
    for rule in rules:
        if "constraint_type" in rule.raw and rule.constraint_type not in CONSTRAINT_TYPES:
            allowed = ", ".join(sorted(CONSTRAINT_TYPES))
            raise ValueError(f"Unknown constraint_type for rule {rule.name}: {rule.constraint_type}. Allowed: {allowed}")
    return rules
