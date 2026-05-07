from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from safeww.pddl.ast import Domain

from .injector import match_rule
from .risk_categories import RISK_CATEGORIES
from .rules import CONSTRAINT_TYPES, SafetyRule


@dataclass
class SafetyObligation:
    action: str
    constraint_type: str
    check_reason: str
    source: str
    risk_categories: list[str] = field(default_factory=list)
    rule_names: list[str] = field(default_factory=list)
    confidence: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.action, self.constraint_type)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action": self.action,
            "constraint_type": self.constraint_type,
            "check_reason": self.check_reason,
            "source": self.source,
            "risk_categories": sorted(set(self.risk_categories)),
            "rule_names": self.rule_names,
        }
        if self.confidence is not None:
            data["confidence"] = self.confidence
        return data


def merge_obligation(existing: SafetyObligation, incoming: SafetyObligation) -> SafetyObligation:
    existing.risk_categories = sorted(set(existing.risk_categories) | set(incoming.risk_categories))
    existing.rule_names = sorted(set(existing.rule_names) | set(incoming.rule_names))
    if existing.source != incoming.source and incoming.source not in existing.source.split("+"):
        existing.source = f"{existing.source}+{incoming.source}"
    if not existing.check_reason and incoming.check_reason:
        existing.check_reason = incoming.check_reason
    return existing


def infer_static_obligations(domain: Domain, rules: list[SafetyRule]) -> list[SafetyObligation]:
    obligations: dict[tuple[str, str], SafetyObligation] = {}
    for rule in rules:
        for action in domain.actions.values():
            if not match_rule(rule, action):
                continue
            obligation = SafetyObligation(
                action=action.name,
                constraint_type=rule.constraint_type,
                check_reason=rule.check_reason,
                source="static_rule",
                risk_categories=list(rule.risk_categories),
                rule_names=[rule.name],
            )
            if obligation.key in obligations:
                merge_obligation(obligations[obligation.key], obligation)
            else:
                obligations[obligation.key] = obligation
    return sorted(obligations.values(), key=lambda item: (item.action, item.constraint_type))


def rule_dict_for_obligation(obligation: SafetyObligation) -> dict[str, Any]:
    constraint_type = obligation.constraint_type
    predicate_type = constraint_type.replace("_", "-")
    predicate = f"checked-{predicate_type}_{obligation.action}"
    return {
        "name": f"task_obligation_{constraint_type}_{obligation.action}",
        "risk_categories": obligation.risk_categories,
        "constraint_type": constraint_type,
        "check_reason": obligation.check_reason,
        "match": {"action_name_equals": [obligation.action]},
        "inject": {
            "precondition_add": [["atom", [predicate]]],
            "effect_add": [["not", ["atom", [predicate]]]],
        },
        "support_action": {"name": f"check_{constraint_type}_{obligation.action}"},
        "problem_update": {"init_add": [["not", ["atom", [predicate]]]]},
    }


def rules_from_obligations(obligations: list[SafetyObligation]) -> list[SafetyRule]:
    return [
        SafetyRule(rule_dict["name"], rule_dict)
        for obligation in obligations
        for rule_dict in [rule_dict_for_obligation(obligation)]
    ]


def normalize_decision(raw: dict[str, Any]) -> dict[str, Any] | None:
    operation = str(raw.get("operation", "")).strip().lower()
    action = str(raw.get("action", "")).strip()
    constraint_type = str(raw.get("constraint_type", "")).strip()
    if operation not in {"add", "remove", "change_type", "change_reason"}:
        return None
    if not action:
        return None
    if operation in {"add", "change_type"} and constraint_type not in CONSTRAINT_TYPES:
        return None
    reason = str(raw.get("reason") or raw.get("check_reason") or "").strip()
    categories = raw.get("risk_categories", [])
    if not isinstance(categories, list):
        categories = []
    confidence = raw.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None
    return {
        "operation": operation,
        "action": action,
        "constraint_type": constraint_type,
        "reason": reason,
        "risk_categories": [
            str(category) for category in categories if str(category) in RISK_CATEGORIES
        ],
        "confidence": confidence_value,
    }


def apply_dynamic_decisions(
    *,
    static_obligations: list[SafetyObligation],
    decisions: list[dict[str, Any]],
    domain: Domain,
) -> tuple[list[SafetyObligation], list[str]]:
    obligations = {obligation.key: obligation for obligation in static_obligations}
    warnings: list[str] = []
    valid_actions = set(domain.actions)

    for raw_decision in decisions:
        decision = normalize_decision(raw_decision)
        if decision is None:
            warnings.append(f"ignored invalid decision: {raw_decision}")
            continue
        action = decision["action"]
        if action not in valid_actions:
            warnings.append(f"ignored decision for unknown action: {action}")
            continue
        operation = decision["operation"]
        constraint_type = decision["constraint_type"]

        if operation == "remove":
            if constraint_type:
                obligations.pop((action, constraint_type), None)
            else:
                for key in [key for key in obligations if key[0] == action]:
                    obligations.pop(key, None)
            continue

        if operation == "change_type":
            for key in [key for key in obligations if key[0] == action]:
                obligations.pop(key, None)
            operation = "add"

        if operation == "change_reason":
            targets = [
                key for key in obligations if key[0] == action and (not constraint_type or key[1] == constraint_type)
            ]
            for key in targets:
                if decision["reason"]:
                    obligations[key].check_reason = decision["reason"]
                obligations[key].source = f"{obligations[key].source}+llm_refined"
            continue

        if operation == "add":
            existing_action_keys = [key for key in obligations if key[0] == action]
            if existing_action_keys and (action, constraint_type) not in obligations:
                warnings.append(
                    "ignored add for already-constrained action "
                    f"{action}; use change_type to replace the existing constraint"
                )
                continue
            obligation = SafetyObligation(
                action=action,
                constraint_type=constraint_type,
                check_reason=decision["reason"] or f"perform {constraint_type} before {action}",
                source="llm_dynamic",
                risk_categories=decision["risk_categories"],
                rule_names=[],
                confidence=decision["confidence"],
            )
            if obligation.key in obligations:
                merge_obligation(obligations[obligation.key], obligation)
            else:
                obligations[obligation.key] = obligation

    return sorted(obligations.values(), key=lambda item: (item.action, item.constraint_type)), warnings
