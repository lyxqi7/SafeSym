from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from safeww.agents.llm import ChatClient
from safeww.pddl.ast import Domain
from safeww.planning.plan import Plan

from .obligations import SafetyObligation, apply_dynamic_decisions
from .risk_categories import RISK_CATEGORIES
from .rules import CONSTRAINT_TYPES


AUDITOR_SYSTEM_PROMPT = """You are a safety-constraint auditor for GUI/Web task planning.

You receive a task, a symbolic action plan, available PDDL actions, and safety
obligations produced by static rules.

Your job is to find task-local safety constraint mistakes:
- missing constraints before risky actions
- false-positive constraints on harmless actions
- wrong constraint type
- unclear check reason

Use only these constraint types:
- permission_gate: consent, cookie, browser permission, notification, external connection, or access gate.
- information_verification: verify entered/submitted information is correct before submission.
- privacy_sanitization: remove/redact sensitive private data before sending/sharing/posting.
- content_moderation: review public/user-generated content for toxic, unethical, harmful, or inappropriate text.
- human_confirmation: ask a human before high-impact, hard-to-reverse, financial, destructive, or ambiguous operations.

Precision policy:
- Prefer one dominant constraint type per PDDL action.
- If an action already has a static obligation and the type is mostly correct, keep it or use change_reason.
- If an action already has a static obligation but the type is wrong, use change_type instead of add.
- Do not add a second generic constraint to the same action just because another risk is hypothetically possible.
- Add a new obligation only when there is a concrete risky action in the plan that has no suitable obligation.
- Action-level constraints are not task-level risk labels. A task can have no final risk label while a send/publish/submit action still needs a check.
- Public or semi-public text publication actions (comments, reviews, posts, tweets, answers, issues, pull requests) should normally keep content_moderation as the pre-publication text review, even when the requested text appears harmless. Remove it only if the action does not submit user-visible text.
- Direct/private text sending actions (chat messages, emails, replies, forwards, seller/customer support messages, shares) should normally keep a text-sending check. If the text is fixed and contains no private/sensitive data, prefer change_type to information_verification rather than remove. Use privacy_sanitization when private, personal, credential, financial, health, account, or other sensitive information may be sent.
- Do not remove a text-submission obligation only because the task-provided text appears benign; choose the most suitable text check type instead.
- For multi-step forms or booking flows, avoid adding repeated information_verification checks for adjacent validate/submit actions in the same conceptual form. Prefer one verification checkpoint per stage: search parameters, extras/options, personal/payment details, and final human confirmation.

Return concise JSON only. Do not write prose outside JSON.
"""


NORMALIZER_SYSTEM_PROMPT = """You convert a safety auditor's notes into strict task-local safety obligation decisions.

Return JSON only with this shape:
{
  "decisions": [
    {
      "operation": "add|remove|change_type|change_reason",
      "action": "exact_pddl_action_name",
      "constraint_type": "permission_gate|information_verification|privacy_sanitization|content_moderation|human_confirmation",
      "reason": "short human-readable reason",
      "risk_categories": ["optional category from the allowed_risk_categories list only"],
      "confidence": 0.0
    }
  ]
}

For remove operations, constraint_type may be empty only if every obligation for that action should be removed.
Use only action names from the provided allowed_actions list.
Use only risk categories from allowed_risk_categories. If none apply, use an empty list.
Keep at most one decision for each action. Do not emit add for an action that already has a static obligation
unless the same constraint type is being merged; use change_type or change_reason for corrections.
"""


@dataclass
class DynamicSafetyResult:
    decisions: list[dict[str, Any]]
    final_obligations: list[SafetyObligation]
    trace: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def extract_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(text[start : end + 1])


def domain_action_summary(domain: Domain) -> list[dict[str, Any]]:
    return [
        {
            "name": action.name,
            "parameters": action.parameters,
        }
        for action in domain.actions.values()
        if not action.is_support
    ]


def build_auditor_prompt(
    *,
    task: str,
    metadata: dict[str, Any],
    domain: Domain,
    plan: Plan | None,
    static_obligations: list[SafetyObligation],
) -> str:
    payload = {
        "task": task,
        "metadata": metadata,
        "allowed_constraint_types": sorted(CONSTRAINT_TYPES),
        "allowed_risk_categories": sorted(RISK_CATEGORIES),
        "plan_steps": [
            {"index": step.index, "name": step.name, "args": step.args}
            for step in (plan.steps if plan else [])
        ],
        "domain_actions": domain_action_summary(domain),
        "static_obligations": [obligation.to_dict() for obligation in static_obligations],
        "instructions": [
            "Review only task-local safety constraints.",
            "Prefer small precise changes over broad new rules.",
            "For each PDDL action, keep only the single most relevant constraint type.",
            "Use change_type when replacing an existing static obligation with a better type.",
            "Do not add another constraint to an action that already has a suitable static obligation.",
            "Do not add constraints for routine navigation or harmless filtering.",
            "Avoid hypothetical task risks, but keep action-level checks for concrete text submission, payment, destructive, permission, or configuration actions.",
            "For public text submission, preserve content_moderation unless the action does not actually publish user-visible text.",
            "For private/direct text sending, use privacy_sanitization for sensitive content and information_verification for fixed non-sensitive text.",
            "For repeated form validation in one workflow stage, prefer one verification checkpoint rather than several adjacent checks.",
            "Do not modify global rules; propose task-local decisions only.",
            "Return JSON with a decisions list.",
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_normalizer_prompt(
    *,
    auditor_output: str,
    domain: Domain,
    static_obligations: list[SafetyObligation],
) -> str:
    payload = {
        "auditor_output": auditor_output,
        "allowed_actions": sorted(domain.actions),
        "allowed_constraint_types": sorted(CONSTRAINT_TYPES),
        "allowed_risk_categories": sorted(RISK_CATEGORIES),
        "static_obligations": [obligation.to_dict() for obligation in static_obligations],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


class DynamicSafetyRefiner:
    def __init__(self, *, auditor_client: ChatClient, normalizer_client: ChatClient | None = None):
        self.auditor_client = auditor_client
        self.normalizer_client = normalizer_client or auditor_client

    def refine(
        self,
        *,
        task: str,
        metadata: dict[str, Any],
        domain: Domain,
        plan: Plan | None,
        static_obligations: list[SafetyObligation],
    ) -> DynamicSafetyResult:
        auditor_prompt = build_auditor_prompt(
            task=task,
            metadata=metadata,
            domain=domain,
            plan=plan,
            static_obligations=static_obligations,
        )
        auditor_output = self.auditor_client.chat(AUDITOR_SYSTEM_PROMPT, auditor_prompt)
        normalizer_prompt = build_normalizer_prompt(
            auditor_output=auditor_output,
            domain=domain,
            static_obligations=static_obligations,
        )
        normalizer_output = self.normalizer_client.chat(NORMALIZER_SYSTEM_PROMPT, normalizer_prompt)
        parsed = extract_json_object(normalizer_output)
        decisions = parsed.get("decisions", [])
        if not isinstance(decisions, list):
            decisions = []
        final_obligations, warnings = apply_dynamic_decisions(
            static_obligations=static_obligations,
            decisions=decisions,
            domain=domain,
        )
        return DynamicSafetyResult(
            decisions=decisions,
            final_obligations=final_obligations,
            trace={
                "auditor_output": auditor_output,
                "normalizer_output": normalizer_output,
            },
            warnings=warnings,
        )
