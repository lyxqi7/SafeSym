from __future__ import annotations

import json
from typing import Any

from .fsm_conformance import build_fsm_symbol_contract


GENERATOR_SYSTEM_PROMPT = """You are a careful PDDL world-model generator for GUI/web tasks.

You receive a GUI/Web FSM JSON and one task instruction.
Generate a task-specific symbolic world model:
1. domain.pddl
2. problem.pddl

Hard requirements:
- Output ONLY the requested sections: ===DOMAIN=== and ===PROBLEM===.
- Use valid PDDL with :strips, :typing, and :negative-preconditions.
- Keep the model task-specific: include only the FSM pages/actions/predicates needed for the task.
- Use a page type and an (at ?p - page) predicate for GUI page tracking.
- Use ONLY the normalized FSM page symbols from the provided symbol contract.
- Use ONLY the normalized FSM action symbols from the provided symbol contract.
- Do not invent, merge, rename, or paraphrase actions. If the FSM action id is ACT_HOME_ACCEPT_COOKIES, the PDDL action name must be act_home_accept_cookies.
- Do not remove the act_ prefix from FSM action ids.
- Do not use free page parameters such as ?from ?to ?page for navigation. Every action must use fixed page constants from the symbol contract.
- Convert relevant FSM preconditions/effects into predicates.
- Every action must include its FSM source page as an (at source_page) precondition.
- Navigation actions must remove the FSM source page and add the FSM target page.
- Use stable lower_snake_case PDDL symbols.
- The problem :domain must exactly match the generated domain name.
- The problem init should include at least the FSM initial page.
- The goal should represent task completion, normally reaching a terminal/success page.
- Do not use predicates in actions/problem unless they are declared.
- Avoid numeric/object parameters unless absolutely necessary; prefer boolean abstraction for GUI state such as field_entered, item_selected, payment_ready.
- The generated model should be solvable by Fast Downward.

If feedback is provided, repair the previous model according to the feedback.
"""


VERIFIER_SYSTEM_PROMPT = """You are a PDDL verifier and repair critic.

You receive a generated PDDL model, task, and deterministic parser/planner feedback.
Your job is not to rewrite the whole model. Instead, explain what the generator should change.

Return concise, actionable feedback only.
Focus on:
- missing predicates/actions/pages
- mismatched domain/problem names
- unreachable goals
- missing precondition achievers
- wrong or over-constrained initial state
- action effects that should set state predicates
- terminal goal selection mistakes
"""


def compact_fsm_for_prompt(fsm: dict[str, Any], *, max_actions: int | None = None) -> dict[str, Any]:
    pages = []
    action_count = 0
    for page in fsm.get("pages", []) or []:
        actions = []
        for action in page.get("actions", []) or []:
            if max_actions is not None and action_count >= max_actions:
                break
            actions.append(
                {
                    "id": action.get("id"),
                    "name": action.get("name"),
                    "from": action.get("from"),
                    "to": action.get("to"),
                    "is_navigation": action.get("is_navigation"),
                    "parameters": action.get("parameters", {}),
                    "preconditions": action.get("preconditions", []),
                    "effects": action.get("effects", []),
                }
            )
            action_count += 1
        pages.append(
            {
                "id": page.get("id"),
                "signature_schema": page.get("signature_schema", {}),
                "actions": actions,
            }
        )
    return {
        "meta": fsm.get("meta", {}),
        "pages": pages,
        "truncated": max_actions is not None and action_count >= max_actions,
        "included_action_count": action_count,
    }


def symbol_contract_for_prompt(fsm: dict[str, Any]) -> dict[str, Any]:
    contract = build_fsm_symbol_contract(fsm)
    actions = [
        {
            "raw_id": action.raw_id,
            "pddl_name": action.name,
            "from_page": action.from_page,
            "to_page": action.to_page,
        }
        for action in contract.actions_by_name.values()
    ]
    return {
        "initial_page": contract.initial_page,
        "terminal_pages": sorted(contract.terminal_pages),
        "pages": [
            {"raw_id": raw_id, "pddl_name": name}
            for raw_id, name in sorted(contract.pages_by_raw_id.items())
        ],
        "actions": actions,
        "rules": [
            "PDDL actions must use pddl_name exactly.",
            "PDDL page constants must use pddl_name exactly.",
            "Do not use raw uppercase FSM ids directly in PDDL.",
            "Do not use page parameters for navigation; use fixed page constants.",
        ],
    }


def build_generator_prompt(
    *,
    fsm: dict[str, Any],
    task: str,
    feedback: str | None = None,
    previous_domain: str | None = None,
    previous_problem: str | None = None,
    max_actions: int | None = None,
) -> str:
    payload = {
        "task": task,
        "symbol_contract": symbol_contract_for_prompt(fsm),
        "fsm": compact_fsm_for_prompt(fsm, max_actions=max_actions),
    }
    parts = [
        "Generate domain.pddl and problem.pddl for this GUI task.",
        "Input JSON:",
        json.dumps(payload, indent=2, ensure_ascii=False),
    ]
    if previous_domain and previous_problem:
        parts.extend(
            [
                "Previous domain.pddl:",
                previous_domain,
                "Previous problem.pddl:",
                previous_problem,
            ]
        )
    if feedback:
        parts.extend(["Repair feedback:", feedback])
    parts.append(
        "Return exactly:\n===DOMAIN===\n<domain.pddl>\n\n===PROBLEM===\n<problem.pddl>"
    )
    return "\n\n".join(parts)


def build_verifier_prompt(
    *,
    task: str,
    domain: str,
    problem: str,
    validation_feedback: str,
    planner_feedback: str,
) -> str:
    return "\n\n".join(
        [
            f"TASK:\n{task}",
            f"DOMAIN:\n{domain}",
            f"PROBLEM:\n{problem}",
            f"LOCAL VALIDATION FEEDBACK:\n{validation_feedback}",
            f"PLANNER FEEDBACK:\n{planner_feedback}",
            "Give concise repair instructions for the generator.",
        ]
    )
