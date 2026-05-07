from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from safeww.pddl.ast import Action, Domain, Expr, Problem

from .naming import normalize_symbol, unique_symbol


@dataclass(frozen=True)
class FsmActionSymbol:
    raw_id: str
    name: str
    from_page: str | None
    to_page: str | None


@dataclass(frozen=True)
class FsmSymbolContract:
    initial_page: str | None
    terminal_pages: set[str]
    pages_by_raw_id: dict[str, str]
    actions_by_name: dict[str, FsmActionSymbol]


@dataclass
class FsmConformanceReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_feedback(self) -> str:
        lines: list[str] = []
        if self.errors:
            lines.append("FSM conformance errors:")
            lines.extend(f"- {error}" for error in self.errors)
        if self.warnings:
            lines.append("FSM conformance warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        if not lines:
            lines.append("Generated PDDL conforms to the FSM symbol/transition contract.")
        return "\n".join(lines)


def build_fsm_symbol_contract(fsm: dict[str, Any]) -> FsmSymbolContract:
    pages_by_raw_id: dict[str, str] = {}
    used_pages: set[str] = set()
    for page in fsm.get("pages", []) or []:
        raw_id = str(page.get("id") or "")
        if raw_id:
            pages_by_raw_id[raw_id] = unique_symbol(
                normalize_symbol(raw_id, prefix="page"), used_pages
            )

    used_actions: set[str] = set()
    actions_by_name: dict[str, FsmActionSymbol] = {}
    for page in fsm.get("pages", []) or []:
        for raw_action in page.get("actions", []) or []:
            raw_id = str(raw_action.get("id") or raw_action.get("name") or "action")
            name = unique_symbol(normalize_symbol(raw_id, prefix="action"), used_actions)
            from_page = pages_by_raw_id.get(str(raw_action.get("from") or ""))
            to_page = pages_by_raw_id.get(str(raw_action.get("to") or ""))
            actions_by_name[name] = FsmActionSymbol(
                raw_id=raw_id,
                name=name,
                from_page=from_page,
                to_page=to_page,
            )

    meta = fsm.get("meta", {}) or {}
    initial_page = pages_by_raw_id.get(str(meta.get("initial_page_id") or ""))
    terminal_pages = {
        pages_by_raw_id[raw_id]
        for raw_id in (str(page_id) for page_id in meta.get("terminal_pages", []) or [])
        if raw_id in pages_by_raw_id
    }
    return FsmSymbolContract(
        initial_page=initial_page,
        terminal_pages=terminal_pages,
        pages_by_raw_id=pages_by_raw_id,
        actions_by_name=actions_by_name,
    )


def atom_tuples(expr: Expr | None, *, negated: bool = False) -> list[tuple[bool, tuple[str, ...]]]:
    if expr is None:
        return []
    if expr.op == "atom":
        return [(negated, tuple(str(arg) for arg in expr.args))]
    if expr.op == "not" and expr.args and isinstance(expr.args[0], Expr):
        return atom_tuples(expr.args[0], negated=not negated)
    atoms: list[tuple[bool, tuple[str, ...]]] = []
    for arg in expr.args:
        if isinstance(arg, Expr):
            atoms.extend(atom_tuples(arg, negated=negated))
    return atoms


def has_atom(expr: Expr | None, *items: str, negated: bool = False) -> bool:
    target = tuple(items)
    return any(is_negated == negated and atom == target for is_negated, atom in atom_tuples(expr))


def page_parameters(action: Action) -> list[str]:
    return [name for name, typ in action.parameters if typ == "page"]


def domain_page_constants(domain: Domain) -> set[str]:
    return set(domain.constants.get("page", []))


def problem_page_objects(problem: Problem) -> set[str]:
    return set(problem.objects.get("page", []))


def validate_fsm_conformance(
    *,
    domain: Domain,
    problem: Problem,
    fsm: dict[str, Any],
) -> FsmConformanceReport:
    contract = build_fsm_symbol_contract(fsm)
    report = FsmConformanceReport(ok=False)

    expected_pages = set(contract.pages_by_raw_id.values())
    seen_pages = domain_page_constants(domain) | problem_page_objects(problem)
    invented_pages = sorted(seen_pages - expected_pages)
    if invented_pages:
        report.errors.append(f"page symbols are not from the FSM: {invented_pages}")

    if contract.initial_page and not has_atom(problem.init, "at", contract.initial_page):
        report.errors.append(
            f"problem init must include FSM initial page: (at {contract.initial_page})"
        )

    if contract.terminal_pages:
        goal_pages = {
            atom[1]
            for negated, atom in atom_tuples(problem.goal)
            if not negated and len(atom) == 2 and atom[0] == "at"
        }
        if goal_pages and not goal_pages.issubset(contract.terminal_pages):
            report.warnings.append(
                f"goal page is not one of the FSM terminal pages: {sorted(goal_pages)}"
            )

    for action in domain.actions.values():
        expected = contract.actions_by_name.get(action.name)
        if expected is None:
            report.errors.append(f"domain action is not an FSM action id: {action.name}")
            continue

        free_pages = page_parameters(action)
        if free_pages:
            report.errors.append(
                f"action {action.name} uses free page parameters {free_pages}; "
                "FSM navigation actions must use fixed page constants"
            )

        if expected.from_page and not has_atom(action.precondition, "at", expected.from_page):
            report.errors.append(
                f"action {action.name} must require its FSM source page: (at {expected.from_page})"
            )

        if expected.from_page and expected.to_page and expected.from_page != expected.to_page:
            if not has_atom(action.effect, "at", expected.from_page, negated=True):
                report.errors.append(
                    f"action {action.name} must remove its FSM source page: "
                    f"(not (at {expected.from_page}))"
                )
            if not has_atom(action.effect, "at", expected.to_page):
                report.errors.append(
                    f"action {action.name} must add its FSM target page: (at {expected.to_page})"
                )

    report.ok = not report.errors
    return report
