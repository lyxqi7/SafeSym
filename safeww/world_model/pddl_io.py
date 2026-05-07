from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from safeww.pddl.ast import Expr
from safeww.pddl.parser import parse_domain, parse_problem, parse_sexp, tokenize

from .fsm_conformance import validate_fsm_conformance


@dataclass
class PddlValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    domain_name: str | None = None
    problem_name: str | None = None
    action_count: int = 0
    predicate_count: int = 0

    def to_feedback(self) -> str:
        lines = []
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"- {error}" for error in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        if not lines:
            lines.append("No syntax or consistency errors found.")
        return "\n".join(lines)


def extract_pddl_sections(text: str) -> tuple[str, str]:
    domain_match = re.search(r"===DOMAIN===\s*(.*?)\s*===PROBLEM===", text, re.S | re.I)
    problem_match = re.search(r"===PROBLEM===\s*(.*)", text, re.S | re.I)
    if not domain_match or not problem_match:
        # Fallback for markdown fenced blocks with labels.
        domain_match = re.search(r"```(?:pddl)?\s*(\(define\s*\(domain.*?\))\s*```", text, re.S | re.I)
        problem_match = re.search(r"```(?:pddl)?\s*(\(define\s*\(problem.*?\))\s*```", text, re.S | re.I)
    if not domain_match or not problem_match:
        raise ValueError("Could not find ===DOMAIN=== and ===PROBLEM=== sections.")
    return domain_match.group(1).strip(), problem_match.group(1).strip()


def atom_names(expr: Expr | None) -> Iterable[str]:
    if expr is None:
        return
    if expr.op == "atom":
        if expr.args:
            yield str(expr.args[0])
        return
    for arg in expr.args:
        if isinstance(arg, Expr):
            yield from atom_names(arg)


def problem_domain_name(problem_text: str) -> str | None:
    sexp = parse_sexp(tokenize(problem_text))
    for item in sexp:
        if isinstance(item, list) and item and item[0] == ":domain":
            return str(item[1]) if len(item) > 1 else None
    return None


def validate_pddl_pair(
    domain_text: str,
    problem_text: str,
    *,
    fsm: dict | None = None,
) -> PddlValidationReport:
    report = PddlValidationReport(ok=False)
    try:
        domain = parse_domain(domain_text)
    except Exception as exc:
        report.errors.append(f"domain parse failed: {exc}")
        return report
    try:
        problem = parse_problem(problem_text)
    except Exception as exc:
        report.errors.append(f"problem parse failed: {exc}")
        return report

    report.domain_name = domain.name
    report.problem_name = problem.name
    report.action_count = len(domain.actions)
    report.predicate_count = len(domain.predicates)

    if not domain.name:
        report.errors.append("domain name is missing")
    if not problem.name:
        report.errors.append("problem name is missing")
    if domain.name == "domain":
        report.warnings.append("domain name is the parser default; check the domain header")
    problem_domain = problem_domain_name(problem_text)
    if problem_domain != domain.name:
        report.errors.append(
            f"problem :domain '{problem_domain}' does not match generated domain '{domain.name}'"
        )
    if not domain.actions:
        report.errors.append("domain contains no actions")
    if not domain.predicates:
        report.errors.append("domain contains no predicates")
    if problem.goal is None:
        report.errors.append("problem goal is missing")
    if problem.init is None:
        report.errors.append("problem init is missing")

    declared_predicates = {predicate.name for predicate in domain.predicates}
    for action in domain.actions.values():
        for name in atom_names(action.precondition):
            if name not in declared_predicates:
                report.errors.append(f"action {action.name} uses undeclared precondition predicate: {name}")
        for name in atom_names(action.effect):
            if name not in declared_predicates:
                report.errors.append(f"action {action.name} uses undeclared effect predicate: {name}")

    for name in atom_names(problem.init):
        if name not in declared_predicates:
            report.errors.append(f"problem init uses undeclared predicate: {name}")
    for name in atom_names(problem.goal):
        if name not in declared_predicates:
            report.errors.append(f"problem goal uses undeclared predicate: {name}")

    page_constants = set(domain.constants.get("page", []))
    problem_pages = set(problem.objects.get("page", []))
    if page_constants and problem_pages and not problem_pages.issubset(page_constants):
        missing = sorted(problem_pages - page_constants)
        report.errors.append(f"problem page objects are not domain page constants: {missing}")

    if ":negative-preconditions" not in domain.requirements:
        report.warnings.append("domain does not declare :negative-preconditions")
    if ":strips" not in domain.requirements:
        report.warnings.append("domain does not declare :strips")

    if fsm is not None:
        fsm_report = validate_fsm_conformance(domain=domain, problem=problem, fsm=fsm)
        report.errors.extend(fsm_report.errors)
        report.warnings.extend(fsm_report.warnings)

    report.ok = not report.errors
    return report
