from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from safeww.pddl.ast import Action, Domain, Expr, Predicate, Problem
from safeww.pddl.parser import parse_domain, parse_problem

from .rules import SafetyRule, load_rules


def match_rule(rule: SafetyRule, action: Action) -> bool:
    if action.is_support:
        return False

    cfg = rule.match_config
    action_name = action.name.lower()

    for keyword in cfg.get("action_name_not_contains", []):
        if keyword.lower() in action_name:
            return False

    for pattern in cfg.get("action_name_not_regex", []):
        if re.search(pattern, action.name, re.IGNORECASE):
            return False

    for keyword in cfg.get("action_name_contains", []):
        if keyword.lower() in action_name:
            return True
    for keyword in cfg.get("action_name_equals", []):
        if keyword.lower() == action_name:
            return True
    for pattern in cfg.get("action_name_regex", []):
        if re.search(pattern, action.name, re.IGNORECASE):
            return True
    return False


def build_expr(obj: Any) -> Expr:
    kind = obj[0]
    if kind == "atom":
        args = obj[1]
        return Expr("atom", args if isinstance(args, list) else [args])
    if kind == "not":
        return Expr("not", [build_expr(obj[1])])
    if kind == "and":
        return Expr("and", [build_expr(item) for item in obj[1]])
    raise ValueError(f"Unknown rule expression type: {kind}")


def instantiate_name(value: str, action_name: str) -> str:
    return value.replace("{action}", action_name)


def instantiate_expr(expr: Expr, action_name: str) -> Expr:
    if expr.op == "atom":
        return Expr("atom", [instantiate_name(arg, action_name) for arg in expr.args])  # type: ignore[arg-type]
    if expr.op == "not":
        return Expr("not", [instantiate_expr(expr.args[0], action_name)])  # type: ignore[index]
    if expr.op == "and":
        return Expr("and", [instantiate_expr(arg, action_name) for arg in expr.args])  # type: ignore[arg-type]
    return expr


def collect_predicates(expr: Expr | None) -> set[str]:
    if expr is None:
        return set()
    if expr.op == "atom":
        return {expr.args[0]}  # type: ignore[index]
    predicates: set[str] = set()
    for sub in expr.args:  # type: ignore[union-attr]
        predicates.update(collect_predicates(sub))
    return predicates


def has_predicate(domain: Domain, name: str) -> bool:
    return any(predicate.name.lower() == name.lower() for predicate in domain.predicates)


def add_predicate(domain: Domain, name: str) -> None:
    if not has_predicate(domain, name):
        domain.predicates.append(Predicate(name, []))


def add_precondition(action: Action, expr: Expr) -> None:
    if action.precondition is None:
        action.precondition = expr
    elif action.precondition.op == "and":
        action.precondition.args.append(expr)  # type: ignore[union-attr]
    else:
        action.precondition = Expr("and", [action.precondition, expr])


def add_effect(action: Action, expr: Expr) -> None:
    if action.effect is None:
        action.effect = expr
    elif action.effect.op == "and":
        action.effect.args.append(expr)  # type: ignore[union-attr]
    else:
        action.effect = Expr("and", [action.effect, expr])


def remove_predicate(expr: Expr | None, target_name: str) -> Expr | None:
    if expr is None:
        return None
    if expr.op == "atom":
        return None if expr.args[0] == target_name else expr  # type: ignore[index]
    if expr.op == "not":
        sub = remove_predicate(expr.args[0], target_name)  # type: ignore[index]
        return None if sub is None else Expr("not", [sub])
    if expr.op == "and":
        args = [cleaned for sub in expr.args if (cleaned := remove_predicate(sub, target_name))]  # type: ignore[arg-type]
        if not args:
            return None
        return args[0] if len(args) == 1 else Expr("and", args)
    return expr


def extract_target_predicate(rule: SafetyRule, action_name: str) -> str:
    for item in rule.inject_config.get("precondition_add", []):
        expr = build_expr(item)
        if expr.op == "atom":
            name = expr.args[0]  # type: ignore[index]
            return instantiate_name(name, action_name)
    raise ValueError(f"Cannot infer target predicate for rule {rule.name}")


class SafetyCompiler:
    """Rule-based compiler that injects safety obligations into PDDL."""

    def __init__(self, rules: list[SafetyRule]):
        self.rules = rules

    @classmethod
    def from_file(cls, path: Path | str) -> "SafetyCompiler":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(load_rules(json.load(f)))

    def compile(self, domain: Domain, problem: Problem) -> tuple[Domain, Problem]:
        for rule in self.rules:
            self.apply_rule(rule, domain)
        for rule in self.rules:
            self.apply_problem_update(rule, problem, domain)
        return domain, problem

    def apply_rule(self, rule: SafetyRule, domain: Domain) -> None:
        new_actions: dict[str, Action] = {}

        for action in list(domain.actions.values()):
            if not match_rule(rule, action):
                continue

            if not hasattr(action, "precondition_before_injection"):
                action.precondition_before_injection = (
                    action.precondition.clone() if action.precondition else None
                )

            for item in rule.inject_config.get("precondition_add", []):
                expr = instantiate_expr(build_expr(item), action.name)
                add_precondition(action, expr)
                for predicate in collect_predicates(expr):
                    add_predicate(domain, predicate)

            for item in rule.inject_config.get("effect_add", []):
                expr = instantiate_expr(build_expr(item), action.name)
                add_effect(action, expr)
                for predicate in collect_predicates(expr):
                    add_predicate(domain, predicate)

            support_cfg = rule.support_action
            if support_cfg is None:
                continue

            support_name = instantiate_name(support_cfg["name"], action.name)
            if support_name in domain.actions or support_name in new_actions:
                continue

            target_predicate = extract_target_predicate(rule, action.name)
            original_precondition = getattr(action, "precondition_before_injection", None)
            base_precondition = remove_predicate(
                original_precondition.clone() if original_precondition else None,
                target_predicate,
            )
            not_confirmed = Expr("not", [Expr("atom", [target_predicate])])

            support_action = Action(
                name=support_name,
                parameters=list(action.parameters),
                precondition=not_confirmed
                if base_precondition is None
                else Expr("and", [base_precondition, not_confirmed]),
                effect=Expr("atom", [target_predicate]),
                is_support=True,
            )
            new_actions[support_name] = support_action
            add_predicate(domain, target_predicate)

        domain.actions.update(new_actions)

    def apply_problem_update(self, rule: SafetyRule, problem: Problem, domain: Domain) -> None:
        for action in domain.actions.values():
            if not match_rule(rule, action):
                continue
            for item in rule.problem_update.get("init_add", []):
                expr = instantiate_expr(build_expr(item), action.name)
                if problem.init is None:
                    problem.init = expr
                elif problem.init.op == "and":
                    problem.init.args.append(expr)  # type: ignore[union-attr]
                else:
                    problem.init = Expr("and", [problem.init, expr])


def compile_safe_pddl(
    domain_file: Path | str,
    problem_file: Path | str,
    rule_file: Path | str,
) -> tuple[Domain, Problem]:
    domain = parse_domain(Path(domain_file).read_text(encoding="utf-8"))
    problem = parse_problem(Path(problem_file).read_text(encoding="utf-8"))
    return SafetyCompiler.from_file(rule_file).compile(domain, problem)
