from __future__ import annotations

import re
from typing import Any

from .ast import Action, Domain, Expr, Predicate, Problem


def tokenize(pddl_text: str) -> list[str]:
    pddl_text = re.sub(r";.*", "", pddl_text)
    return re.findall(r"\(|\)|[^\s()]+", pddl_text)


def parse_sexp(tokens: list[str]) -> Any:
    if not tokens:
        raise ValueError("Unexpected EOF while parsing PDDL")

    token = tokens.pop(0)
    if token == "(":
        items = []
        while tokens and tokens[0] != ")":
            items.append(parse_sexp(tokens))
        if not tokens:
            raise ValueError("Missing closing parenthesis")
        tokens.pop(0)
        return items
    if token == ")":
        raise ValueError("Unexpected closing parenthesis")
    return token


def build_expr(sexp: Any) -> Expr | None:
    if isinstance(sexp, str):
        return Expr("atom", [sexp])
    if not sexp:
        return None

    op = sexp[0]
    if op == "and":
        return Expr("and", [expr for item in sexp[1:] if (expr := build_expr(item))])
    if op == "not":
        inner = build_expr(sexp[1])
        if inner is None:
            raise ValueError("Invalid empty not expression")
        return Expr("not", [inner])
    return Expr("atom", list(sexp))


def parse_typed_symbols(items: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    typed: dict[str, list[str]] = {}
    raw: list[str] = []
    if "-" not in items:
        return typed, list(items)

    i = 0
    while i < len(items):
        names = []
        while i < len(items) and items[i] != "-":
            names.append(items[i])
            i += 1
        if i >= len(items):
            raw.extend(names)
            break
        i += 1
        if i >= len(items):
            raise ValueError("Missing type after '-'")
        typ = items[i]
        i += 1
        typed.setdefault(typ, []).extend(names)
    return typed, raw


def parse_parameters(items: list[str]) -> list[tuple[str, str | None]]:
    params: list[tuple[str, str | None]] = []
    i = 0
    while i < len(items):
        var = items[i]
        if i + 2 < len(items) and items[i + 1] == "-":
            params.append((var, items[i + 2]))
            i += 3
        else:
            params.append((var, None))
            i += 1
    return params


def parse_domain(pddl_text: str) -> Domain:
    sexp = parse_sexp(tokenize(pddl_text))
    domain = Domain()

    for item in sexp:
        if not isinstance(item, list) or not item:
            continue
        if item[0] == "domain":
            domain.name = item[1]
        elif item[0] == ":requirements":
            domain.requirements.update(item[1:])
        elif item[0] == ":types":
            domain.types.extend(item[1:])
        elif item[0] == ":constants":
            domain.constants, domain.constants_raw = parse_typed_symbols(item[1:])
        elif item[0] == ":predicates":
            for pred in item[1:]:
                domain.predicates.append(Predicate(pred[0], parse_parameters(pred[1:])))
        elif item[0] == ":action":
            action = parse_action(item)
            domain.actions[action.name] = action

    return domain


def parse_action(action_sexp: list[Any]) -> Action:
    action = Action(action_sexp[1])
    i = 2
    while i < len(action_sexp):
        key = action_sexp[i]
        if key == ":parameters":
            action.parameters = parse_parameters(action_sexp[i + 1])
            i += 2
        elif key == ":precondition":
            action.precondition = build_expr(action_sexp[i + 1])
            i += 2
        elif key == ":effect":
            action.effect = build_expr(action_sexp[i + 1])
            i += 2
        else:
            i += 1
    return action


def flatten(items: list[Any]) -> list[str]:
    flat: list[str] = []
    for item in items:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat


def parse_problem(pddl_text: str) -> Problem:
    sexp = parse_sexp(tokenize(pddl_text))
    problem = Problem()

    if sexp[0] == "define" and isinstance(sexp[1], list) and sexp[1][0] == "problem":
        problem.name = sexp[1][1]

    for item in sexp:
        if not isinstance(item, list) or not item:
            continue
        if item[0] == ":objects":
            problem.objects, problem.objects_raw = parse_typed_symbols(flatten(item[1:]))
        elif item[0] == ":init":
            problem.init = build_expr(["and"] + item[1:])
        elif item[0] == ":goal":
            problem.goal = build_expr(item[1])

    return problem

