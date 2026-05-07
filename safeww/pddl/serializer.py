from __future__ import annotations

from .ast import Action, Domain, Expr, Predicate, Problem


def expr_to_pddl(expr: Expr | None) -> str:
    if expr is None:
        return "()"
    if expr.op == "atom":
        return f"({' '.join(expr.args)})"  # type: ignore[arg-type]
    if expr.op == "not":
        return f"(not {expr_to_pddl(expr.args[0])})"  # type: ignore[index]
    if expr.op == "and":
        return f"(and {' '.join(expr_to_pddl(arg) for arg in expr.args)})"  # type: ignore[arg-type]
    raise ValueError(f"Unknown expression op: {expr.op}")


def predicate_to_pddl(predicate: Predicate) -> str:
    if not predicate.args:
        return f"({predicate.name})"
    args = " ".join(f"{var} - {typ}" if typ else var for var, typ in predicate.args)
    return f"({predicate.name} {args})"


def typed_block_to_pddl(typed: dict[str, list[str]], raw: list[str]) -> str:
    lines = [" ".join(names) + f" - {typ}" for typ, names in typed.items() if names]
    if raw:
        lines.append(" ".join(raw))
    return "\n    ".join(lines)


def action_to_pddl(action: Action) -> str:
    params = ""
    if action.parameters:
        params_str = " ".join(
            f"{var} - {typ}" if typ else var for var, typ in action.parameters
        )
        params = f"\n    :parameters ({params_str})"

    return f"""  (:action {action.name}{params}
    :precondition {expr_to_pddl(action.precondition)}
    :effect {expr_to_pddl(action.effect)}
  )"""


def domain_to_pddl(domain: Domain) -> str:
    requirements = " ".join(sorted(domain.requirements)) if domain.requirements else ":strips"
    types = " ".join(domain.types)
    constants = typed_block_to_pddl(domain.constants, domain.constants_raw)
    predicates = "\n    ".join(predicate_to_pddl(p) for p in domain.predicates)
    actions = "\n\n".join(action_to_pddl(a) for a in domain.actions.values())

    return f"""(define (domain {domain.name})
  (:requirements {requirements})

  (:types
    {types}
  )

  (:constants
    {constants}
  )

  (:predicates
    {predicates}
  )

{actions}
)
"""


def expr_list_to_pddl(expr: Expr | None) -> str:
    if expr is not None and expr.op == "and":
        return "\n    ".join(expr_to_pddl(arg) for arg in expr.args)  # type: ignore[arg-type]
    return expr_to_pddl(expr)


def problem_to_pddl(problem: Problem, domain_name: str | None = None) -> str:
    objects = typed_block_to_pddl(problem.objects, problem.objects_raw)
    domain = domain_name or "domain"
    return f"""(define (problem {problem.name})
  (:domain {domain})

  (:objects
    {objects}
  )

  (:init
    {expr_list_to_pddl(problem.init)}
  )

  (:goal
    {expr_to_pddl(problem.goal)}
  )
)
"""

