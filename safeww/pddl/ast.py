from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Expr:
    op: str
    args: list["Expr"] | list[str] = field(default_factory=list)

    def clone(self) -> "Expr":
        if self.op == "atom":
            return Expr("atom", list(self.args))  # type: ignore[arg-type]
        return Expr(self.op, [arg.clone() for arg in self.args])  # type: ignore[union-attr]


@dataclass
class Predicate:
    name: str
    args: list[tuple[str, str | None]] = field(default_factory=list)


@dataclass
class Action:
    name: str
    parameters: list[tuple[str, str | None]] = field(default_factory=list)
    precondition: Expr | None = None
    effect: Expr | None = None
    is_support: bool = False


@dataclass
class Domain:
    name: str = "domain"
    requirements: set[str] = field(default_factory=set)
    types: list[str] = field(default_factory=list)
    constants: dict[str, list[str]] = field(default_factory=dict)
    constants_raw: list[str] = field(default_factory=list)
    predicates: list[Predicate] = field(default_factory=list)
    actions: dict[str, Action] = field(default_factory=dict)


@dataclass
class Problem:
    name: str = "problem"
    objects: dict[str, list[str]] = field(default_factory=dict)
    objects_raw: list[str] = field(default_factory=list)
    init: Expr | None = None
    goal: Expr | None = None

