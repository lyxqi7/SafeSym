from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlanStep:
    index: int
    name: str
    args: tuple[str, ...] = ()
    raw: str = ""

    @property
    def is_check(self) -> bool:
        return self.name.lower().startswith("check")


@dataclass(frozen=True)
class Plan:
    steps: list[PlanStep]
    cost: str | None = None

    def to_prompt_text(self) -> str:
        return "\n".join(step.raw for step in self.steps)

    @property
    def check_steps(self) -> list[PlanStep]:
        return [step for step in self.steps if step.is_check]


def parse_plan(text: str) -> Plan:
    steps: list[PlanStep] = []
    cost: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(";"):
            cost = stripped
            continue

        inner = stripped[1:-1] if stripped.startswith("(") and stripped.endswith(")") else stripped
        parts = inner.split()
        if not parts:
            continue
        steps.append(PlanStep(len(steps), parts[0], tuple(parts[1:]), stripped))

    return Plan(steps=steps, cost=cost)


def load_plan(path: Path | str) -> Plan:
    return parse_plan(Path(path).read_text(encoding="utf-8"))

