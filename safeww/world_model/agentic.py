from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from safeww.agents.llm import ChatClient
from safeww.planning.fast_downward import solve_pddl

from .pddl_io import extract_pddl_sections, validate_pddl_pair
from .prompts import (
    GENERATOR_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    build_generator_prompt,
    build_verifier_prompt,
)


@dataclass
class GenerationAttempt:
    round_index: int
    syntax_attempt_index: int
    stage: str
    ok: bool
    feedback: str
    domain: str | None = None
    problem: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "syntax_attempt_index": self.syntax_attempt_index,
            "stage": self.stage,
            "ok": self.ok,
            "feedback": self.feedback,
            "has_domain": self.domain is not None,
            "has_problem": self.problem is not None,
        }


@dataclass
class WorldModelGenerationResult:
    ok: bool
    domain: str | None
    problem: str | None
    plan_path: Path | None
    attempts: list[GenerationAttempt] = field(default_factory=list)
    final_feedback: str = ""

    def report(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "plan_path": str(self.plan_path) if self.plan_path else None,
            "final_feedback": self.final_feedback,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


class AgenticWorldModelPipeline:
    def __init__(
        self,
        *,
        generator_client: ChatClient,
        verifier_client: ChatClient | None = None,
        fast_downward: Path | None = None,
        max_rounds: int = 3,
        max_syntax_repairs: int = 3,
        max_fsm_actions: int | None = None,
        search: str = "astar(lmcut())",
    ):
        self.generator_client = generator_client
        self.verifier_client = verifier_client or generator_client
        self.fast_downward = fast_downward
        self.max_rounds = max_rounds
        self.max_syntax_repairs = max_syntax_repairs
        self.max_fsm_actions = max_fsm_actions
        self.search = search

    def run(self, *, fsm: dict[str, Any], task: str, output_dir: Path) -> WorldModelGenerationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        attempts: list[GenerationAttempt] = []
        feedback: str | None = None
        previous_domain: str | None = None
        previous_problem: str | None = None

        for round_index in range(1, self.max_rounds + 1):
            syntax_result = self._generate_until_syntax_ok(
                fsm=fsm,
                task=task,
                feedback=feedback,
                previous_domain=previous_domain,
                previous_problem=previous_problem,
                round_index=round_index,
                attempts=attempts,
            )
            if syntax_result is None:
                final_feedback = "PDDL syntax/consistency repair limit reached."
                self._write_trace(output_dir, attempts, final_feedback)
                return WorldModelGenerationResult(False, previous_domain, previous_problem, None, attempts, final_feedback)

            domain, problem = syntax_result
            previous_domain, previous_problem = domain, problem
            domain_file = output_dir / "domain.pddl"
            problem_file = output_dir / "problem.pddl"
            plan_file = output_dir / "sas_plan"
            domain_file.write_text(domain, encoding="utf-8")
            problem_file.write_text(problem, encoding="utf-8")

            if self.fast_downward is None:
                final_feedback = "Syntax/consistency validation passed; planner validation skipped."
                attempts.append(GenerationAttempt(round_index, 0, "planner_skipped", True, final_feedback, domain, problem))
                self._write_trace(output_dir, attempts, final_feedback)
                return WorldModelGenerationResult(True, domain, problem, None, attempts, final_feedback)

            planner_result = solve_pddl(
                self.fast_downward,
                domain_file,
                problem_file,
                plan_file,
                search=self.search,
                cwd=output_dir,
            )
            if planner_result.returncode == 0 and plan_file.exists():
                final_feedback = "Planner validation passed."
                attempts.append(GenerationAttempt(round_index, 0, "planner", True, final_feedback, domain, problem))
                self._write_trace(output_dir, attempts, final_feedback)
                return WorldModelGenerationResult(True, domain, problem, plan_file, attempts, final_feedback)

            planner_feedback = self._planner_feedback(planner_result)
            if round_index >= self.max_rounds:
                attempts.append(
                    GenerationAttempt(
                        round_index,
                        0,
                        "planner",
                        False,
                        planner_feedback,
                        domain,
                        problem,
                    )
                )
                final_feedback = "Planner validation failed on the final repair round."
                self._write_trace(output_dir, attempts, final_feedback)
                return WorldModelGenerationResult(False, domain, problem, None, attempts, final_feedback)

            verifier_feedback = self._verifier_feedback(
                task=task,
                domain=domain,
                problem=problem,
                validation_feedback="Local parser/consistency validation passed.",
                planner_feedback=planner_feedback,
            )
            attempts.append(
                GenerationAttempt(
                    round_index,
                    0,
                    "planner",
                    False,
                    verifier_feedback,
                    domain,
                    problem,
                )
            )
            feedback = verifier_feedback

        final_feedback = "Planner repair round limit reached."
        self._write_trace(output_dir, attempts, final_feedback)
        return WorldModelGenerationResult(False, previous_domain, previous_problem, None, attempts, final_feedback)

    def _generate_until_syntax_ok(
        self,
        *,
        fsm: dict[str, Any],
        task: str,
        feedback: str | None,
        previous_domain: str | None,
        previous_problem: str | None,
        round_index: int,
        attempts: list[GenerationAttempt],
    ) -> tuple[str, str] | None:
        local_feedback = feedback
        for syntax_attempt in range(1, self.max_syntax_repairs + 1):
            prompt = build_generator_prompt(
                fsm=fsm,
                task=task,
                feedback=local_feedback,
                previous_domain=previous_domain,
                previous_problem=previous_problem,
                max_actions=self.max_fsm_actions,
            )
            raw = self.generator_client.chat(GENERATOR_SYSTEM_PROMPT, prompt)
            try:
                domain, problem = extract_pddl_sections(raw)
            except Exception as exc:
                local_feedback = f"Output format error: {exc}. Return exact ===DOMAIN=== and ===PROBLEM=== sections."
                attempts.append(GenerationAttempt(round_index, syntax_attempt, "extract", False, local_feedback))
                continue

            validation = validate_pddl_pair(domain, problem, fsm=fsm)
            attempts.append(
                GenerationAttempt(
                    round_index,
                    syntax_attempt,
                    "syntax",
                    validation.ok,
                    validation.to_feedback(),
                    domain,
                    problem,
                )
            )
            if validation.ok:
                return domain, problem
            local_feedback = validation.to_feedback()
            previous_domain, previous_problem = domain, problem
        return None

    def _verifier_feedback(
        self,
        *,
        task: str,
        domain: str,
        problem: str,
        validation_feedback: str,
        planner_feedback: str,
    ) -> str:
        prompt = build_verifier_prompt(
            task=task,
            domain=domain,
            problem=problem,
            validation_feedback=validation_feedback,
            planner_feedback=planner_feedback,
        )
        try:
            return self.verifier_client.chat(VERIFIER_SYSTEM_PROMPT, prompt)
        except Exception as exc:
            return (
                "Planner failed and verifier LLM feedback failed. "
                f"Use deterministic planner feedback instead:\n{planner_feedback}\nVerifier error: {exc}"
            )

    def _planner_feedback(self, planner_result) -> str:
        stdout = (planner_result.stdout or "")[-3000:]
        stderr = (planner_result.stderr or "")[-3000:]
        return (
            f"Fast Downward return code: {planner_result.returncode}\n"
            f"STDOUT tail:\n{stdout}\n\nSTDERR tail:\n{stderr}"
        )

    def _write_trace(
        self, output_dir: Path, attempts: list[GenerationAttempt], final_feedback: str
    ) -> None:
        trace_file = output_dir / "generation_trace.jsonl"
        with trace_file.open("w", encoding="utf-8") as f:
            for attempt in attempts:
                f.write(json.dumps(attempt.to_dict(), ensure_ascii=False) + "\n")
        report = {
            "final_feedback": final_feedback,
            "attempts": [attempt.to_dict() for attempt in attempts],
        }
        (output_dir / "world_model_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
