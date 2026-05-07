from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safeww.agents.llm import ChatClient
from safeww.cli.generate_world_model import load_model
from safeww.data.artifacts import PddlArtifact, iter_task_artifacts
from safeww.pddl.parser import parse_domain, parse_problem
from safeww.pddl.serializer import domain_to_pddl, problem_to_pddl
from safeww.planning.plan import load_plan
from safeww.safety.dynamic import DynamicSafetyRefiner
from safeww.safety.injector import SafetyCompiler
from safeww.safety.obligations import infer_static_obligations, rules_from_obligations
from safeww.safety.rules import load_rules


def is_internal_task_dir(task_dir: Path, root: Path) -> bool:
    try:
        relative_parts = task_dir.relative_to(root).parts
    except ValueError:
        return False
    return any(part.startswith("_") for part in relative_parts)


def load_metadata(artifact: PddlArtifact) -> dict[str, Any]:
    if artifact.metadata.exists():
        return json.loads(artifact.metadata.read_text(encoding="utf-8"))
    return {}


def safety_refinement_complete(task_dir: Path, *, static_only: bool) -> bool:
    artifact = PddlArtifact(task_dir)
    obligation_file = task_dir / "safety_obligations.json"
    required_files = [artifact.safe_domain, artifact.safe_problem, obligation_file]
    if any(not path.exists() or path.stat().st_size == 0 for path in required_files):
        return False
    try:
        payload = json.loads(obligation_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected_mode = "static_only" if static_only else "dynamic_task_local"
    return payload.get("mode") == expected_mode


def write_obligation_artifact(
    artifact: PddlArtifact,
    *,
    metadata: dict[str, Any],
    static_obligations,
    decisions: list[dict[str, Any]],
    final_obligations,
    warnings: list[str],
    trace: dict[str, Any],
    static_only: bool,
) -> None:
    payload = {
        "schema_version": "safesym-safety-obligations-v1",
        "task_dir": str(artifact.task_dir),
        "website": metadata.get("website"),
        "task_index": metadata.get("task_index"),
        "instruction": metadata.get("instruction"),
        "mode": "static_only" if static_only else "dynamic_task_local",
        "static_obligations": [obligation.to_dict() for obligation in static_obligations],
        "llm_decisions": decisions,
        "final_obligations": [obligation.to_dict() for obligation in final_obligations],
        "warnings": warnings,
        "trace": trace,
    }
    (artifact.task_dir / "safety_obligations.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def refine_task(
    *,
    task_dir: Path,
    rule_file: Path,
    refiner: DynamicSafetyRefiner | None,
    static_only: bool,
) -> bool:
    artifact = PddlArtifact(task_dir)
    if not artifact.has_base_pddl():
        print(f"[SKIP] missing domain/problem: {task_dir}")
        return False

    metadata = load_metadata(artifact)
    task_text = str(metadata.get("instruction") or "")
    domain = parse_domain(artifact.domain.read_text(encoding="utf-8"))
    problem = parse_problem(artifact.problem.read_text(encoding="utf-8"))
    rules = load_rules(json.loads(rule_file.read_text(encoding="utf-8")))
    static_obligations = infer_static_obligations(domain, rules)
    plan = load_plan(artifact.plan) if artifact.plan.exists() else None

    decisions: list[dict[str, Any]] = []
    warnings: list[str] = []
    trace: dict[str, Any] = {}
    final_obligations = static_obligations

    if not static_only:
        assert refiner is not None
        result = refiner.refine(
            task=task_text,
            metadata=metadata,
            domain=domain,
            plan=plan,
            static_obligations=static_obligations,
        )
        decisions = result.decisions
        final_obligations = result.final_obligations
        warnings = result.warnings
        trace = result.trace

    compiler = SafetyCompiler(rules_from_obligations(final_obligations))
    safe_domain, safe_problem = compiler.compile(domain, problem)
    artifact.safe_domain.write_text(domain_to_pddl(safe_domain), encoding="utf-8")
    artifact.safe_problem.write_text(problem_to_pddl(safe_problem, safe_domain.name), encoding="utf-8")
    write_obligation_artifact(
        artifact,
        metadata=metadata,
        static_obligations=static_obligations,
        decisions=decisions,
        final_obligations=final_obligations,
        warnings=warnings,
        trace=trace,
        static_only=static_only,
    )
    print(
        f"[OK] {task_dir} static={len(static_obligations)} final={len(final_obligations)}"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine task-local safety obligations and inject safe PDDL."
    )
    parser.add_argument("--task-dir", type=Path, help="Single task artifact directory.")
    parser.add_argument("--pddl-root", type=Path, help="Root containing site/task artifact directories.")
    parser.add_argument("--rules", type=Path, required=True, help="Static constraint rule JSON file.")
    parser.add_argument("--models-config", type=Path)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--normalizer-model")
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Skip LLM refinement and write obligations from static rules only.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks whose matching safety_obligations.json and safe PDDL already exist.",
    )
    args = parser.parse_args()

    if not args.task_dir and not args.pddl_root:
        parser.error("Provide --task-dir or --pddl-root")

    if args.task_dir:
        task_dirs = [args.task_dir]
    else:
        task_dirs = [
            artifact.task_dir
            for artifact in iter_task_artifacts(args.pddl_root)
            if not is_internal_task_dir(artifact.task_dir, args.pddl_root)
        ]
    skipped = 0
    if args.resume:
        pending_dirs: list[Path] = []
        for task_dir in task_dirs:
            if safety_refinement_complete(task_dir, static_only=args.static_only):
                print(f"[SKIP] resume complete: {task_dir}")
                skipped += 1
            else:
                pending_dirs.append(task_dir)
        task_dirs = pending_dirs

    refiner = None
    if not args.static_only and task_dirs:
        auditor_spec = load_model(args.models_config, args.model)
        normalizer_spec = (
            load_model(args.models_config, args.normalizer_model)
            if args.normalizer_model
            else auditor_spec
        )
        refiner = DynamicSafetyRefiner(
            auditor_client=ChatClient(auditor_spec.to_llm_config()),
            normalizer_client=ChatClient(normalizer_spec.to_llm_config()),
        )
    total = len(task_dirs)
    success = 0
    for task_dir in task_dirs:
        try:
            if refine_task(
                task_dir=task_dir,
                rule_file=args.rules,
                refiner=refiner,
                static_only=args.static_only,
            ):
                success += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {task_dir}: {type(exc).__name__}: {exc}")
    print(f"Processed {success}/{total} tasks")
    if skipped:
        print(f"Skipped {skipped} completed tasks with --resume")


if __name__ == "__main__":
    main()
