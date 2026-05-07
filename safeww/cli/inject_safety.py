from __future__ import annotations

import argparse
from pathlib import Path

from safeww.data.artifacts import PddlArtifact, iter_task_artifacts
from safeww.pddl.serializer import domain_to_pddl, problem_to_pddl
from safeww.safety import compile_safe_pddl


def inject_task(task_dir: Path, rule_file: Path) -> bool:
    artifact = PddlArtifact(task_dir)
    if not artifact.has_base_pddl():
        print(f"[SKIP] missing domain/problem: {task_dir}")
        return False

    domain, problem = compile_safe_pddl(artifact.domain, artifact.problem, rule_file)
    artifact.safe_domain.write_text(domain_to_pddl(domain), encoding="utf-8")
    artifact.safe_problem.write_text(problem_to_pddl(problem, domain.name), encoding="utf-8")
    print(f"[OK] {task_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject safety constraints into PDDL artifacts.")
    parser.add_argument("--task-dir", type=Path, help="Single task artifact directory.")
    parser.add_argument("--pddl-root", type=Path, help="Root containing site/task artifact directories.")
    parser.add_argument("--rules", type=Path, required=True, help="Safety rules JSON file.")
    args = parser.parse_args()

    if not args.task_dir and not args.pddl_root:
        parser.error("Provide --task-dir or --pddl-root")

    task_dirs = [args.task_dir] if args.task_dir else [a.task_dir for a in iter_task_artifacts(args.pddl_root)]
    total = len(task_dirs)
    success = sum(1 for task_dir in task_dirs if inject_task(task_dir, args.rules))
    print(f"Processed {success}/{total} tasks")


if __name__ == "__main__":
    main()

