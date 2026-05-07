from __future__ import annotations

import argparse
from pathlib import Path

from safeww.data.artifacts import PddlArtifact, iter_task_artifacts
from safeww.planning.fast_downward import solve_pddl


def solve_task(task_dir: Path, fast_downward: Path, safe: bool, resume: bool = False) -> bool:
    artifact = PddlArtifact(task_dir)
    domain = artifact.safe_domain if safe else artifact.domain
    problem = artifact.safe_problem if safe else artifact.problem
    output = artifact.safe_plan if safe else artifact.plan

    if resume and output.exists() and output.stat().st_size > 0:
        print(f"[SKIP] resume complete: {output}")
        return True

    if not domain.exists() or not problem.exists():
        print(f"[SKIP] missing PDDL: {task_dir}")
        return False

    result = solve_pddl(fast_downward, domain, problem, output, cwd=task_dir)
    if result.returncode != 0 or not output.exists():
        print(f"[FAIL] {task_dir}")
        if result.stderr:
            print(result.stderr[-1000:])
        return False

    print(f"[OK] {output}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve base or safe PDDL artifacts.")
    parser.add_argument("--task-dir", type=Path, help="Single task artifact directory.")
    parser.add_argument("--pddl-root", type=Path, help="Root containing site/task artifact directories.")
    parser.add_argument("--fast-downward", type=Path, required=True)
    parser.add_argument("--safe", action="store_true", help="Solve safe_domain/safe_problem.")
    parser.add_argument("--resume", action="store_true", help="Skip tasks with an existing non-empty plan output.")
    args = parser.parse_args()

    if not args.task_dir and not args.pddl_root:
        parser.error("Provide --task-dir or --pddl-root")

    task_dirs = [args.task_dir] if args.task_dir else [a.task_dir for a in iter_task_artifacts(args.pddl_root)]
    total = len(task_dirs)
    success = sum(1 for task_dir in task_dirs if solve_task(task_dir, args.fast_downward, args.safe, args.resume))
    print(f"Solved {success}/{total} tasks")


if __name__ == "__main__":
    main()
