from __future__ import annotations

import argparse
from pathlib import Path

from safeww.data.artifacts import PddlArtifact, TaskSpec
from safeww.planning.plan import load_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one SafeSym task artifact.")
    parser.add_argument("task_dir", type=Path)
    args = parser.parse_args()

    artifact = PddlArtifact(args.task_dir)
    task = TaskSpec.from_artifact(artifact)
    print(f"website: {task.website}")
    print(f"task_index: {task.task_index}")
    print(f"instruction: {task.instruction}")
    print(f"site_path: {task.site_path}")

    for label, path in [("plan", artifact.plan), ("safe_plan", artifact.safe_plan)]:
        if path.exists():
            plan = load_plan(path)
            print(f"{label}: {len(plan.steps)} steps, {len(plan.check_steps)} check steps")
        else:
            print(f"{label}: missing")


if __name__ == "__main__":
    main()
