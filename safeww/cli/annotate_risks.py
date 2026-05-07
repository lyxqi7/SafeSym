from __future__ import annotations

import argparse
import json
from pathlib import Path

from safeww.safety.risk_profile import annotate_risk_profiles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer task-level safety risk categories from base sas_plan and legacy safety rules."
    )
    parser.add_argument("--pddl-root", type=Path, required=True)
    parser.add_argument("--risk-rules", type=Path, default=Path("configs/safety_rules.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print inferred profiles without updating metadata.json.")
    parser.add_argument("--jsonl", type=Path, help="Optional JSONL export of inferred risk profiles.")
    args = parser.parse_args()

    results = annotate_risk_profiles(
        pddl_root=args.pddl_root,
        risk_rules_file=args.risk_rules,
        write_metadata=not args.dry_run,
    )

    rows: list[dict[str, object]] = []
    for artifact, profile in results:
        row = {
            "task_dir": str(artifact.task_dir),
            "site": artifact.task_dir.parent.name,
            "task": artifact.task_dir.name,
            "risk_categories": profile.risk_categories,
            "risk_matches": [match.__dict__ for match in profile.risk_matches],
        }
        rows.append(row)

    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    non_empty = sum(1 for _, profile in results if profile.risk_categories)
    print(f"Annotated {len(results)} tasks; {non_empty} tasks have at least one risk category.")


if __name__ == "__main__":
    main()
