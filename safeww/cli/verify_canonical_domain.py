from __future__ import annotations

import argparse
import json
from pathlib import Path

from safeww.world_model.verify import (
    find_canonical_artifact_dirs,
    verify_canonical_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify canonical FSM-derived PDDL domains and manifests."
    )
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        help="One directory containing canonical_domain.pddl and canonical_manifest.json.",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        help="Root containing canonical domain artifact directories.",
    )
    parser.add_argument(
        "--fast-downward",
        type=Path,
        help="Optional path to fast-downward.py for terminal reachability smoke solves.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        help="Optional JSON report output path.",
    )
    parser.add_argument("--search", default="astar(lmcut())")
    args = parser.parse_args()

    if not args.canonical_dir and not args.canonical_root:
        parser.error("Provide --canonical-dir or --canonical-root")

    artifact_dirs = (
        [args.canonical_dir]
        if args.canonical_dir
        else find_canonical_artifact_dirs(args.canonical_root)
    )
    reports = [
        verify_canonical_artifact(
            artifact_dir,
            fast_downward=args.fast_downward,
            search=args.search,
        )
        for artifact_dir in artifact_dirs
    ]

    payload = [report.to_dict() for report in reports]
    if args.report_file:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    ok_count = sum(1 for report in reports if report.ok)
    print(f"Verified {ok_count}/{len(reports)} canonical domain(s).")
    for report in reports:
        status = "OK" if report.ok else "FAIL"
        errors = sum(1 for issue in report.issues if issue.level == "error")
        warnings = sum(1 for issue in report.issues if issue.level == "warning")
        reach = ""
        if report.terminal_reachability:
            solved = sum(1 for item in report.terminal_reachability if item.solved)
            reach = f", terminals {solved}/{len(report.terminal_reachability)}"
        print(
            f"[{status}] {report.artifact_dir} "
            f"pages={report.pages} actions={report.actions} predicates={report.predicates} "
            f"errors={errors} warnings={warnings}{reach}"
        )

    if ok_count != len(reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
