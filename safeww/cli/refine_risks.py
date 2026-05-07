from __future__ import annotations

import argparse
import json
from pathlib import Path

from safeww.agents.llm import ChatClient
from safeww.cli.generate_world_model import load_model
from safeww.data.artifacts import PddlArtifact, iter_task_artifacts
from safeww.planning.plan import load_plan
from safeww.safety.risk_dynamic import RiskLabelReviewer
from safeww.safety.risk_profile import (
    RiskProfile,
    infer_artifact_risk_profile,
    load_risk_rules,
    write_metadata_risk_profile,
)


def load_metadata(artifact: PddlArtifact) -> dict:
    if artifact.metadata.exists():
        return json.loads(artifact.metadata.read_text(encoding="utf-8"))
    return {}


def is_internal_task_dir(task_dir: Path, root: Path) -> bool:
    try:
        return any(part.startswith("_") for part in task_dir.relative_to(root).parts)
    except ValueError:
        return False


def risk_refinement_complete(
    task_dir: Path,
    *,
    static_only: bool,
    write_metadata: bool,
) -> bool:
    artifact = PddlArtifact(task_dir)
    output = task_dir / "risk_profile_refined.json"
    if not output.exists() or output.stat().st_size == 0:
        return False
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected_mode = "static_only" if static_only else "dynamic_review"
    if payload.get("mode") != expected_mode:
        return False
    if write_metadata:
        if not artifact.metadata.exists():
            return False
        metadata = json.loads(artifact.metadata.read_text(encoding="utf-8"))
        return isinstance(metadata.get("risk_categories"), list)
    return True


def write_refined_risk_artifact(
    artifact: PddlArtifact,
    *,
    static_profile: RiskProfile,
    final_categories: list[str],
    mode: str,
    reason: str = "",
    raw_output: str | None = None,
    warnings: list[str] | None = None,
) -> None:
    payload = {
        "schema_version": "safesym-task-risk-profile-v1",
        "task_dir": str(artifact.task_dir),
        "mode": mode,
        "static_risk_categories": static_profile.risk_categories,
        "static_risk_matches": [match.__dict__ for match in static_profile.risk_matches],
        "final_risk_categories": final_categories,
        "reason": reason,
        "raw_output": raw_output,
        "warnings": warnings or [],
    }
    (artifact.task_dir / "risk_profile_refined.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def refine_risk_task(
    *,
    task_dir: Path,
    risk_rules_file: Path,
    reviewer: RiskLabelReviewer | None,
    static_only: bool,
    write_metadata: bool,
) -> bool:
    artifact = PddlArtifact(task_dir)
    if not artifact.plan.exists():
        print(f"[SKIP] missing sas_plan: {task_dir}")
        return False
    rules = load_risk_rules(risk_rules_file)
    static_profile = infer_artifact_risk_profile(artifact, rules)
    final_categories = list(static_profile.risk_categories)
    reason = ""
    raw_output = None
    warnings: list[str] = []
    mode = "static_only"

    if not static_only:
        assert reviewer is not None
        metadata = load_metadata(artifact)
        result = reviewer.review(
            task=str(metadata.get("instruction") or ""),
            metadata=metadata,
            plan=load_plan(artifact.plan),
            static_profile=static_profile,
        )
        final_categories = result.risk_categories
        reason = result.reason
        raw_output = result.raw_output
        warnings = result.warnings or []
        mode = "dynamic_review"

    refined_profile = RiskProfile(risk_categories=final_categories, risk_matches=static_profile.risk_matches)
    if write_metadata and artifact.metadata.exists():
        write_metadata_risk_profile(artifact, refined_profile)
    write_refined_risk_artifact(
        artifact,
        static_profile=static_profile,
        final_categories=final_categories,
        mode=mode,
        reason=reason,
        raw_output=raw_output,
        warnings=warnings,
    )
    print(f"[OK] {task_dir} static={static_profile.risk_categories} final={final_categories}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine task-level risk labels from static rules and optional LLM review."
    )
    parser.add_argument("--task-dir", type=Path)
    parser.add_argument("--pddl-root", type=Path)
    parser.add_argument("--risk-rules", type=Path, default=Path("configs/safety_rules.json"))
    parser.add_argument("--models-config", type=Path)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--write-metadata", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks whose matching risk_profile_refined.json already exists.",
    )
    args = parser.parse_args()

    if not args.task_dir and not args.pddl_root:
        parser.error("Provide --task-dir or --pddl-root")

    task_dirs = [args.task_dir] if args.task_dir else [
        artifact.task_dir
        for artifact in iter_task_artifacts(args.pddl_root)
        if not is_internal_task_dir(artifact.task_dir, args.pddl_root)
    ]
    skipped = 0
    if args.resume:
        pending_dirs: list[Path] = []
        for task_dir in task_dirs:
            if risk_refinement_complete(
                task_dir,
                static_only=args.static_only,
                write_metadata=args.write_metadata,
            ):
                print(f"[SKIP] resume complete: {task_dir}")
                skipped += 1
            else:
                pending_dirs.append(task_dir)
        task_dirs = pending_dirs

    reviewer = None
    if not args.static_only and task_dirs:
        spec = load_model(args.models_config, args.model)
        reviewer = RiskLabelReviewer(client=ChatClient(spec.to_llm_config()))

    success = 0
    for task_dir in task_dirs:
        try:
            if refine_risk_task(
                task_dir=task_dir,
                risk_rules_file=args.risk_rules,
                reviewer=reviewer,
                static_only=args.static_only,
                write_metadata=args.write_metadata,
            ):
                success += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {task_dir}: {type(exc).__name__}: {exc}")
    print(f"Processed {success}/{len(task_dirs)} tasks")
    if skipped:
        print(f"Skipped {skipped} completed tasks with --resume")


if __name__ == "__main__":
    main()
