from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from safeww.agents.llm import ChatClient
from safeww.cli.generate_world_model import load_model
from safeww.world_model.agentic import AgenticWorldModelPipeline
from safeww.world_model.autowebworld_batch import (
    build_task_metadata,
    group_tasks_by_instruction,
    iter_site_batches,
)


WORLD_MODEL_ARTIFACTS = [
    "domain.pddl",
    "problem.pddl",
    "sas_plan",
    "generation_trace.jsonl",
    "world_model_report.json",
]


def copy_success_artifacts(temp_dir: Path, task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    for filename in WORLD_MODEL_ARTIFACTS:
        src = temp_dir / filename
        if src.exists():
            shutil.copy2(src, task_dir / filename)


def reusable_task_dir(reuse_root: Path | None, website: str, task_index: int) -> Path | None:
    if reuse_root is None:
        return None
    source_dir = reuse_root / website / f"task_{task_index}"
    if task_artifacts_exist(source_dir):
        return source_dir
    return None


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def copy_reused_world_model_artifacts(source_dir: Path, task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    for filename in WORLD_MODEL_ARTIFACTS:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, task_dir / filename)


def run_task_generation(
    *,
    generator_spec,
    verifier_spec,
    fast_downward: Path | None,
    max_rounds: int,
    max_syntax_repairs: int,
    max_fsm_actions: int | None,
    search: str,
    fsm: dict[str, Any],
    instruction: str,
    temp_dir: Path,
) -> tuple[bool, str]:
    pipeline = AgenticWorldModelPipeline(
        generator_client=ChatClient(generator_spec.to_llm_config()),
        verifier_client=ChatClient(verifier_spec.to_llm_config()),
        fast_downward=fast_downward,
        max_rounds=max_rounds,
        max_syntax_repairs=max_syntax_repairs,
        max_fsm_actions=max_fsm_actions,
        search=search,
    )
    result = pipeline.run(fsm=fsm, task=instruction, output_dir=temp_dir)
    return result.ok, result.final_feedback


def task_artifacts_exist(task_dir: Path) -> bool:
    return (task_dir / "domain.pddl").exists() and (task_dir / "problem.pddl").exists()


def failed_artifacts_exist(failed_root: Path, website: str, task_index: int) -> bool:
    return (failed_root / website / f"task_{task_index}").exists()


def stable_site_seed(seed: int, website: str) -> int:
    digest = hashlib.sha256(f"{seed}:{website}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def order_task_groups(task_groups, *, strategy: str, seed: int, website: str):
    ordered = list(task_groups)
    if strategy == "random":
        rng = random.Random(stable_site_seed(seed, website))
        rng.shuffle(ordered)
    return ordered


def sampled_task_window(
    ordered_task_groups,
    *,
    max_success_per_site: int,
    max_attempts_per_site: int,
    reuse_root: Path | None,
    additional_success_per_site: int | None,
):
    if reuse_root is not None and additional_success_per_site is None:
        window_size = max(max_success_per_site, max_attempts_per_site)
        return list(ordered_task_groups[:window_size])
    return list(ordered_task_groups)


def generation_quota_reached(generated_success_count: int, in_flight_count: int, success_goal: int) -> bool:
    return generated_success_count + in_flight_count >= success_goal


def generate_task_attempt(
    *,
    generator_spec,
    verifier_spec,
    fast_downward: Path | None,
    max_rounds: int,
    max_syntax_repairs: int,
    max_fsm_actions: int | None,
    search: str,
    fsm: dict[str, Any],
    instruction: str,
    temp_dir: Path,
) -> dict[str, Any]:
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    try:
        ok, feedback = run_task_generation(
            generator_spec=generator_spec,
            verifier_spec=verifier_spec,
            fast_downward=fast_downward,
            max_rounds=max_rounds,
            max_syntax_repairs=max_syntax_repairs,
            max_fsm_actions=max_fsm_actions,
            search=search,
            fsm=fsm,
            instruction=instruction,
            temp_dir=temp_dir,
        )
    except Exception as exc:  # noqa: BLE001 - long batch runs should continue.
        ok = False
        feedback = f"exception: {type(exc).__name__}: {exc}"
    return {"ok": ok, "feedback": feedback, "temp_dir": str(temp_dir)}


def write_summary(
    summary_file: Path,
    *,
    total_success: int,
    total_attempted: int,
    records: list[dict[str, Any]],
    total_reused: int = 0,
    total_skipped_existing: int = 0,
    total_skipped_known_failures: int = 0,
    run_config: dict[str, Any] | None = None,
) -> None:
    payload = {
        "total_success": total_success,
        "total_attempted": total_attempted,
        "total_reused": total_reused,
        "total_skipped_existing": total_skipped_existing,
        "total_skipped_known_failures": total_skipped_known_failures,
        "run_config": run_config or {},
        "records": records,
    }
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-generate task-specific PDDL world models with the agentic generator/verifier loop."
    )
    parser.add_argument(
        "--data-json",
        type=Path,
        default=Path("../AutoWebWorld/mixed_travel_media_commerce_communication_productivity_train/train.json"),
    )
    parser.add_argument(
        "--sites-root",
        type=Path,
        default=Path("../AutoWebWorld/autowebworld/whole_pipeline/env_generator/data_augmentation/new_all_projects"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("../AutoWebWorld/pddl_results"))
    parser.add_argument("--failed-root", type=Path)
    parser.add_argument(
        "--reuse-pddl-root",
        type=Path,
        help=(
            "Optional existing PDDL result root to reuse task-specific world-model artifacts from. "
            "The task selection order is still determined first; reuse only avoids regenerating a "
            "selected website/task that already has domain.pddl and problem.pddl in this root."
        ),
    )
    parser.add_argument("--fast-downward", type=Path)
    parser.add_argument("--models-config", type=Path)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--verifier-model")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-syntax-repairs", type=int, default=3)
    parser.add_argument("--max-fsm-actions", type=int)
    parser.add_argument("--search", default="astar(lmcut())")
    parser.add_argument("--max-success-per-site", type=int, default=8)
    parser.add_argument(
        "--sample-strategy",
        choices=["first", "random"],
        default="first",
        help="Task-group sampling order within each website. Default keeps the original contiguous order.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=2026,
        help="Seed used by --sample-strategy random. The seed is combined with the website name.",
    )
    parser.add_argument(
        "--additional-success-per-site",
        type=int,
        help=(
            "Generate at most this many new successful tasks per site in this run. "
            "When set, existing successful task directories do not count toward this limit."
        ),
    )
    parser.add_argument("--max-attempts-per-site", type=int, default=16)
    parser.add_argument(
        "--task-workers",
        type=int,
        default=1,
        help=(
            "Number of task generations to run concurrently within each website. "
            "Each worker creates its own LLM clients and writes to a task-specific temp directory."
        ),
    )
    parser.add_argument(
        "--skip-known-failures",
        action="store_true",
        help="Skip tasks that already have artifacts under failed-root.",
    )
    parser.add_argument("--sites", nargs="*", help="Optional website names to include.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-file",
        type=Path,
        help="Optional summary JSON path. Defaults to output-root/generation_summary.json.",
    )
    args = parser.parse_args()
    if args.max_success_per_site < 1:
        parser.error("--max-success-per-site must be at least 1.")
    if args.additional_success_per_site is not None and args.additional_success_per_site < 1:
        parser.error("--additional-success-per-site must be at least 1.")
    if args.max_attempts_per_site < 1:
        parser.error("--max-attempts-per-site must be at least 1.")
    if args.task_workers < 1:
        parser.error("--task-workers must be at least 1.")

    output_root = args.output_root
    failed_root = args.failed_root or (output_root / "_failed_generation")
    reuse_root = args.reuse_pddl_root
    temp_root = output_root / "_generation_tmp"
    summary_file = args.summary_file or (output_root / "generation_summary.json")

    data = json.loads(args.data_json.read_text(encoding="utf-8"))
    selected_sites = {site.lower() for site in args.sites} if args.sites else None
    sites = [
        site
        for site in iter_site_batches(data, args.sites_root)
        if selected_sites is None or site.website.lower() in selected_sites
    ]

    if args.dry_run:
        for site in sites:
            task_groups = group_tasks_by_instruction(site.items)
            ordered = order_task_groups(
                task_groups,
                strategy=args.sample_strategy,
                seed=args.sample_seed,
                website=site.website,
            )
            sampled = sampled_task_window(
                ordered,
                max_success_per_site=args.max_success_per_site,
                max_attempts_per_site=args.max_attempts_per_site,
                reuse_root=reuse_root,
                additional_success_per_site=args.additional_success_per_site,
            )
            preview = [task.task_index for task in sampled[: args.max_success_per_site]]
            attempt_window = [task.task_index for task in sampled[: args.max_attempts_per_site]]
            reusable_preview = [
                task.task_index
                for task in sampled[: args.max_attempts_per_site]
                if reusable_task_dir(reuse_root, site.website, task.task_index) is not None
            ]
            print(
                f"[DRY] {site.website}: {len(task_groups)} grouped tasks; "
                f"sample={args.sample_strategy}, seed={args.sample_seed}, preview={preview}, "
                f"attempt_window={attempt_window}"
                + (f", reusable={reusable_preview}" if reuse_root else "")
            )
        return

    generator_spec = load_model(args.models_config, args.model)
    verifier_spec = load_model(args.models_config, args.verifier_model) if args.verifier_model else generator_spec

    output_root.mkdir(parents=True, exist_ok=True)
    failed_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    run_config = {
        "sample_strategy": args.sample_strategy,
        "sample_seed": args.sample_seed,
        "max_success_per_site": args.max_success_per_site,
        "additional_success_per_site": args.additional_success_per_site,
        "max_attempts_per_site": args.max_attempts_per_site,
        "task_workers": args.task_workers,
        "reuse_pddl_root": str(reuse_root) if reuse_root else None,
    }
    total_success = 0
    total_reused = 0
    total_attempted = 0
    total_skipped_existing = 0
    total_skipped_known_failures = 0

    try:
        for site in sites:
            fsm_path = site.site_path / "fsm.json"
            if not fsm_path.exists():
                summary.append(
                    {
                        "website": site.website,
                        "status": "missing_fsm",
                        "site_path": str(site.site_path),
                    }
                )
                write_summary(
                    summary_file,
                    total_success=total_success,
                    total_attempted=total_attempted,
                    records=summary,
                    total_reused=total_reused,
                    total_skipped_existing=total_skipped_existing,
                    total_skipped_known_failures=total_skipped_known_failures,
                    run_config=run_config,
                )
                print(f"[SKIP] missing FSM: {site.site_path}")
                continue

            fsm = json.loads(fsm_path.read_text(encoding="utf-8"))
            task_groups = group_tasks_by_instruction(site.items)
            ordered_task_groups = order_task_groups(
                task_groups,
                strategy=args.sample_strategy,
                seed=args.sample_seed,
                website=site.website,
            )
            sampled_task_groups = sampled_task_window(
                ordered_task_groups,
                max_success_per_site=args.max_success_per_site,
                max_attempts_per_site=args.max_attempts_per_site,
                reuse_root=reuse_root,
                additional_success_per_site=args.additional_success_per_site,
            )
            site_success_count = 0
            site_new_success_count = 0
            site_generated_success_count = 0
            attempt_count = 0
            print(
                f"\n===== Processing site: {site.website} ({len(task_groups)} grouped tasks, "
                f"sample={args.sample_strategy}, seed={args.sample_seed}) ====="
            )

            next_candidate_index = 0
            futures: dict[Future[dict[str, Any]], Any] = {}

            def target_reached() -> bool:
                if args.additional_success_per_site is not None:
                    return site_new_success_count >= args.additional_success_per_site
                return site_success_count >= args.max_success_per_site

            def potential_target_reached() -> bool:
                if args.additional_success_per_site is not None:
                    return site_new_success_count + len(futures) >= args.additional_success_per_site
                return site_success_count + len(futures) >= args.max_success_per_site

            def write_current_summary() -> None:
                write_summary(
                    summary_file,
                    total_success=total_success,
                    total_attempted=total_attempted,
                    records=summary,
                    total_reused=total_reused,
                    total_skipped_existing=total_skipped_existing,
                    total_skipped_known_failures=total_skipped_known_failures,
                    run_config=run_config,
                )

            def submit_next(executor: ThreadPoolExecutor) -> bool:
                nonlocal next_candidate_index
                nonlocal site_success_count, site_new_success_count
                nonlocal attempt_count, total_attempted, total_success
                nonlocal total_reused, total_skipped_existing, total_skipped_known_failures

                while next_candidate_index < len(sampled_task_groups):
                    if target_reached() or potential_target_reached():
                        return False
                    task = sampled_task_groups[next_candidate_index]
                    next_candidate_index += 1

                    if attempt_count >= args.max_attempts_per_site:
                        return False

                    task_dir = output_root / site.website / f"task_{task.task_index}"
                    if task_artifacts_exist(task_dir) and not args.overwrite:
                        print(f"[SKIP] existing task: {task_dir}")
                        site_success_count += 1
                        total_skipped_existing += 1
                        summary.append(
                            {
                                "website": site.website,
                                "site_type": site.site_type,
                                "task_index": task.task_index,
                                "status": "skipped_existing",
                                "task_dir": str(task_dir),
                                "sample_strategy": args.sample_strategy,
                                "sample_seed": args.sample_seed,
                            }
                        )
                        write_current_summary()
                        continue

                    source_dir = reusable_task_dir(reuse_root, site.website, task.task_index)
                    if source_dir is not None and source_dir.resolve() != task_dir.resolve():
                        print(f"[REUSE] {site.website}/task_{task.task_index}: {source_dir}")
                        copy_reused_world_model_artifacts(source_dir, task_dir)
                        metadata = build_task_metadata(
                            site=site,
                            task=task,
                            status="success",
                            attempt_index=0,
                            final_feedback=f"reused world model artifacts from {source_dir}",
                        )
                        metadata["sample_strategy"] = args.sample_strategy
                        metadata["sample_seed"] = args.sample_seed
                        metadata["world_model_source"] = "reused"
                        metadata["reused_from"] = str(source_dir)
                        source_metadata = read_json_if_exists(source_dir / "metadata.json")
                        if source_metadata:
                            metadata["reused_source_status"] = source_metadata.get("status")
                            metadata["reused_source_sample_strategy"] = source_metadata.get("sample_strategy")
                            metadata["reused_source_sample_seed"] = source_metadata.get("sample_seed")
                        (task_dir / "metadata.json").write_text(
                            json.dumps(metadata, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        site_success_count += 1
                        site_new_success_count += 1
                        total_success += 1
                        total_reused += 1
                        summary.append(
                            {
                                "website": site.website,
                                "site_type": site.site_type,
                                "task_index": task.task_index,
                                "status": "reused",
                                "task_dir": str(task_dir),
                                "reused_from": str(source_dir),
                                "sample_strategy": args.sample_strategy,
                                "sample_seed": args.sample_seed,
                            }
                        )
                        write_current_summary()
                        continue

                    if (
                        args.skip_known_failures
                        and not args.overwrite
                        and failed_artifacts_exist(failed_root, site.website, task.task_index)
                    ):
                        print(f"[SKIP] known failed task: {site.website}/task_{task.task_index}")
                        total_skipped_known_failures += 1
                        summary.append(
                            {
                                "website": site.website,
                                "site_type": site.site_type,
                                "task_index": task.task_index,
                                "status": "skipped_known_failure",
                                "failed_dir": str(failed_root / site.website / f"task_{task.task_index}"),
                                "sample_strategy": args.sample_strategy,
                                "sample_seed": args.sample_seed,
                            }
                        )
                        write_current_summary()
                        continue

                    attempt_count += 1
                    total_attempted += 1
                    temp_dir = temp_root / site.website / f"task_{task.task_index}"
                    print(f"--- {site.website}/task_{task.task_index}: {task.instruction[:100]}")
                    future = executor.submit(
                        generate_task_attempt,
                        generator_spec=generator_spec,
                        verifier_spec=verifier_spec,
                        fast_downward=args.fast_downward,
                        max_rounds=args.max_rounds,
                        max_syntax_repairs=args.max_syntax_repairs,
                        max_fsm_actions=args.max_fsm_actions,
                        search=args.search,
                        fsm=fsm,
                        instruction=task.instruction,
                        temp_dir=temp_dir,
                    )
                    futures[future] = task
                    return True
                return False

            if not target_reached() and sampled_task_groups:
                with ThreadPoolExecutor(max_workers=args.task_workers) as executor:
                    while len(futures) < args.task_workers and submit_next(executor):
                        pass

                    while futures:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            task = futures.pop(future)
                            result = future.result()
                            ok = bool(result["ok"])
                            feedback = str(result["feedback"])
                            temp_dir = Path(str(result["temp_dir"]))
                            summary.append(
                                {
                                    "website": site.website,
                                    "site_type": site.site_type,
                                    "task_index": task.task_index,
                                    "status": "success" if ok else "failed",
                                    "feedback": feedback,
                                    "sample_strategy": args.sample_strategy,
                                    "sample_seed": args.sample_seed,
                                }
                            )
                            write_summary(
                                summary_file,
                                total_success=total_success,
                                total_attempted=total_attempted,
                                records=summary,
                                total_reused=total_reused,
                                total_skipped_existing=total_skipped_existing,
                                total_skipped_known_failures=total_skipped_known_failures,
                                run_config=run_config,
                            )

                            if ok:
                                task_dir = output_root / site.website / f"task_{task.task_index}"
                                copy_success_artifacts(temp_dir, task_dir)
                                metadata = build_task_metadata(
                                    site=site,
                                    task=task,
                                    status="success",
                                    attempt_index=1,
                                    final_feedback=feedback,
                                )
                                metadata["sample_strategy"] = args.sample_strategy
                                metadata["sample_seed"] = args.sample_seed
                                (task_dir / "metadata.json").write_text(
                                    json.dumps(metadata, indent=2, ensure_ascii=False),
                                    encoding="utf-8",
                                )
                                site_success_count += 1
                                site_new_success_count += 1
                                site_generated_success_count += 1
                                total_success += 1
                                write_summary(
                                    summary_file,
                                    total_success=total_success,
                                    total_attempted=total_attempted,
                                    records=summary,
                                    total_reused=total_reused,
                                    total_skipped_existing=total_skipped_existing,
                                    total_skipped_known_failures=total_skipped_known_failures,
                                    run_config=run_config,
                                )
                                print(f"[OK] {task_dir}")
                            else:
                                failed_dir = failed_root / site.website / f"task_{task.task_index}"
                                if temp_dir.exists():
                                    copy_success_artifacts(temp_dir, failed_dir)
                                metadata = build_task_metadata(
                                    site=site,
                                    task=task,
                                    status="failed",
                                    attempt_index=1,
                                    final_feedback=feedback,
                                )
                                metadata["sample_strategy"] = args.sample_strategy
                                metadata["sample_seed"] = args.sample_seed
                                failed_dir.mkdir(parents=True, exist_ok=True)
                                (failed_dir / "metadata.json").write_text(
                                    json.dumps(metadata, indent=2, ensure_ascii=False),
                                    encoding="utf-8",
                                )
                                print(f"[FAIL] {site.website}/task_{task.task_index}: {feedback}")

                        while len(futures) < args.task_workers and submit_next(executor):
                            pass

            print(
                f"Finished {site.website}: {site_new_success_count} new successful tasks, "
                f"{site_success_count} total successes seen, {attempt_count} attempted"
            )

    except KeyboardInterrupt:
        write_summary(
            summary_file,
            total_success=total_success,
            total_attempted=total_attempted,
            records=summary,
            total_reused=total_reused,
            total_skipped_existing=total_skipped_existing,
            total_skipped_known_failures=total_skipped_known_failures,
            run_config=run_config,
        )
        print(f"\nInterrupted. Partial summary written to {summary_file}")
        raise

    write_summary(
        summary_file,
        total_success=total_success,
        total_attempted=total_attempted,
        records=summary,
        total_reused=total_reused,
        total_skipped_existing=total_skipped_existing,
        total_skipped_known_failures=total_skipped_known_failures,
        run_config=run_config,
    )
    print(f"\nPrepared {total_success} successful tasks from {total_attempted} generation attempts.")
    if total_reused:
        print(f"Reused {total_reused} successful task directories from {reuse_root}.")
    if total_skipped_existing:
        print(f"Skipped {total_skipped_existing} existing successful task directories.")
    if total_skipped_known_failures:
        print(f"Skipped {total_skipped_known_failures} known failed task directories.")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
