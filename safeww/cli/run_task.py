from __future__ import annotations

import argparse
from pathlib import Path

from safeww.agents.model_config import load_model_specs
from safeww.eval.runner import TaskRunConfig, run_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one SafeSym task.")
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--agent", choices=["baseline", "planning", "safety_planning"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--webarena-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--website-port", type=int, default=5173)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--models-config", type=Path, help="JSON file describing per-model provider/env/base-url settings.")
    parser.add_argument("--model-name", help="Model name or alias to select from --models-config.")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument(
        "--enforce-safety-checks",
        action="store_true",
        help="Runtime safety override: convert risky steps into check actions when a pending obligation exists.",
    )
    parser.add_argument(
        "--structured-safety-status",
        action="store_true",
        help="Add structured pending-obligation status to safety_planning prompts and reject early check actions.",
    )
    args = parser.parse_args()
    llm_config = None
    model_name = args.model
    if args.models_config:
        specs = load_model_specs(args.models_config, [args.model_name or args.model])
        spec = specs[0]
        model_name = spec.name
        llm_config = spec.to_llm_config()

    result = run_task(
        TaskRunConfig(
            task_dir=args.task_dir,
            agent=args.agent,
            output_dir=args.output_dir,
            config_file=args.config_file,
            webarena_root=args.webarena_root,
            max_steps=args.max_steps,
            website_port=args.website_port,
            install_website_deps=not args.skip_install,
            model=model_name,
            llm_config=llm_config,
            enforce_safety_checks=args.enforce_safety_checks,
            structured_safety_status=args.structured_safety_status,
        )
    )
    print(f"success={result.success} num_steps={result.num_steps}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Active website processes have been asked to stop.")
        raise SystemExit(130)
