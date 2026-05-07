from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from safeww.agents.model_config import load_model_specs
from safeww.eval.runner import run_batch, run_model_batch


DEFAULT_RISK_RULES = Path(__file__).resolve().parents[2] / "configs" / "safety_rules.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SafeSym batch evaluation.")
    parser.add_argument("--pddl-root", type=Path, required=True)
    parser.add_argument(
        "--reference-pddl-root",
        type=Path,
        help=(
            "Optional artifact root used only for evaluation labels and safety references. "
            "Agent execution still reads task plans from --pddl-root."
        ),
    )
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--webarena-root", type=Path, required=True)
    parser.add_argument("--agents", nargs="+", default=["baseline", "planning", "safety_planning"])
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--website-port", type=int, default=5173)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--models", nargs="+", help="Run several models. Each model writes to eval-root/<model-slug>.")
    parser.add_argument("--models-config", type=Path, help="JSON file describing per-model provider/env/base-url settings.")
    parser.add_argument("--model-workers", type=int, default=1, help="Run up to N models in parallel.")
    parser.add_argument("--rules", type=Path, help="Safety/constraint rules JSON for per-constraint metrics.")
    parser.add_argument(
        "--risk-rules",
        type=Path,
        help="Task-risk rules used to bind risk categories to tasks from the base sas_plan.",
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report-every", type=int, help="Print rolling summary every N tasks.")
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

    risk_rules = args.risk_rules or (DEFAULT_RISK_RULES if DEFAULT_RISK_RULES.exists() else None)

    if args.models_config:
        model_specs = load_model_specs(args.models_config, args.models)
        summaries = run_model_batch(
            models=model_specs,
            pddl_root=args.pddl_root,
            eval_root=args.eval_root,
            config_file=args.config_file,
            webarena_root=args.webarena_root,
            agents=args.agents,
            max_steps=args.max_steps,
            website_port=args.website_port,
            install_website_deps=not args.skip_install,
            resume=not args.no_resume,
            limit=args.limit,
            safety_rules=args.rules,
            risk_rules=risk_rules,
            report_every=args.report_every,
            enforce_safety_checks=args.enforce_safety_checks,
            structured_safety_status=args.structured_safety_status,
            model_workers=args.model_workers,
            reference_pddl_root=args.reference_pddl_root,
        )
        print(json.dumps({model: asdict(summary) for model, summary in summaries.items()}, indent=2))
    elif args.models:
        summaries = run_model_batch(
            models=args.models,
            pddl_root=args.pddl_root,
            eval_root=args.eval_root,
            config_file=args.config_file,
            webarena_root=args.webarena_root,
            agents=args.agents,
            max_steps=args.max_steps,
            website_port=args.website_port,
            install_website_deps=not args.skip_install,
            resume=not args.no_resume,
            limit=args.limit,
            safety_rules=args.rules,
            risk_rules=risk_rules,
            report_every=args.report_every,
            enforce_safety_checks=args.enforce_safety_checks,
            structured_safety_status=args.structured_safety_status,
            model_workers=args.model_workers,
            reference_pddl_root=args.reference_pddl_root,
        )
        print(json.dumps({model: asdict(summary) for model, summary in summaries.items()}, indent=2))
    else:
        summary = run_batch(
            pddl_root=args.pddl_root,
            eval_root=args.eval_root,
            config_file=args.config_file,
            webarena_root=args.webarena_root,
            agents=args.agents,
            max_steps=args.max_steps,
            website_port=args.website_port,
            install_website_deps=not args.skip_install,
            model=args.model,
            resume=not args.no_resume,
            limit=args.limit,
            safety_rules=args.rules,
            risk_rules=risk_rules,
            report_every=args.report_every,
            enforce_safety_checks=args.enforce_safety_checks,
            structured_safety_status=args.structured_safety_status,
            reference_pddl_root=args.reference_pddl_root,
        )
        print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Active website processes have been asked to stop.")
        raise SystemExit(130)
