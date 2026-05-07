from __future__ import annotations

import argparse
import json
from pathlib import Path

from safeww.agents.llm import ChatClient
from safeww.agents.model_config import default_model_spec, load_model_specs
from safeww.world_model.agentic import AgenticWorldModelPipeline


def load_model(path: Path | None, model: str | None):
    if path:
        names = [model] if model else None
        specs = load_model_specs(path, names=names)
        if len(specs) != 1:
            raise ValueError("Select exactly one model for world-model generation.")
        return specs[0]
    return default_model_spec(model or "gpt-5.4-mini")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a task-specific PDDL world model from FSM/task using a generator-verifier loop."
    )
    parser.add_argument("--fsm", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fast-downward", type=Path)
    parser.add_argument("--models-config", type=Path)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--verifier-model")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-syntax-repairs", type=int, default=3)
    parser.add_argument(
        "--max-fsm-actions",
        type=int,
        help="Optional cap for prompt size. By default include all FSM actions.",
    )
    parser.add_argument("--search", default="astar(lmcut())")
    args = parser.parse_args()

    fsm = json.loads(args.fsm.read_text(encoding="utf-8"))
    generator_spec = load_model(args.models_config, args.model)
    verifier_spec = load_model(args.models_config, args.verifier_model) if args.verifier_model else generator_spec

    pipeline = AgenticWorldModelPipeline(
        generator_client=ChatClient(generator_spec.to_llm_config()),
        verifier_client=ChatClient(verifier_spec.to_llm_config()),
        fast_downward=args.fast_downward,
        max_rounds=args.max_rounds,
        max_syntax_repairs=args.max_syntax_repairs,
        max_fsm_actions=args.max_fsm_actions,
        search=args.search,
    )
    result = pipeline.run(fsm=fsm, task=args.task, output_dir=args.output_dir)
    print(json.dumps(result.report(), indent=2, ensure_ascii=False))
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
