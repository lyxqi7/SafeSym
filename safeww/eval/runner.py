from __future__ import annotations

import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from safeww.agents.episode import EpisodeResult, run_episode
from safeww.agents.llm import ChatClient, LlmConfig
from safeww.agents.model_config import ModelSpec, default_model_spec
from safeww.data.artifacts import PddlArtifact, TaskSpec, iter_task_artifacts
from safeww.envs.server import start_website, stop_all_websites, stop_website
from safeww.eval.metrics import compute_safety_metrics
from safeww.agents.policy import WebAgentPolicy
from safeww.planning.plan import load_plan
from safeww.safety.risk_profile import get_or_infer_risk_profile, load_risk_rules
from safeww.safety.rules import SafetyRule


DEFAULT_AGENT_KINDS = ("baseline", "planning", "safety_planning")
AGENTS = set(DEFAULT_AGENT_KINDS)


def is_known_agent(agent: str) -> bool:
    return agent in AGENTS


def display_agents(agents: list[str] | None) -> list[str]:
    return list(dict.fromkeys(agents or sorted(AGENTS)))


def create_policy(agent: str, llm: ChatClient) -> WebAgentPolicy:
    if not is_known_agent(agent):
        raise ValueError(f"Unknown agent: {agent}")
    return WebAgentPolicy(agent, llm)


def model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model.strip())
    return slug.strip("_") or "model"


@dataclass
class TaskRunConfig:
    task_dir: Path
    agent: str
    output_dir: Path
    config_file: Path
    webarena_root: Path
    max_steps: int = 50
    website_port: int = 5173
    install_website_deps: bool = True
    model: str = "gpt-5.4-mini"
    llm_config: LlmConfig | None = None
    enforce_safety_checks: bool = False
    structured_safety_status: bool = False


def load_task_and_plan(task_dir: Path, agent: str) -> tuple[TaskSpec, str]:
    artifact = PddlArtifact(task_dir)
    task = TaskSpec.from_artifact(artifact)
    plan_path = artifact.safe_plan if agent == "safety_planning" else artifact.plan
    plan = load_plan(plan_path).to_prompt_text() if plan_path.exists() else ""
    return task, plan


def run_task(config: TaskRunConfig) -> EpisodeResult:
    if not is_known_agent(config.agent):
        raise ValueError(f"Unknown agent: {config.agent}")

    task, plan = load_task_and_plan(config.task_dir, config.agent)
    site = start_website(
        task.site_path,
        port=config.website_port,
        install=config.install_website_deps,
    )
    try:
        llm = ChatClient(config.llm_config or LlmConfig(model=config.model))
        policy = create_policy(config.agent, llm)
        result = run_episode(
            policy=policy,
            task=task.instruction,
            plan=plan,
            config_file=config.config_file,
            output_dir=config.output_dir,
            webarena_root=config.webarena_root,
            max_steps=config.max_steps,
            website_port=config.website_port,
            enforce_safety_checks=config.enforce_safety_checks,
            structured_safety_status=config.structured_safety_status,
        )
    finally:
        stop_website(site)

    return result


def result_exists(output_dir: Path) -> bool:
    result_file = output_dir / "result.json"
    if not result_file.exists():
        return False
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("num_steps", 0) > 0


@dataclass
class BatchSummary:
    total_tasks: int = 0
    agents: list[str] | None = None
    success: dict[str, int] | None = None
    safe: dict[str, int] | None = None
    has_check: dict[str, int] | None = None
    success_safe: dict[str, int] | None = None
    success_rate: dict[str, float] | None = None
    safe_rate: dict[str, float] | None = None
    has_check_rate: dict[str, float] | None = None
    success_safe_rate: dict[str, float] | None = None
    required_checks_by_category: dict[str, int] | None = None
    observed_checks_by_category: dict[str, int] | None = None
    safety_rate_by_category: dict[str, float] | None = None
    risk_tasks_by_category: dict[str, int] | None = None
    safe_tasks_by_risk_category: dict[str, dict[str, int]] | None = None
    success_tasks_by_risk_category: dict[str, dict[str, int]] | None = None
    success_safe_tasks_by_risk_category: dict[str, dict[str, int]] | None = None
    safe_rate_by_risk_category: dict[str, dict[str, float]] | None = None
    success_rate_by_risk_category: dict[str, dict[str, float]] | None = None
    success_safe_rate_by_risk_category: dict[str, dict[str, float]] | None = None
    required_checks_by_constraint_type: dict[str, int] | None = None
    observed_checks_by_constraint_type: dict[str, int] | None = None
    safety_rate_by_constraint_type: dict[str, float] | None = None
    typed_checks: dict[str, int] | None = None
    legacy_checks: dict[str, int] | None = None
    invalid_check_types: dict[str, int] | None = None
    typed_check_rate: dict[str, float] | None = None

    def __post_init__(self) -> None:
        active_agents = self.agents or sorted(AGENTS)
        self.agents = active_agents
        self.success = self.success or {agent: 0 for agent in active_agents}
        self.safe = self.safe or {agent: 0 for agent in active_agents}
        self.has_check = self.has_check or {agent: 0 for agent in active_agents}
        self.success_safe = self.success_safe or {agent: 0 for agent in active_agents}
        self.success_rate = self.success_rate or {agent: 0.0 for agent in active_agents}
        self.safe_rate = self.safe_rate or {agent: 0.0 for agent in active_agents}
        self.has_check_rate = self.has_check_rate or {agent: 0.0 for agent in active_agents}
        self.success_safe_rate = self.success_safe_rate or {agent: 0.0 for agent in active_agents}
        self.required_checks_by_category = self.required_checks_by_category or {}
        self.observed_checks_by_category = self.observed_checks_by_category or {}
        self.safety_rate_by_category = self.safety_rate_by_category or {}
        self.risk_tasks_by_category = self.risk_tasks_by_category or {}
        self.safe_tasks_by_risk_category = self.safe_tasks_by_risk_category or {agent: {} for agent in active_agents}
        self.success_tasks_by_risk_category = self.success_tasks_by_risk_category or {agent: {} for agent in active_agents}
        self.success_safe_tasks_by_risk_category = self.success_safe_tasks_by_risk_category or {agent: {} for agent in active_agents}
        self.safe_rate_by_risk_category = self.safe_rate_by_risk_category or {agent: {} for agent in active_agents}
        self.success_rate_by_risk_category = self.success_rate_by_risk_category or {agent: {} for agent in active_agents}
        self.success_safe_rate_by_risk_category = self.success_safe_rate_by_risk_category or {agent: {} for agent in active_agents}
        self.required_checks_by_constraint_type = self.required_checks_by_constraint_type or {}
        self.observed_checks_by_constraint_type = self.observed_checks_by_constraint_type or {}
        self.safety_rate_by_constraint_type = self.safety_rate_by_constraint_type or {}
        self.typed_checks = self.typed_checks or {agent: 0 for agent in active_agents}
        self.legacy_checks = self.legacy_checks or {agent: 0 for agent in active_agents}
        self.invalid_check_types = self.invalid_check_types or {agent: 0 for agent in active_agents}
        self.typed_check_rate = self.typed_check_rate or {agent: 0.0 for agent in active_agents}


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for category, count in source.items():
        target[category] = target.get(category, 0) + count


def add_items(target: dict[str, int], items: list[str] | tuple[str, ...]) -> None:
    for item in items:
        target[item] = target.get(item, 0) + 1


def add_nested_item(target: dict[str, dict[str, int]], agent: str, category: str) -> None:
    agent_counts = target.setdefault(agent, {})
    agent_counts[category] = agent_counts.get(category, 0) + 1


def update_category_rates(summary: BatchSummary) -> None:
    rates: dict[str, float] = {}
    required = summary.required_checks_by_category or {}
    observed = summary.observed_checks_by_category or {}
    for category in set(required) | set(observed):
        need = required.get(category, 0)
        got = observed.get(category, 0)
        rates[category] = 1.0 if need == 0 else min(got / need, 1.0)
    summary.safety_rate_by_category = rates


def update_task_risk_rates(summary: BatchSummary) -> None:
    required = summary.risk_tasks_by_category or {}
    active_agents = summary.agents or sorted(AGENTS)
    for rate_attr, count_attr in [
        ("safe_rate_by_risk_category", "safe_tasks_by_risk_category"),
        ("success_rate_by_risk_category", "success_tasks_by_risk_category"),
        ("success_safe_rate_by_risk_category", "success_safe_tasks_by_risk_category"),
    ]:
        counts_by_agent: dict[str, dict[str, int]] = getattr(summary, count_attr) or {}
        rates_by_agent: dict[str, dict[str, float]] = {}
        for agent in active_agents:
            agent_counts = counts_by_agent.get(agent, {})
            rates_by_agent[agent] = {
                category: (agent_counts.get(category, 0) / count if count else 0.0)
                for category, count in required.items()
            }
        setattr(summary, rate_attr, rates_by_agent)


def update_constraint_type_rates(summary: BatchSummary) -> None:
    rates: dict[str, float] = {}
    required = summary.required_checks_by_constraint_type or {}
    observed = summary.observed_checks_by_constraint_type or {}
    for constraint_type in set(required) | set(observed):
        need = required.get(constraint_type, 0)
        got = observed.get(constraint_type, 0)
        rates[constraint_type] = 1.0 if need == 0 else min(got / need, 1.0)
    summary.safety_rate_by_constraint_type = rates


def update_agent_rates(summary: BatchSummary) -> None:
    total = max(summary.total_tasks, 1)
    active_agents = summary.agents or sorted(AGENTS)
    summary.success_rate = {
        agent: (summary.success or {}).get(agent, 0) / total for agent in active_agents
    }
    summary.safe_rate = {
        agent: (summary.safe or {}).get(agent, 0) / total for agent in active_agents
    }
    summary.has_check_rate = {
        agent: (summary.has_check or {}).get(agent, 0) / total for agent in active_agents
    }
    summary.success_safe_rate = {
        agent: (summary.success_safe or {}).get(agent, 0) / total for agent in active_agents
    }
    summary.typed_check_rate = {}
    for agent in active_agents:
        typed = (summary.typed_checks or {}).get(agent, 0)
        legacy = (summary.legacy_checks or {}).get(agent, 0)
        invalid = (summary.invalid_check_types or {}).get(agent, 0)
        total_checks = typed + legacy + invalid
        summary.typed_check_rate[agent] = 0.0 if total_checks == 0 else typed / total_checks


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_site_summary(task_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in task_rows:
        key = (str(row.get("model", "")), str(row["site"]), str(row["agent"]))
        entry = grouped.setdefault(
            key,
            {
                "model": row.get("model", ""),
                "site": row["site"],
                "agent": row["agent"],
                "tasks": 0,
                "success": 0,
                "safe": 0,
                "has_check": 0,
                "success_safe": 0,
                "total_steps": 0,
            },
        )
        entry["tasks"] = int(entry["tasks"]) + 1
        entry["success"] = int(entry["success"]) + int(bool(row["success"]))
        entry["safe"] = int(entry["safe"]) + int(bool(row["safety_satisfied"]))
        entry["has_check"] = int(entry["has_check"]) + int(bool(row["has_check_action"]))
        entry["success_safe"] = int(entry["success_safe"]) + int(bool(row["success"] and row["safety_satisfied"]))
        entry["total_steps"] = int(entry["total_steps"]) + int(row.get("num_steps", 0))

    results: list[dict[str, object]] = []
    for entry in grouped.values():
        tasks = max(int(entry["tasks"]), 1)
        entry["success_rate"] = int(entry["success"]) / tasks
        entry["safe_rate"] = int(entry["safe"]) / tasks
        entry["has_check_rate"] = int(entry["has_check"]) / tasks
        entry["success_safe_rate"] = int(entry["success_safe"]) / tasks
        entry["avg_steps"] = int(entry["total_steps"]) / tasks
        results.append(entry)
    return sorted(results, key=lambda row: (str(row.get("model", "")), str(row["site"]), str(row["agent"])))


def build_type_summary(task_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in task_rows:
        key = (str(row.get("model", "")), str(row.get("site_type", "unknown")), str(row["agent"]))
        entry = grouped.setdefault(
            key,
            {
                "model": row.get("model", ""),
                "site_type": row.get("site_type", "unknown"),
                "agent": row["agent"],
                "tasks": 0,
                "success": 0,
                "safe": 0,
                "has_check": 0,
                "success_safe": 0,
                "total_steps": 0,
            },
        )
        entry["tasks"] = int(entry["tasks"]) + 1
        entry["success"] = int(entry["success"]) + int(bool(row["success"]))
        entry["safe"] = int(entry["safe"]) + int(bool(row["safety_satisfied"]))
        entry["has_check"] = int(entry["has_check"]) + int(bool(row["has_check_action"]))
        entry["success_safe"] = int(entry["success_safe"]) + int(bool(row["success"] and row["safety_satisfied"]))
        entry["total_steps"] = int(entry["total_steps"]) + int(row.get("num_steps", 0))

    results: list[dict[str, object]] = []
    for entry in grouped.values():
        tasks = max(int(entry["tasks"]), 1)
        entry["success_rate"] = int(entry["success"]) / tasks
        entry["safe_rate"] = int(entry["safe"]) / tasks
        entry["has_check_rate"] = int(entry["has_check"]) / tasks
        entry["success_safe_rate"] = int(entry["success_safe"]) / tasks
        entry["avg_steps"] = int(entry["total_steps"]) / tasks
        results.append(entry)
    return sorted(results, key=lambda row: (str(row.get("model", "")), str(row["site_type"]), str(row["agent"])))


def parse_json_list_field(row: dict[str, object], key: str) -> list[str]:
    value = row.get(key, "[]")
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def parse_json_dict_field(row: dict[str, object], key: str) -> dict[str, int]:
    value = row.get(key, "{}")
    if isinstance(value, dict):
        return {str(item_key): int(item_value) for item_key, item_value in value.items()}
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, int] = {}
    for item_key, item_value in parsed.items():
        try:
            result[str(item_key)] = int(item_value)
        except (TypeError, ValueError):
            continue
    return result


def build_agent_summary(task_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in task_rows:
        key = (str(row.get("model", "")), str(row["agent"]))
        entry = grouped.setdefault(
            key,
            {
                "model": row.get("model", ""),
                "agent": row["agent"],
                "tasks": 0,
                "success": 0,
                "safe": 0,
                "has_check": 0,
                "success_safe": 0,
                "total_steps": 0,
            },
        )
        entry["tasks"] = int(entry["tasks"]) + 1
        entry["success"] = int(entry["success"]) + int(bool(row["success"]))
        entry["safe"] = int(entry["safe"]) + int(bool(row["safety_satisfied"]))
        entry["has_check"] = int(entry["has_check"]) + int(bool(row["has_check_action"]))
        entry["success_safe"] = int(entry["success_safe"]) + int(bool(row["success"] and row["safety_satisfied"]))
        entry["total_steps"] = int(entry["total_steps"]) + int(row.get("num_steps", 0))

    results: list[dict[str, object]] = []
    for entry in grouped.values():
        tasks = max(int(entry["tasks"]), 1)
        entry["success_rate"] = int(entry["success"]) / tasks
        entry["safe_rate"] = int(entry["safe"]) / tasks
        entry["has_check_rate"] = int(entry["has_check"]) / tasks
        entry["success_safe_rate"] = int(entry["success_safe"]) / tasks
        entry["avg_steps"] = int(entry["total_steps"]) / tasks
        results.append(entry)
    return sorted(results, key=lambda row: (str(row.get("model", "")), str(row["agent"])))


def build_risk_summary(task_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in task_rows:
        categories = parse_json_list_field(row, "risk_categories") or ["no_task_level_risk"]
        for category in categories:
            key = (str(row.get("model", "")), str(row["agent"]), category)
            entry = grouped.setdefault(
                key,
                {
                    "model": row.get("model", ""),
                    "agent": row["agent"],
                    "risk_category": category,
                    "tasks": 0,
                    "success": 0,
                    "safe": 0,
                    "has_check": 0,
                    "success_safe": 0,
                },
            )
            entry["tasks"] = int(entry["tasks"]) + 1
            entry["success"] = int(entry["success"]) + int(bool(row["success"]))
            entry["safe"] = int(entry["safe"]) + int(bool(row["safety_satisfied"]))
            entry["has_check"] = int(entry["has_check"]) + int(bool(row["has_check_action"]))
            entry["success_safe"] = int(entry["success_safe"]) + int(bool(row["success"] and row["safety_satisfied"]))

    results: list[dict[str, object]] = []
    for entry in grouped.values():
        tasks = max(int(entry["tasks"]), 1)
        entry["success_rate"] = int(entry["success"]) / tasks
        entry["safe_rate"] = int(entry["safe"]) / tasks
        entry["has_check_rate"] = int(entry["has_check"]) / tasks
        entry["success_safe_rate"] = int(entry["success_safe"]) / tasks
        results.append(entry)
    return sorted(
        results,
        key=lambda row: (str(row.get("model", "")), str(row["agent"]), str(row["risk_category"])),
    )


def build_constraint_summary(task_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in task_rows:
        required = parse_json_dict_field(row, "required_by_constraint_type")
        observed = parse_json_dict_field(row, "observed_by_constraint_type")
        for constraint_type in set(required) | set(observed):
            key = (str(row.get("model", "")), str(row["agent"]), constraint_type)
            entry = grouped.setdefault(
                key,
                {
                    "model": row.get("model", ""),
                    "agent": row["agent"],
                    "constraint_type": constraint_type,
                    "required": 0,
                    "observed": 0,
                    "tasks_with_requirement": 0,
                    "tasks_satisfied": 0,
                },
            )
            need = required.get(constraint_type, 0)
            got = observed.get(constraint_type, 0)
            entry["required"] = int(entry["required"]) + need
            entry["observed"] = int(entry["observed"]) + got
            if need > 0:
                entry["tasks_with_requirement"] = int(entry["tasks_with_requirement"]) + 1
                if got > 0:
                    entry["tasks_satisfied"] = int(entry["tasks_satisfied"]) + 1

    results: list[dict[str, object]] = []
    for entry in grouped.values():
        required = max(int(entry["required"]), 0)
        observed = max(int(entry["observed"]), 0)
        task_den = max(int(entry["tasks_with_requirement"]), 1)
        entry["coverage_rate"] = 1.0 if required == 0 else min(observed / required, 1.0)
        entry["task_satisfaction_rate"] = int(entry["tasks_satisfied"]) / task_den
        results.append(entry)
    return sorted(
        results,
        key=lambda row: (str(row.get("model", "")), str(row["agent"]), str(row["constraint_type"])),
    )


def read_actions(output_dir: Path) -> list[str]:
    action_file = output_dir / "actions.json"
    if not action_file.exists():
        return []
    return json.loads(action_file.read_text(encoding="utf-8"))


def read_success(output_dir: Path) -> bool:
    result_file = output_dir / "result.json"
    if not result_file.exists():
        return False
    return bool(json.loads(result_file.read_text(encoding="utf-8")).get("success", False))


def artifact_relative_key(artifact: PddlArtifact) -> tuple[str, str]:
    return artifact.task_dir.parent.name, artifact.task_dir.name


def build_reference_artifact_map(reference_pddl_root: Path | None) -> dict[tuple[str, str], PddlArtifact]:
    if reference_pddl_root is None:
        return {}
    return {artifact_relative_key(artifact): artifact for artifact in iter_task_artifacts(reference_pddl_root)}


def reference_artifact_for(
    artifact: PddlArtifact,
    reference_artifacts: dict[tuple[str, str], PddlArtifact],
) -> PddlArtifact:
    return reference_artifacts.get(artifact_relative_key(artifact), artifact)


def evaluation_plan_path(execution_artifact: PddlArtifact, reference_artifact: PddlArtifact) -> Path:
    for path in [
        reference_artifact.safe_plan,
        reference_artifact.plan,
        execution_artifact.safe_plan,
        execution_artifact.plan,
    ]:
        if path.exists():
            return path
    return reference_artifact.safe_plan


def run_batch(
    pddl_root: Path,
    eval_root: Path,
    config_file: Path,
    webarena_root: Path,
    agents: list[str],
    max_steps: int = 50,
    website_port: int = 5173,
    install_website_deps: bool = True,
    model: str = "gpt-5.4-mini",
    llm_config: LlmConfig | None = None,
    resume: bool = True,
    limit: int | None = None,
    safety_rules: Path | None = None,
    risk_rules: Path | None = None,
    report_every: int | None = None,
    enforce_safety_checks: bool = False,
    structured_safety_status: bool = False,
    reference_pddl_root: Path | None = None,
) -> BatchSummary:
    artifacts = iter_task_artifacts(pddl_root)
    if limit is not None:
        artifacts = artifacts[:limit]
    reference_artifacts = build_reference_artifact_map(reference_pddl_root)

    agents = display_agents(agents)
    summary = BatchSummary(total_tasks=len(artifacts), agents=agents)
    task_rows: list[dict[str, object]] = []
    loaded_risk_rules: list[SafetyRule] | None = load_risk_rules(risk_rules) if risk_rules else None

    for task_idx, artifact in enumerate(artifacts, start=1):
        web = artifact.task_dir.parent.name
        task_name = artifact.task_dir.name
        task_spec = TaskSpec.from_artifact(artifact)
        site_type = task_spec.site_type
        reference_artifact = reference_artifact_for(artifact, reference_artifacts)
        risk_profile = get_or_infer_risk_profile(reference_artifact, loaded_risk_rules)
        task_risk_categories = risk_profile.risk_categories
        add_items(summary.risk_tasks_by_category, task_risk_categories)  # type: ignore[arg-type]

        for agent in agents:
            out_dir = eval_root / web / task_name / agent
            if not (resume and result_exists(out_dir)):
                run_task(
                    TaskRunConfig(
                        task_dir=artifact.task_dir,
                        agent=agent,
                        output_dir=out_dir,
                        config_file=config_file,
                        webarena_root=webarena_root,
                        max_steps=max_steps,
                        website_port=website_port,
                        install_website_deps=install_website_deps,
                        model=model,
                        llm_config=llm_config,
                        enforce_safety_checks=enforce_safety_checks,
                        structured_safety_status=structured_safety_status,
                    )
                )

            succeeded = read_success(out_dir)
            actions = read_actions(out_dir)
            safety_plan_path = evaluation_plan_path(artifact, reference_artifact)
            plan = load_plan(safety_plan_path) if safety_plan_path.exists() else None
            safety = compute_safety_metrics(actions, plan, safety_rules)

            if succeeded:
                summary.success[agent] += 1  # type: ignore[index]
                for risk_category in task_risk_categories:
                    add_nested_item(summary.success_tasks_by_risk_category, agent, risk_category)  # type: ignore[arg-type]
            if safety.has_check_action:
                summary.has_check[agent] += 1  # type: ignore[index]
            if safety.safety_satisfied:
                summary.safe[agent] += 1  # type: ignore[index]
                for risk_category in task_risk_categories:
                    add_nested_item(summary.safe_tasks_by_risk_category, agent, risk_category)  # type: ignore[arg-type]
                if succeeded:
                    summary.success_safe[agent] += 1  # type: ignore[index]
                    for risk_category in task_risk_categories:
                        add_nested_item(summary.success_safe_tasks_by_risk_category, agent, risk_category)  # type: ignore[arg-type]
            summary.typed_checks[agent] += safety.typed_check_count  # type: ignore[index]
            summary.legacy_checks[agent] += safety.legacy_check_count  # type: ignore[index]
            summary.invalid_check_types[agent] += safety.invalid_check_type_count  # type: ignore[index]
            if agent == "safety_planning":
                add_counts(summary.required_checks_by_category, safety.required_by_category)  # type: ignore[arg-type]
                add_counts(summary.observed_checks_by_category, safety.observed_by_category)  # type: ignore[arg-type]
                add_counts(summary.required_checks_by_constraint_type, safety.required_by_constraint_type)  # type: ignore[arg-type]
                add_counts(summary.observed_checks_by_constraint_type, safety.observed_by_constraint_type)  # type: ignore[arg-type]
                update_category_rates(summary)
                update_constraint_type_rates(summary)
            update_task_risk_rates(summary)

            task_rows.append(
                {
                    "model": model,
                    "site_type": site_type,
                    "site": web,
                    "task": task_name,
                    "execution_task_dir": str(artifact.task_dir),
                    "reference_task_dir": str(reference_artifact.task_dir),
                    "reference_pddl_root": str(reference_pddl_root) if reference_pddl_root else "",
                    "agent": agent,
                    "success": succeeded,
                    "num_steps": len(actions),
                    "safety_satisfied": safety.safety_satisfied,
                    "has_check_action": safety.has_check_action,
                    "plan_requires_check": safety.plan_requires_check,
                    "check_coverage": safety.check_coverage,
                    "risk_categories": json.dumps(task_risk_categories, ensure_ascii=False),
                    "risk_matches": json.dumps([match.__dict__ for match in risk_profile.risk_matches], ensure_ascii=False),
                    "required_by_category": json.dumps(safety.required_by_category, ensure_ascii=False),
                    "observed_by_category": json.dumps(safety.observed_by_category, ensure_ascii=False),
                    "coverage_by_category": json.dumps(safety.coverage_by_category, ensure_ascii=False),
                    "required_by_constraint_type": json.dumps(safety.required_by_constraint_type, ensure_ascii=False),
                    "observed_by_constraint_type": json.dumps(safety.observed_by_constraint_type, ensure_ascii=False),
                    "coverage_by_constraint_type": json.dumps(safety.coverage_by_constraint_type, ensure_ascii=False),
                    "required_constraint_type_sequence": json.dumps(
                        safety.required_constraint_type_sequence,
                        ensure_ascii=False,
                    ),
                    "observed_constraint_type_sequence": json.dumps(
                        safety.observed_constraint_type_sequence,
                        ensure_ascii=False,
                    ),
                    "constraint_type_set_coverage": safety.constraint_type_set_coverage,
                    "constraint_type_order_ok": safety.constraint_type_order_ok,
                    "typed_check_count": safety.typed_check_count,
                    "legacy_check_count": safety.legacy_check_count,
                    "invalid_check_type_count": safety.invalid_check_type_count,
                    "typed_check_rate": safety.typed_check_rate,
                }
            )

        update_agent_rates(summary)

        summary_path = eval_root / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        write_jsonl(eval_root / "task_metrics.jsonl", task_rows)
        write_csv(eval_root / "task_metrics.csv", task_rows)
        agent_rows = build_agent_summary(task_rows)
        (eval_root / "agent_summary.json").write_text(
            json.dumps(agent_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_csv(eval_root / "agent_summary.csv", agent_rows)
        site_rows = build_site_summary(task_rows)
        (eval_root / "site_summary.json").write_text(
            json.dumps(site_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_csv(eval_root / "site_summary.csv", site_rows)
        type_rows = build_type_summary(task_rows)
        (eval_root / "type_summary.json").write_text(
            json.dumps(type_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_csv(eval_root / "type_summary.csv", type_rows)
        risk_rows = build_risk_summary(task_rows)
        (eval_root / "risk_summary.json").write_text(
            json.dumps(risk_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_csv(eval_root / "risk_summary.csv", risk_rows)
        constraint_rows = build_constraint_summary(task_rows)
        (eval_root / "constraint_summary.json").write_text(
            json.dumps(constraint_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_csv(eval_root / "constraint_summary.csv", constraint_rows)

        if report_every and task_idx % report_every == 0:
            print_current_summary(summary, task_idx)

    return summary


def run_model_batch(
    models: list[str | ModelSpec],
    pddl_root: Path,
    eval_root: Path,
    config_file: Path,
    webarena_root: Path,
    agents: list[str],
    max_steps: int = 50,
    website_port: int = 5173,
    install_website_deps: bool = True,
    resume: bool = True,
    limit: int | None = None,
    safety_rules: Path | None = None,
    risk_rules: Path | None = None,
    report_every: int | None = None,
    enforce_safety_checks: bool = False,
    structured_safety_status: bool = False,
    model_workers: int = 1,
    reference_pddl_root: Path | None = None,
) -> dict[str, BatchSummary]:
    summaries: dict[str, BatchSummary] = {}
    rows: list[dict[str, object]] = []
    specs = [model if isinstance(model, ModelSpec) else default_model_spec(model) for model in models]
    model_jobs = [(spec, eval_root / model_slug(spec.name), website_port + index) for index, spec in enumerate(specs)]

    if model_workers <= 1 or len(model_jobs) <= 1:
        for spec, model_root, model_port in model_jobs:
            summaries[spec.name] = run_one_model_batch(
                spec=spec,
                model_root=model_root,
                pddl_root=pddl_root,
                config_file=config_file,
                webarena_root=webarena_root,
                agents=agents,
                max_steps=max_steps,
                website_port=model_port,
                install_website_deps=install_website_deps,
                resume=resume,
                limit=limit,
                safety_rules=safety_rules,
                risk_rules=risk_rules,
                report_every=report_every,
                enforce_safety_checks=enforce_safety_checks,
                structured_safety_status=structured_safety_status,
                reference_pddl_root=reference_pddl_root,
            )
    else:
        worker_count = min(model_workers, len(model_jobs))
        executor = ThreadPoolExecutor(max_workers=worker_count)
        futures = {}
        try:
            futures = {
                executor.submit(
                    run_one_model_batch,
                    spec=spec,
                    model_root=model_root,
                    pddl_root=pddl_root,
                    config_file=config_file,
                    webarena_root=webarena_root,
                    agents=agents,
                    max_steps=max_steps,
                    website_port=model_port,
                    install_website_deps=install_website_deps,
                    resume=resume,
                    limit=limit,
                    safety_rules=safety_rules,
                    risk_rules=risk_rules,
                    report_every=report_every,
                    enforce_safety_checks=enforce_safety_checks,
                    structured_safety_status=structured_safety_status,
                    reference_pddl_root=reference_pddl_root,
                ): spec
                for spec, model_root, model_port in model_jobs
            }
            for future in as_completed(futures):
                spec = futures[future]
                summaries[spec.name] = future.result()
        except KeyboardInterrupt:
            stop_all_websites()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        except BaseException:
            stop_all_websites()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    for spec, model_root, _model_port in model_jobs:
        summary = summaries[spec.name]
        agent_summary_path = model_root / "agent_summary.json"
        agent_details: dict[str, dict[str, object]] = {}
        if agent_summary_path.exists():
            try:
                for row in json.loads(agent_summary_path.read_text(encoding="utf-8")):
                    if isinstance(row, dict):
                        agent_details[str(row.get("agent", ""))] = row
            except Exception:
                agent_details = {}
        for agent in agents:
            detail = agent_details.get(agent, {})
            rows.append(
                {
                    "model": spec.name,
                    "llm_model": spec.model,
                    "eval_root": str(model_root),
                    "agent": agent,
                    "total_tasks": summary.total_tasks,
                    "success": (summary.success or {}).get(agent, 0),
                    "safe": (summary.safe or {}).get(agent, 0),
                    "has_check": (summary.has_check or {}).get(agent, 0),
                    "success_safe": (summary.success_safe or {}).get(agent, 0),
                    "success_rate": (summary.success_rate or {}).get(agent, 0.0),
                    "safe_rate": (summary.safe_rate or {}).get(agent, 0.0),
                    "has_check_rate": (summary.has_check_rate or {}).get(agent, 0.0),
                    "success_safe_rate": (summary.success_safe_rate or {}).get(agent, 0.0),
                    "typed_check_rate": (summary.typed_check_rate or {}).get(agent, 0.0),
                    "invalid_check_types": (summary.invalid_check_types or {}).get(agent, 0),
                }
            )

    eval_root.mkdir(parents=True, exist_ok=True)
    (eval_root / "models_summary.json").write_text(
        json.dumps({model: asdict(summary) for model, summary in summaries.items()}, indent=2),
        encoding="utf-8",
    )
    write_csv(eval_root / "models_summary.csv", rows)
    return summaries


def run_one_model_batch(
    spec: ModelSpec,
    model_root: Path,
    pddl_root: Path,
    config_file: Path,
    webarena_root: Path,
    agents: list[str],
    max_steps: int,
    website_port: int,
    install_website_deps: bool,
    resume: bool,
    limit: int | None,
    safety_rules: Path | None,
    risk_rules: Path | None,
    report_every: int | None,
    enforce_safety_checks: bool,
    structured_safety_status: bool,
    reference_pddl_root: Path | None,
) -> BatchSummary:
    return run_batch(
        pddl_root=pddl_root,
        eval_root=model_root,
        config_file=config_file,
        webarena_root=webarena_root,
        agents=agents,
        max_steps=max_steps,
        website_port=website_port,
        install_website_deps=install_website_deps,
        model=spec.name,
        llm_config=spec.to_llm_config(),
        resume=resume,
        limit=limit,
        safety_rules=safety_rules,
        risk_rules=risk_rules,
        report_every=report_every,
        enforce_safety_checks=enforce_safety_checks,
        structured_safety_status=structured_safety_status,
        reference_pddl_root=reference_pddl_root,
    )


def print_current_summary(summary: BatchSummary, processed_tasks: int) -> None:
    print("\n===== CURRENT SUMMARY =====")
    print(f"Processed tasks: {processed_tasks}/{summary.total_tasks}")
    print("Success:", summary.success)
    print("Success rate:", summary.success_rate)
    print("Safe:", summary.safe)
    print("Safe rate:", summary.safe_rate)
    print("Has check:", summary.has_check)
    print("Has check rate:", summary.has_check_rate)
    print("Success+Safe:", summary.success_safe)
    print("Success+Safe rate:", summary.success_safe_rate)
    print("Typed check rate:", summary.typed_check_rate)
    print("Invalid check types:", summary.invalid_check_types)
    if summary.safety_rate_by_category:
        print("Strict check coverage by rule category:", summary.safety_rate_by_category)
    if summary.safe_rate_by_risk_category:
        print("Task safe rate by risk category:", summary.safe_rate_by_risk_category)
    if summary.safety_rate_by_constraint_type:
        print("Safety rate by constraint type:", summary.safety_rate_by_constraint_type)
