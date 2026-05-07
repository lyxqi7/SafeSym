from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from safeww.data.artifacts import PddlArtifact, iter_task_artifacts
from safeww.pddl.ast import Action
from safeww.planning.plan import Plan, load_plan
from safeww.safety.injector import match_rule
from safeww.safety.rules import SafetyRule, load_rules


@dataclass(frozen=True)
class RiskMatch:
    risk_category: str
    rule_name: str
    action_name: str
    plan_step_index: int


@dataclass(frozen=True)
class RiskProfile:
    risk_categories: list[str]
    risk_matches: list[RiskMatch]

    def to_metadata(self) -> dict[str, object]:
        return {
            "risk_categories": self.risk_categories,
            "risk_matches": [asdict(match) for match in self.risk_matches],
        }


def load_risk_rules(path: Path | str) -> list[SafetyRule]:
    with Path(path).open("r", encoding="utf-8") as f:
        return load_rules(json.load(f))


def infer_plan_risk_profile(plan: Plan, rules: list[SafetyRule]) -> RiskProfile:
    matches: list[RiskMatch] = []
    seen_categories: set[str] = set()
    categories: list[str] = []

    for step in plan.steps:
        if step.is_check:
            continue
        action = Action(name=step.name)
        for rule in rules:
            if not match_rule(rule, action):
                continue
            for category in rule.risk_categories or [rule.risk_category]:
                if category not in seen_categories:
                    seen_categories.add(category)
                    categories.append(category)
                matches.append(
                    RiskMatch(
                        risk_category=category,
                        rule_name=rule.name,
                        action_name=step.name,
                        plan_step_index=step.index,
                    )
                )

    return RiskProfile(risk_categories=categories, risk_matches=matches)


def infer_artifact_risk_profile(
    artifact: PddlArtifact,
    rules: list[SafetyRule],
) -> RiskProfile:
    if not artifact.plan.exists():
        return RiskProfile(risk_categories=[], risk_matches=[])
    return infer_plan_risk_profile(load_plan(artifact.plan), rules)


def read_metadata_risk_profile(artifact: PddlArtifact) -> RiskProfile | None:
    if not artifact.metadata.exists():
        return None
    metadata = json.loads(artifact.metadata.read_text(encoding="utf-8"))
    categories = metadata.get("risk_categories")
    if not isinstance(categories, list):
        return None
    raw_matches = metadata.get("risk_matches", [])
    matches: list[RiskMatch] = []
    if isinstance(raw_matches, list):
        for item in raw_matches:
            if not isinstance(item, dict):
                continue
            try:
                matches.append(
                    RiskMatch(
                        risk_category=str(item["risk_category"]),
                        rule_name=str(item["rule_name"]),
                        action_name=str(item["action_name"]),
                        plan_step_index=int(item["plan_step_index"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return RiskProfile(risk_categories=[str(category) for category in categories], risk_matches=matches)


def read_refined_risk_profile(artifact: PddlArtifact) -> RiskProfile | None:
    path = artifact.task_dir / "risk_profile_refined.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    categories = payload.get("final_risk_categories")
    if not isinstance(categories, list):
        return None

    raw_matches = payload.get("static_risk_matches", [])
    matches: list[RiskMatch] = []
    if isinstance(raw_matches, list):
        for item in raw_matches:
            if not isinstance(item, dict):
                continue
            try:
                matches.append(
                    RiskMatch(
                        risk_category=str(item["risk_category"]),
                        rule_name=str(item["rule_name"]),
                        action_name=str(item["action_name"]),
                        plan_step_index=int(item["plan_step_index"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return RiskProfile(risk_categories=[str(category) for category in categories], risk_matches=matches)


def get_or_infer_risk_profile(
    artifact: PddlArtifact,
    risk_rules: list[SafetyRule] | None = None,
) -> RiskProfile:
    refined_profile = read_refined_risk_profile(artifact)
    if refined_profile is not None:
        return refined_profile

    if risk_rules is not None:
        return infer_artifact_risk_profile(artifact, risk_rules)

    metadata_profile = read_metadata_risk_profile(artifact)
    if metadata_profile is not None:
        return metadata_profile
    else:
        return RiskProfile(risk_categories=[], risk_matches=[])


def write_metadata_risk_profile(artifact: PddlArtifact, profile: RiskProfile) -> None:
    metadata = json.loads(artifact.metadata.read_text(encoding="utf-8"))
    metadata.update(profile.to_metadata())
    artifact.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def annotate_risk_profiles(
    pddl_root: Path | str,
    risk_rules_file: Path | str,
    write_metadata: bool = True,
) -> list[tuple[PddlArtifact, RiskProfile]]:
    rules = load_risk_rules(risk_rules_file)
    results: list[tuple[PddlArtifact, RiskProfile]] = []
    for artifact in iter_task_artifacts(pddl_root):
        profile = infer_artifact_risk_profile(artifact, rules)
        if write_metadata:
            write_metadata_risk_profile(artifact, profile)
        results.append((artifact, profile))
    return results
