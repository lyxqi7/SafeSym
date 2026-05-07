from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PddlArtifact:
    task_dir: Path

    @property
    def domain(self) -> Path:
        return self.task_dir / "domain.pddl"

    @property
    def problem(self) -> Path:
        return self.task_dir / "problem.pddl"

    @property
    def plan(self) -> Path:
        return self.task_dir / "sas_plan"

    @property
    def safe_domain(self) -> Path:
        return self.task_dir / "safe_domain.pddl"

    @property
    def safe_problem(self) -> Path:
        return self.task_dir / "safe_problem.pddl"

    @property
    def safe_plan(self) -> Path:
        return self.task_dir / "safe_sas_plan"

    @property
    def metadata(self) -> Path:
        return self.task_dir / "metadata.json"

    def has_base_pddl(self) -> bool:
        return self.domain.exists() and self.problem.exists()

    def has_safe_pddl(self) -> bool:
        return self.safe_domain.exists() and self.safe_problem.exists()


@dataclass(frozen=True)
class TaskSpec:
    task_index: int
    instruction: str
    website: str
    site_type: str
    site_path: Path
    artifact: PddlArtifact
    img_filename: str | None = None
    risk_categories: tuple[str, ...] = ()

    @classmethod
    def from_artifact(cls, artifact: PddlArtifact) -> "TaskSpec":
        with artifact.metadata.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return cls(
            task_index=int(metadata["task_index"]),
            instruction=metadata["instruction"],
            website=metadata["website"],
            site_type=metadata.get("site_type") or infer_site_type(Path(metadata["site_path"]), metadata["website"]),
            site_path=Path(metadata["site_path"]),
            artifact=artifact,
            img_filename=metadata.get("img_filename"),
            risk_categories=tuple(str(category) for category in metadata.get("risk_categories", [])),
        )


def infer_site_type(site_path: Path, website: str) -> str:
    """Infer the AutoWebWorld category that contains a website project."""

    parts = list(site_path.parts)
    lowered = [part.lower() for part in parts]
    website_lower = website.lower()
    if website_lower in lowered:
        index = lowered.index(website_lower)
        if index > 0:
            return parts[index - 1]

    # Expected layout: .../<site_type>/<website>/web
    if site_path.name == "web" and site_path.parent.name.lower() == website_lower:
        return site_path.parent.parent.name
    return "unknown"


def iter_task_artifacts(root: Path | str) -> list[PddlArtifact]:
    root = Path(root)
    artifacts: list[PddlArtifact] = []
    for site_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        for task_dir in sorted(path for path in site_dir.iterdir() if path.is_dir() and not path.name.startswith("_")):
            artifacts.append(PddlArtifact(task_dir))
    return artifacts
