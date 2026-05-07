from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved roots for SafeSym and its upstream dependencies."""

    root: Path
    autowebworld_root: Path
    webarena_root: Path
    artifacts_root: Path
    fast_downward: Path

    @classmethod
    def from_repo_root(cls, root: Path | str) -> "ProjectPaths":
        root = Path(root).resolve()
        workspace = root.parent
        return cls(
            root=root,
            autowebworld_root=workspace / "AutoWebWorld",
            webarena_root=workspace / "webarena",
            artifacts_root=root / "artifacts",
            fast_downward=workspace / "AutoWebWorld" / "downward" / "fast-downward.py",
        )
