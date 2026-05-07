from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def add_webarena_to_path(webarena_root: Path | str) -> None:
    root = str(Path(webarena_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def create_env(
    webarena_root: Path | str,
    viewport_width: int = 1920,
    viewport_height: int = 1080,
    sleep_after_execution: float = 0.75,
) -> Any:
    add_webarena_to_path(webarena_root)
    from browser_env import ScriptBrowserEnv

    return ScriptBrowserEnv(
        headless=True,
        observation_type="accessibility_tree",
        current_viewport_only=True,
        viewport_size={"width": viewport_width, "height": viewport_height},
        sleep_after_execution=sleep_after_execution,
    )


def create_env_action(webarena_root: Path | str, action_str: str) -> Any:
    add_webarena_to_path(webarena_root)
    from browser_env import create_id_based_action

    return create_id_based_action(action_str)
