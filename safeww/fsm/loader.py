from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FsmDocument:
    path: Path
    raw: dict[str, Any]

    @property
    def app_name(self) -> str:
        return self.raw.get("meta", {}).get("app", self.path.parent.name)

    @property
    def initial_page_id(self) -> str | None:
        return self.raw.get("meta", {}).get("initial_page_id")

    @property
    def pages(self) -> list[dict[str, Any]]:
        return self.raw.get("pages", [])

    @property
    def actions(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for page in self.pages:
            result.extend(page.get("actions", []))
        return result


def load_fsm(path: Path | str) -> FsmDocument:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return FsmDocument(path=path, raw=json.load(f))

