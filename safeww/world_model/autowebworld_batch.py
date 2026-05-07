from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SITE_SPLITS = [
    0,
    1153,
    1840,
    2329,
    3331,
    4269,
    4295,
    4728,
    4923,
    5211,
    5466,
    5614,
    5671,
    5858,
    6371,
    7115,
    7167,
    7189,
    7383,
    8440,
    8661,
    9084,
    9693,
    10351,
    11191,
    11707,
    12558,
]


SITE_RELATIVE_WEB_PATHS: list[str | None] = [
    "Travel_Real_Estate/skyscanner/web",
    "Media_Learning_Wellness/youtube/web",
    "Media_Learning_Wellness/coursera/web",
    "Media_Learning_Wellness/headspace/web",
    "Media_Learning_Wellness/health/web",
    None,
    "Commerce_Finance/revolut/web",
    "Commerce_Finance/aliexpress/web",
    "Commerce_Finance/JD_COM/web",
    None,
    "Communication_Social/facebook/web",
    "Communication_Social/medium/web",
    "Communication_Social/microsoft_teams/web",
    "Communication_Social/outlook/web",
    "Communication_Social/quora/web",
    "Communication_Social/twitter/web",
    "Communication_Social/zoom/web",
    "Communication_Social/signal/web",
    "Communication_Social/slack/web",
    "Productivity_Dev_Ops/airtable/web",
    "Productivity_Dev_Ops/asana/web",
    "Productivity_Dev_Ops/bitbucket/web",
    "Productivity_Dev_Ops/freshdesk/web",
    "Productivity_Dev_Ops/github/web",
    "Productivity_Dev_Ops/onenote/web",
    "Productivity_Dev_Ops/Optimizely/web",
]


@dataclass(frozen=True)
class AutoWebWorldSiteBatch:
    index: int
    start: int
    end: int
    site_type: str
    website: str
    site_path: Path
    items: list[dict[str, Any]]


@dataclass(frozen=True)
class AutoWebWorldTaskGroup:
    task_index: int
    instruction: str
    items: list[dict[str, Any]]

    @property
    def representative(self) -> dict[str, Any]:
        return self.items[0]


def iter_site_batches(
    data: list[dict[str, Any]], sites_root: Path
) -> list[AutoWebWorldSiteBatch]:
    batches: list[AutoWebWorldSiteBatch] = []
    for index, relative_path in enumerate(SITE_RELATIVE_WEB_PATHS):
        if relative_path is None:
            continue
        start = SITE_SPLITS[index]
        end = SITE_SPLITS[index + 1] if index + 1 < len(SITE_SPLITS) else len(data)
        site_path = sites_root / relative_path
        website = site_path.parent.name
        site_type = site_path.parent.parent.name
        batches.append(
            AutoWebWorldSiteBatch(
                index=index,
                start=start,
                end=end,
                site_type=site_type,
                website=website,
                site_path=site_path,
                items=data[start:end],
            )
        )
    return batches


def group_tasks_by_instruction(items: list[dict[str, Any]]) -> list[AutoWebWorldTaskGroup]:
    groups: list[AutoWebWorldTaskGroup] = []
    current: list[dict[str, Any]] = []
    for item in items:
        if not current or item.get("instruction") == current[0].get("instruction"):
            current.append(item)
        else:
            groups.append(
                AutoWebWorldTaskGroup(
                    task_index=len(groups),
                    instruction=str(current[0].get("instruction", "")),
                    items=current,
                )
            )
            current = [item]
    if current:
        groups.append(
            AutoWebWorldTaskGroup(
                task_index=len(groups),
                instruction=str(current[0].get("instruction", "")),
                items=current,
            )
        )
    return groups


def build_task_metadata(
    *,
    site: AutoWebWorldSiteBatch,
    task: AutoWebWorldTaskGroup,
    status: str,
    attempt_index: int,
    final_feedback: str,
) -> dict[str, Any]:
    representative = task.representative
    return {
        "task_index": task.task_index,
        "img_filename": representative.get("img_filename"),
        "instruction": task.instruction,
        "website": site.website,
        "site_type": site.site_type,
        "site_path": str(site.site_path),
        "generation_pipeline": "agentic_world_model_v1",
        "generation_status": status,
        "generation_attempt_index": attempt_index,
        "generation_final_feedback": final_feedback,
        "source_site_split_index": site.index,
        "source_data_start": site.start,
        "source_data_end": site.end,
        "source_group_size": len(task.items),
    }
