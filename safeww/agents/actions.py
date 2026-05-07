from __future__ import annotations

import re
from dataclasses import dataclass


VALID_ACTIONS = {"click", "type", "press", "hover", "scroll", "check", "success", "fail"}
VALID_CHECK_TYPES = {
    "permission_gate",
    "information_verification",
    "privacy_sanitization",
    "content_moderation",
    "human_confirmation",
}


@dataclass(frozen=True)
class AgentAction:
    kind: str
    raw: str
    element_id: int | None = None
    text: str | None = None
    check_type: str | None = None
    reason: str | None = None
    target_hint: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.kind in {"success", "fail"}

    @property
    def is_check(self) -> bool:
        return self.kind == "check"


def clean_action_text(action_text: str | None) -> str | None:
    if action_text is None:
        return None
    for raw_line in action_text.strip().splitlines():
        line = normalize_action_line(raw_line)
        if not line:
            continue
        line = normalize_common_format_errors(line)
        prefix = line.split("[", 1)[0].strip()
        if not prefix:
            continue
        parts = prefix.split()
        if not parts:
            continue
        kind = parts[0]
        if kind in VALID_ACTIONS:
            return line
    return None


def normalize_action_line(line: str) -> str:
    line = line.strip()
    if not line or line in {"```", "```text", "```python", "```json"}:
        return ""
    line = re.sub(r"^[-*]\s+", "", line)
    line = re.sub(r"^(?:action|next action|output)\s*:\s*", "", line, flags=re.IGNORECASE)
    line = line.strip("`").strip()
    return line


def normalize_common_format_errors(line: str) -> str:
    """Fix frequent LLM formatting slips while preserving strict action types."""

    match = re.fullmatch(r"(click|hover)\s+(\d+)", line)
    if match:
        return f"{match.group(1)} [{match.group(2)}]"

    match = re.fullmatch(r"scroll\s+(down|up)", line)
    if match:
        return f"scroll [{match.group(1)}]"

    match = re.fullmatch(r"press\s+(.+)", line)
    if match and "[" not in line:
        return f"press [{match.group(1).strip()}]"

    return line


def normalize_label(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = re.sub(r"^[\"']|[\"']$", "", normalized)
    return normalized


def build_observation_label_index(obs_text: str) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    priority_order = {"button": 0, "link": 1, "tab": 2, "menuitem": 3, "img": 4}
    candidates: list[tuple[int, int, str]] = []

    for line in obs_text.splitlines():
        match = re.match(r"\s*\[(\d+)\]\s+([A-Za-z_]+).*?'([^']+)'", line)
        if not match:
            continue
        element_id = int(match.group(1))
        role = match.group(2).lower()
        label = normalize_label(match.group(3))
        if not label:
            continue
        candidates.append((priority_order.get(role, 10), element_id, label))

    for _, element_id, label in sorted(candidates):
        index.setdefault(label, []).append(element_id)
    return index


def resolve_textual_target(target_hint: str, obs_text: str) -> int | None:
    target = normalize_label(target_hint)
    if not target:
        return None

    index = build_observation_label_index(obs_text)
    if target in index:
        return index[target][0]

    for label, ids in index.items():
        if label == target:
            return ids[0]
        if label.endswith(target) or target.endswith(label):
            return ids[0]
    return None


def parse_agent_action(action_text: str | None, obs_text: str | None = None) -> AgentAction | None:
    cleaned = clean_action_text(action_text)
    if cleaned is None:
        return None

    kind = cleaned.split("[", 1)[0].strip().split()[0]
    element_match = re.search(r"\[(\d+)\]", cleaned)
    bracket_values = re.findall(r"\[(.*?)\]", cleaned)
    target_hint = bracket_values[0] if bracket_values else None
    element_id = int(element_match.group(1)) if element_match else None

    if kind == "check":
        if not bracket_values:
            return None
        if len(bracket_values) >= 2:
            check_type = normalize_label(bracket_values[0]).replace(" ", "_").replace("-", "_")
            if check_type not in VALID_CHECK_TYPES:
                return None
            return AgentAction(
                kind=kind,
                raw=cleaned,
                check_type=check_type,
                reason=bracket_values[1],
                target_hint=target_hint,
            )
        return AgentAction(
            kind=kind,
            raw=cleaned,
            reason=bracket_values[0],
            target_hint=target_hint,
        )

    if kind == "type" and len(bracket_values) < 2:
        return None

    if element_id is None and obs_text and kind in {"click", "hover", "type"} and target_hint:
        element_id = resolve_textual_target(target_hint, obs_text)
        if element_id is not None:
            if kind == "type" and len(bracket_values) > 1:
                cleaned = f"type [{element_id}] [{bracket_values[1]}]"
            else:
                cleaned = f"{kind} [{element_id}]"

    return AgentAction(
        kind=kind,
        raw=cleaned,
        element_id=element_id,
        text=bracket_values[1] if kind == "type" and len(bracket_values) > 1 else None,
        check_type=None,
        reason=bracket_values[0] if kind in {"check", "success", "fail"} and bracket_values else None,
        target_hint=target_hint,
    )
