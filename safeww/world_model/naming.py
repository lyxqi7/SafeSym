from __future__ import annotations

import re


_NON_SYMBOL_CHARS = re.compile(r"[^a-z0-9_]+")
_REPEATED_UNDERSCORES = re.compile(r"_+")


def normalize_symbol(value: object, *, prefix: str = "sym") -> str:
    """Convert external FSM identifiers into stable PDDL-safe symbols."""

    text = str(value or "").strip().lower()
    text = text.replace("$.", "")
    text = text.replace("$", "")
    text = text.replace("[*]", "_any")
    text = text.replace(".", "_")
    text = text.replace("-", "_")
    text = _NON_SYMBOL_CHARS.sub("_", text)
    text = _REPEATED_UNDERSCORES.sub("_", text).strip("_")

    if not text:
        text = prefix
    if text[0].isdigit():
        text = f"{prefix}_{text}"
    return text


def unique_symbol(base: str, used: set[str]) -> str:
    symbol = base
    idx = 2
    while symbol in used:
        symbol = f"{base}_{idx}"
        idx += 1
    used.add(symbol)
    return symbol
