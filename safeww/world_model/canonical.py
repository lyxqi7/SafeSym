from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safeww.fsm.loader import FsmDocument, load_fsm
from safeww.pddl.ast import Action, Domain, Expr, Predicate
from safeww.pddl.serializer import domain_to_pddl

from .naming import normalize_symbol, unique_symbol


@dataclass(frozen=True)
class CanonicalBuildResult:
    domain: Domain
    manifest: dict[str, Any]

    def to_pddl(self) -> str:
        return domain_to_pddl(self.domain)


def atom(name: str, *args: str) -> Expr:
    return Expr("atom", [name, *args])


def neg(expr: Expr) -> Expr:
    return Expr("not", [expr])


def conj(exprs: list[Expr]) -> Expr:
    if not exprs:
        return Expr("and", [])
    return Expr("and", exprs)


def predicate_for_path(path: str) -> str:
    return f"state_{normalize_symbol(path, prefix='state')}"


def predicate_for_path_value(path: str, value: object) -> str:
    value_symbol = normalize_symbol(value, prefix="value")
    return f"{predicate_for_path(path)}_is_{value_symbol}"


def _is_dynamic_value(value: object) -> bool:
    return isinstance(value, str) and value.startswith("{") and value.endswith("}")


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _is_true(value: object) -> bool:
    return _is_bool(value) and value is True


def _is_false(value: object) -> bool:
    return _is_bool(value) and value is False


def _needs_value_predicate(value: object) -> bool:
    return value is not None and not _is_bool(value) and not _is_dynamic_value(value)


def canonical_state_path(path: object) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if text.startswith("$."):
        return text
    if text.startswith("$"):
        return f"$.{text[1:].lstrip('.')}"
    return f"$.{text}"


class CanonicalDomainBuilder:
    def __init__(self, fsm: FsmDocument, *, domain_name: str | None = None):
        self.fsm = fsm
        raw_domain = domain_name or f"canonical_{fsm.path.parent.parent.name}"
        self.domain_name = normalize_symbol(raw_domain, prefix="domain")
        self.used_actions: set[str] = set()
        self.page_symbols: dict[str, str] = {}
        self.path_predicates: dict[str, str] = {}
        self.value_predicates: dict[tuple[str, str], str] = {}
        self.unsupported_conditions: list[dict[str, Any]] = []
        self.unsupported_effects: list[dict[str, Any]] = []

    def build(self) -> CanonicalBuildResult:
        pages = self._collect_pages()
        actions = self._collect_actions()
        predicates = self._collect_predicates()

        domain = Domain(
            name=self.domain_name,
            requirements={":strips", ":typing", ":negative-preconditions"},
            types=["page"],
            constants={"page": [pages[page_id] for page_id in sorted(pages)]},
            predicates=predicates,
            actions={action.name: action for action in actions},
        )

        manifest = self._build_manifest(actions)
        return CanonicalBuildResult(domain=domain, manifest=manifest)

    def _collect_pages(self) -> dict[str, str]:
        used: set[str] = set()
        for page in self.fsm.pages:
            page_id = str(page.get("id", ""))
            base = normalize_symbol(page_id, prefix="page")
            self.page_symbols[page_id] = unique_symbol(base, used)
        return self.page_symbols

    def _collect_predicates(self) -> list[Predicate]:
        paths: set[str] = set()
        value_preds: set[tuple[str, str]] = set()

        for page in self.fsm.pages:
            for path in (page.get("signature_schema") or {}).keys():
                paths.add(canonical_state_path(path))
            for action in page.get("actions", []):
                for condition in action.get("preconditions", []) or []:
                    path = condition.get("path")
                    if path:
                        paths.add(canonical_state_path(path))
                    if condition.get("cond") == "eq":
                        value = condition.get("value")
                        if _needs_value_predicate(value):
                            value_preds.add((canonical_state_path(path), str(value)))
                for effect in action.get("effects", []) or []:
                    path = effect.get("path")
                    if path:
                        paths.add(canonical_state_path(path))
                    has_value = "value" in effect
                    value = effect.get("value")
                    if (
                        effect.get("op") == "set"
                        and has_value
                        and _needs_value_predicate(value)
                    ):
                        value_preds.add((canonical_state_path(path), str(value)))

        self.path_predicates = {
            path: predicate_for_path(path) for path in sorted(paths) if path
        }
        self.value_predicates = {
            (path, value): predicate_for_path_value(path, value)
            for path, value in sorted(value_preds)
        }

        predicates = [Predicate("at", [("?p", "page")])]
        for name in sorted(self.path_predicates.values()):
            predicates.append(Predicate(name))
        for name in sorted(self.value_predicates.values()):
            predicates.append(Predicate(name))
        return predicates

    def _collect_actions(self) -> list[Action]:
        result: list[Action] = []
        for page in self.fsm.pages:
            for raw_action in page.get("actions", []) or []:
                action_id = str(raw_action.get("id") or raw_action.get("name") or "action")
                base = normalize_symbol(action_id, prefix="action")
                name = unique_symbol(base, self.used_actions)
                result.append(
                    Action(
                        name=name,
                        parameters=[],
                        precondition=self._action_precondition(raw_action),
                        effect=self._action_effect(raw_action),
                    )
                )
        return result

    def _action_precondition(self, raw_action: dict[str, Any]) -> Expr:
        exprs: list[Expr] = []
        from_page = raw_action.get("from")
        if from_page in self.page_symbols:
            exprs.append(atom("at", self.page_symbols[str(from_page)]))

        for condition in raw_action.get("preconditions", []) or []:
            condition_expr = self._condition_expr(condition, raw_action)
            if condition_expr is not None:
                exprs.append(condition_expr)
        return conj(exprs)

    def _condition_expr(
        self, condition: dict[str, Any], raw_action: dict[str, Any]
    ) -> Expr | None:
        path = condition.get("path")
        cond = condition.get("cond")
        value = condition.get("value")
        if not path:
            self.unsupported_conditions.append(
                {"action_id": raw_action.get("id"), "condition": condition, "reason": "missing_path"}
            )
            return None

        path = canonical_state_path(path)
        path_atom = atom(self.path_predicates.setdefault(path, predicate_for_path(path)))

        if cond == "eq":
            if _is_true(value):
                return path_atom
            if _is_false(value) or value is None:
                return neg(path_atom)
            if _is_dynamic_value(value):
                return path_atom
            return atom(predicate_for_path_value(path, value))
        if cond == "neq":
            if _is_true(value):
                return neg(path_atom)
            if _is_false(value) or value is None:
                return path_atom
            if _is_dynamic_value(value):
                return path_atom
            return neg(atom(predicate_for_path_value(path, value)))
        if cond in {"length_gt", "exists", "truthy", "gt", "gte", "lt", "lte"}:
            return path_atom

        self.unsupported_conditions.append(
            {"action_id": raw_action.get("id"), "condition": condition, "reason": "unsupported_cond"}
        )
        return path_atom

    def _action_effect(self, raw_action: dict[str, Any]) -> Expr:
        exprs: list[Expr] = []
        from_page = raw_action.get("from")
        to_page = raw_action.get("to")
        if from_page in self.page_symbols and to_page in self.page_symbols and from_page != to_page:
            exprs.append(neg(atom("at", self.page_symbols[str(from_page)])))
            exprs.append(atom("at", self.page_symbols[str(to_page)]))

        for effect in raw_action.get("effects", []) or []:
            exprs.extend(self._effect_exprs(effect, raw_action))
        if not exprs and from_page in self.page_symbols:
            exprs.append(atom("at", self.page_symbols[str(from_page)]))
        return conj(exprs)

    def _effect_exprs(
        self, effect: dict[str, Any], raw_action: dict[str, Any]
    ) -> list[Expr]:
        path = effect.get("path")
        op = effect.get("op")
        has_value = "value" in effect
        has_value_ref = "value_ref" in effect
        value = effect.get("value")
        if not path:
            self.unsupported_effects.append(
                {"action_id": raw_action.get("id"), "effect": effect, "reason": "missing_path"}
            )
            return []

        path = canonical_state_path(path)
        path_atom = atom(self.path_predicates.setdefault(path, predicate_for_path(path)))
        if op == "clear":
            return [neg(path_atom)]
        if op == "set":
            if has_value and (_is_false(value) or value is None):
                return [neg(path_atom)]
            exprs = [path_atom]
            if (
                has_value
                and _needs_value_predicate(value)
                and not has_value_ref
            ):
                exprs.append(atom(predicate_for_path_value(path, value)))
            return exprs

        self.unsupported_effects.append(
            {"action_id": raw_action.get("id"), "effect": effect, "reason": "unsupported_op"}
        )
        return [path_atom]

    def _build_manifest(self, actions: list[Action]) -> dict[str, Any]:
        action_records: list[dict[str, Any]] = []
        action_by_raw: dict[str, str] = {}
        action_iter = iter(actions)

        for page in self.fsm.pages:
            for raw_action in page.get("actions", []) or []:
                action = next(action_iter)
                raw_id = str(raw_action.get("id") or "")
                if raw_id:
                    action_by_raw[raw_id] = action.name
                action_records.append(
                    {
                        "raw_id": raw_id,
                        "canonical_name": action.name,
                        "raw_name": raw_action.get("name"),
                        "from": raw_action.get("from"),
                        "to": raw_action.get("to"),
                        "canonical_from": self.page_symbols.get(str(raw_action.get("from"))),
                        "canonical_to": self.page_symbols.get(str(raw_action.get("to"))),
                        "is_navigation": bool(raw_action.get("is_navigation", False)),
                    }
                )

        terminal_pages = self.fsm.raw.get("meta", {}).get("terminal_pages", []) or []
        return {
            "schema_version": "canonical-fsm-domain-v1",
            "domain_name": self.domain_name,
            "fsm_path": str(self.fsm.path),
            "app_name": self.fsm.app_name,
            "initial_page_id": self.fsm.initial_page_id,
            "initial_page": self.page_symbols.get(str(self.fsm.initial_page_id)),
            "terminal_pages": [
                {"raw_id": page_id, "canonical_name": self.page_symbols.get(str(page_id))}
                for page_id in terminal_pages
            ],
            "pages": [
                {"raw_id": raw_id, "canonical_name": symbol}
                for raw_id, symbol in sorted(self.page_symbols.items())
            ],
            "predicates": [
                {"path": path, "canonical_name": symbol, "kind": "state_path"}
                for path, symbol in sorted(self.path_predicates.items())
            ]
            + [
                {
                    "path": path,
                    "value": value,
                    "canonical_name": symbol,
                    "kind": "state_value",
                }
                for (path, value), symbol in sorted(self.value_predicates.items())
            ],
            "actions": action_records,
            "action_by_raw_id": action_by_raw,
            "unsupported_conditions": self.unsupported_conditions,
            "unsupported_effects": self.unsupported_effects,
            "notes": [
                "This canonical domain is deterministic and full-site scoped.",
                "Task-specific problem generators should only reference symbols from this manifest.",
            ],
        }


def build_canonical_domain(
    fsm: FsmDocument | str | Path, *, domain_name: str | None = None
) -> CanonicalBuildResult:
    document = load_fsm(fsm) if isinstance(fsm, (str, Path)) else fsm
    return CanonicalDomainBuilder(document, domain_name=domain_name).build()
