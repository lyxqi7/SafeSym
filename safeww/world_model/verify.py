from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from safeww.pddl.ast import Expr, Problem
from safeww.pddl.parser import parse_domain
from safeww.pddl.serializer import problem_to_pddl
from safeww.planning.fast_downward import solve_pddl


@dataclass
class VerificationIssue:
    level: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerminalReachability:
    terminal_page: str
    canonical_page: str | None
    solved: bool
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class CanonicalVerificationReport:
    artifact_dir: Path
    ok: bool
    domain_name: str | None = None
    pages: int = 0
    actions: int = 0
    predicates: int = 0
    issues: list[VerificationIssue] = field(default_factory=list)
    terminal_reachability: list[TerminalReachability] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "ok": self.ok,
            "domain_name": self.domain_name,
            "pages": self.pages,
            "actions": self.actions,
            "predicates": self.predicates,
            "issues": [
                {"level": i.level, "message": i.message, "detail": i.detail}
                for i in self.issues
            ],
            "terminal_reachability": [
                {
                    "terminal_page": item.terminal_page,
                    "canonical_page": item.canonical_page,
                    "solved": item.solved,
                    "returncode": item.returncode,
                    "stdout_tail": item.stdout_tail,
                    "stderr_tail": item.stderr_tail,
                }
                for item in self.terminal_reachability
            ],
        }


def find_canonical_artifact_dirs(root: Path) -> list[Path]:
    if (root / "canonical_domain.pddl").exists() and (
        root / "canonical_manifest.json"
    ).exists():
        return [root]
    return sorted(
        path.parent
        for path in root.rglob("canonical_domain.pddl")
        if (path.parent / "canonical_manifest.json").exists()
    )


def verify_canonical_artifact(
    artifact_dir: Path,
    *,
    fast_downward: Path | None = None,
    smoke_dir: Path | None = None,
    search: str = "astar(lmcut())",
) -> CanonicalVerificationReport:
    artifact_dir = Path(artifact_dir)
    domain_file = artifact_dir / "canonical_domain.pddl"
    manifest_file = artifact_dir / "canonical_manifest.json"
    issues: list[VerificationIssue] = []

    if not domain_file.exists():
        issues.append(VerificationIssue("error", "missing canonical_domain.pddl"))
        return CanonicalVerificationReport(artifact_dir, ok=False, issues=issues)
    if not manifest_file.exists():
        issues.append(VerificationIssue("error", "missing canonical_manifest.json"))
        return CanonicalVerificationReport(artifact_dir, ok=False, issues=issues)

    try:
        domain = parse_domain(domain_file.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive detail path
        issues.append(
            VerificationIssue("error", "domain parse failed", {"error": str(exc)})
        )
        return CanonicalVerificationReport(artifact_dir, ok=False, issues=issues)

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(
            VerificationIssue("error", "manifest parse failed", {"error": str(exc)})
        )
        return CanonicalVerificationReport(artifact_dir, ok=False, issues=issues)

    report = CanonicalVerificationReport(
        artifact_dir=artifact_dir,
        ok=True,
        domain_name=domain.name,
        pages=len(domain.constants.get("page", [])),
        actions=len(domain.actions),
        predicates=len(domain.predicates),
        issues=issues,
    )

    _check_manifest_consistency(domain, manifest, report)

    if fast_downward:
        _check_terminal_reachability(
            domain_file,
            manifest,
            report,
            fast_downward=Path(fast_downward),
            smoke_dir=smoke_dir or (artifact_dir / "_verify_smoke"),
            search=search,
        )

    report.ok = not any(issue.level == "error" for issue in report.issues) and all(
        item.solved for item in report.terminal_reachability
    )
    return report


def _check_manifest_consistency(domain, manifest: dict[str, Any], report) -> None:
    if manifest.get("domain_name") != domain.name:
        report.issues.append(
            VerificationIssue(
                "error",
                "manifest domain_name does not match PDDL domain",
                {"manifest": manifest.get("domain_name"), "domain": domain.name},
            )
        )

    page_constants = set(domain.constants.get("page", []))
    manifest_pages = {
        item.get("canonical_name") for item in manifest.get("pages", []) if item
    }
    missing_pages = sorted(page for page in manifest_pages if page not in page_constants)
    extra_pages = sorted(page for page in page_constants if page not in manifest_pages)
    if missing_pages:
        report.issues.append(
            VerificationIssue(
                "error", "manifest pages missing from domain constants", {"pages": missing_pages}
            )
        )
    if extra_pages:
        report.issues.append(
            VerificationIssue(
                "warning", "domain page constants not listed in manifest", {"pages": extra_pages}
            )
        )

    predicate_names = {predicate.name for predicate in domain.predicates}
    manifest_predicates = {
        item.get("canonical_name") for item in manifest.get("predicates", []) if item
    }
    missing_predicates = sorted(
        pred for pred in manifest_predicates if pred not in predicate_names
    )
    if missing_predicates:
        report.issues.append(
            VerificationIssue(
                "error",
                "manifest predicates missing from domain predicates",
                {"predicates": missing_predicates},
            )
        )

    duplicate_predicates = sorted(
        name
        for name in predicate_names
        if sum(1 for predicate in domain.predicates if predicate.name == name) > 1
    )
    if duplicate_predicates:
        report.issues.append(
            VerificationIssue(
                "error", "duplicate domain predicate declarations", {"predicates": duplicate_predicates}
            )
        )

    manifest_actions = {
        item.get("canonical_name") for item in manifest.get("actions", []) if item
    }
    missing_actions = sorted(
        action for action in manifest_actions if action not in domain.actions
    )
    extra_actions = sorted(
        action for action in domain.actions if action not in manifest_actions
    )
    if missing_actions:
        report.issues.append(
            VerificationIssue(
                "error", "manifest actions missing from domain", {"actions": missing_actions}
            )
        )
    if extra_actions:
        report.issues.append(
            VerificationIssue(
                "warning", "domain actions not listed in manifest", {"actions": extra_actions}
            )
        )

    if manifest.get("unsupported_conditions"):
        report.issues.append(
            VerificationIssue(
                "error",
                "manifest reports unsupported FSM conditions",
                {"count": len(manifest["unsupported_conditions"])},
            )
        )
    if manifest.get("unsupported_effects"):
        report.issues.append(
            VerificationIssue(
                "error",
                "manifest reports unsupported FSM effects",
                {"count": len(manifest["unsupported_effects"])},
            )
        )


def _check_terminal_reachability(
    domain_file: Path,
    manifest: dict[str, Any],
    report: CanonicalVerificationReport,
    *,
    fast_downward: Path,
    smoke_dir: Path,
    search: str,
) -> None:
    initial_page = manifest.get("initial_page")
    domain_name = manifest.get("domain_name")
    if not initial_page or not domain_name:
        report.issues.append(
            VerificationIssue(
                "error",
                "manifest missing initial_page or domain_name for smoke solve",
            )
        )
        return

    smoke_dir.mkdir(parents=True, exist_ok=True)
    for terminal in manifest.get("terminal_pages", []):
        raw_id = str(terminal.get("raw_id"))
        canonical_page = terminal.get("canonical_name")
        if not canonical_page:
            report.terminal_reachability.append(
                TerminalReachability(raw_id, None, solved=False)
            )
            continue

        problem_file = smoke_dir / f"problem_{canonical_page}.pddl"
        output_plan = smoke_dir / f"sas_plan_{canonical_page}"
        problem = Problem(
            name=f"reach_{canonical_page}",
            init=Expr("and", [Expr("atom", ["at", initial_page])]),
            goal=Expr("and", [Expr("atom", ["at", canonical_page])]),
        )
        problem_file.write_text(
            problem_to_pddl(problem, domain_name=domain_name), encoding="utf-8"
        )
        result = solve_pddl(
            fast_downward,
            domain_file,
            problem_file,
            output_plan,
            search=search,
            cwd=smoke_dir,
        )
        solved = result.returncode == 0 and output_plan.exists()
        report.terminal_reachability.append(
            TerminalReachability(
                terminal_page=raw_id,
                canonical_page=canonical_page,
                solved=solved,
                returncode=result.returncode,
                stdout_tail=(result.stdout or "")[-1000:],
                stderr_tail=(result.stderr or "")[-1000:],
            )
        )
