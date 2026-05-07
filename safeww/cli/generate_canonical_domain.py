from __future__ import annotations

import argparse
import json
from pathlib import Path

from safeww.world_model import build_canonical_domain


def write_result(fsm_path: Path, output_dir: Path, *, domain_name: str | None = None) -> dict:
    result = build_canonical_domain(fsm_path, domain_name=domain_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "canonical_domain.pddl").write_text(result.to_pddl(), encoding="utf-8")
    (output_dir / "canonical_manifest.json").write_text(
        json.dumps(result.manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "fsm_path": str(fsm_path),
        "output_dir": str(output_dir),
        "domain_name": result.manifest["domain_name"],
        "pages": len(result.manifest["pages"]),
        "actions": len(result.manifest["actions"]),
        "predicates": len(result.manifest["predicates"]),
        "unsupported_conditions": len(result.manifest["unsupported_conditions"]),
        "unsupported_effects": len(result.manifest["unsupported_effects"]),
    }


def iter_site_fsms(sites_root: Path) -> list[tuple[Path, Path]]:
    fsms: list[tuple[Path, Path]] = []
    for category_dir in sorted(p for p in sites_root.iterdir() if p.is_dir()):
        for site_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            fsm_path = site_dir / "web" / "fsm.json"
            if fsm_path.exists():
                fsms.append((fsm_path, Path(category_dir.name) / site_dir.name))
    return fsms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic canonical full-site PDDL domains from AutoWebWorld FSM files."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fsm", type=Path, help="Path to one FSM JSON file.")
    source.add_argument(
        "--sites-root",
        type=Path,
        help="Root containing category/site/web/fsm.json directories.",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory for --fsm.")
    parser.add_argument("--output-root", type=Path, help="Output root for --sites-root.")
    parser.add_argument("--domain-name", help="Optional domain name for a single FSM.")
    parser.add_argument(
        "--summary-file",
        type=Path,
        help="Optional JSON summary path for batch generation.",
    )
    args = parser.parse_args()

    if args.fsm:
        if not args.output_dir:
            parser.error("--output-dir is required with --fsm")
        summary = [write_result(args.fsm, args.output_dir, domain_name=args.domain_name)]
    else:
        if not args.output_root:
            parser.error("--output-root is required with --sites-root")
        if args.domain_name:
            parser.error("--domain-name is only supported with --fsm")
        summary = [
            write_result(fsm_path, args.output_root / relative_dir)
            for fsm_path, relative_dir in iter_site_fsms(args.sites_root)
        ]

    if args.summary_file:
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        args.summary_file.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    total_actions = sum(item["actions"] for item in summary)
    total_pages = sum(item["pages"] for item in summary)
    print(
        f"Generated {len(summary)} canonical domain(s): "
        f"{total_pages} pages, {total_actions} actions."
    )
    for item in summary:
        print(
            f"- {item['domain_name']}: {item['pages']} pages, {item['actions']} actions -> {item['output_dir']}"
        )


if __name__ == "__main__":
    main()
