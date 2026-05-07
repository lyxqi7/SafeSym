#!/usr/bin/env bash
set -euo pipefail

python -m safeww.cli.inject_safety \
  --pddl-root "${PDDL_ROOT:?set PDDL_ROOT}" \
  --rules "${SAFETY_RULES:-configs/constraint_rules.json}"
