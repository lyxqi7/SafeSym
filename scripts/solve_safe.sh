#!/usr/bin/env bash
set -euo pipefail

python -m safeww.cli.solve \
  --pddl-root "${PDDL_ROOT:?set PDDL_ROOT}" \
  --fast-downward "${FAST_DOWNWARD:?set FAST_DOWNWARD}" \
  --safe

