# SafeSym

This repository contains the core implementation of SafeSym.

## Project Structure

- `safeww/world_model/`: symbolic world-model generation, PDDL I/O,
  verification, canonical-domain utilities, and FSM conversion.
- `safeww/safety/`: safety obligation rules, dynamic constraint refinement,
  task-risk annotation, and safety-constraint injection.
- `safeww/planning/`: planner wrappers and plan parsing utilities.
- `safeww/agents/`: GUI-agent prompts, action parsing, and SafeSym execution
  policies.
- `safeww/eval/`: evaluation runner and safety metric computation.
- `safeww/cli/`: command-line entry points for world-model construction,
  safety injection, planning, refinement, inspection, and evaluation.
- `configs/`: example configuration files and rule libraries.
- `scripts/`: lightweight shell wrappers for safety injection and planning.
