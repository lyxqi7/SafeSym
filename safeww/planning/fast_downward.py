from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def solve_pddl(
    fast_downward: Path | str,
    domain_file: Path | str,
    problem_file: Path | str,
    output_plan: Path | str,
    search: str = "astar(lmcut())",
    cwd: Path | str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run Fast Downward without clobbering an existing plan in the task dir."""

    fast_downward = Path(fast_downward).resolve()
    domain_file = Path(domain_file).resolve()
    problem_file = Path(problem_file).resolve()
    output_plan = Path(output_plan).resolve()
    temp_parent = Path(cwd).resolve() if cwd else output_plan.parent
    temp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="safeww_fd_", dir=temp_parent) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        temp_domain = temp_dir / domain_file.name
        temp_problem = temp_dir / problem_file.name
        shutil.copy2(domain_file, temp_domain)
        shutil.copy2(problem_file, temp_problem)

        result = subprocess.run(
            [
                "python",
                str(fast_downward),
                str(temp_domain),
                str(temp_problem),
                "--search",
                search,
            ],
            cwd=temp_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        generated = temp_dir / "sas_plan"
        if result.returncode == 0 and generated.exists():
            output_plan.parent.mkdir(parents=True, exist_ok=True)
            if output_plan.exists():
                output_plan.unlink()
            shutil.move(str(generated), str(output_plan))

        return result
