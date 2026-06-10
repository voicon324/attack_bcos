from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


CODE_DATASET = "attack-bcos-github"
CODE_OWNER = "hkhnhduy"


def find_repo() -> Path:
    candidates = [
        Path("/kaggle/input") / CODE_DATASET,
        Path("/kaggle/input/datasets") / CODE_OWNER / CODE_DATASET,
        Path("/kaggle/input/datasets") / CODE_DATASET,
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / "kaggle" / "camopatch_job" / "run_camopatch_job.py").is_file():
            return candidate

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for candidate in sorted(kaggle_input.rglob("kaggle/camopatch_job/run_camopatch_job.py")):
            return candidate.parents[2]
    raise FileNotFoundError("Could not find attack-bcos-github code dataset.")


def main() -> None:
    repo = find_repo()
    work_repo = Path("/kaggle/working/attack_bcos")
    if repo.resolve() != work_repo.resolve():
        if work_repo.exists():
            shutil.rmtree(work_repo)
        shutil.copytree(
            repo,
            work_repo,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "artifacts", "kaggle_runs"),
        )
    runner = work_repo / "kaggle" / "camopatch_job" / "run_camopatch_job.py"
    config = Path("job_config.json").resolve()
    subprocess.check_call([sys.executable, "-u", str(runner), "--config", str(config)])


if __name__ == "__main__":
    main()
