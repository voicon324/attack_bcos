from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


CODE_DATASET = "attack-bcos-github"
CODE_OWNER = "hkhnhduy"
DEFAULT_GITHUB_REPO = "https://github.com/voicon324/attack_bcos.git"
DEFAULT_GITHUB_REF = "main"
EMBEDDED_JOB_CONFIG = None


def load_job_config() -> dict:
    if EMBEDDED_JOB_CONFIG is not None:
        return dict(EMBEDDED_JOB_CONFIG)
    config_path = Path("job_config.json")
    if not config_path.is_file():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


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
    return clone_repo(load_job_config())


def clone_repo(config: dict) -> Path:
    repo_url = str(config.get("github_repo", DEFAULT_GITHUB_REPO))
    git_ref = str(config.get("github_ref", DEFAULT_GITHUB_REF))
    work_repo = Path("/kaggle/working/attack_bcos")
    if work_repo.exists():
        shutil.rmtree(work_repo)
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", git_ref, repo_url, str(work_repo)])
    except subprocess.CalledProcessError:
        if work_repo.exists():
            shutil.rmtree(work_repo)
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(work_repo)])
        subprocess.check_call(["git", "fetch", "--depth", "1", "origin", git_ref], cwd=str(work_repo))
        subprocess.check_call(["git", "checkout", "FETCH_HEAD"], cwd=str(work_repo))
    return work_repo


def main() -> None:
    job_config = load_job_config()
    if job_config:
        Path("/kaggle/working/job_config.json").write_text(json.dumps(job_config, indent=2) + "\n")
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
    config = Path("/kaggle/working/job_config.json").resolve()
    subprocess.check_call([sys.executable, "-u", str(runner), "--config", str(config)], cwd=str(work_repo))


if __name__ == "__main__":
    main()
