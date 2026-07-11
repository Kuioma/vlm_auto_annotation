import shutil
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_root_artifact_ignore_does_not_hide_source_packages(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text(
        (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    paths = {
        "runtime": Path("artifacts/run-1/video-1/final.json"),
        "source": Path("src/auto_annotation/artifacts/store.py"),
        "test": Path("tests/artifacts/test_store_export.py"),
    }
    for path in paths.values():
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("placeholder", encoding="utf-8")

    def check_ignore(path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", str(path)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

    assert check_ignore(paths["runtime"]).returncode == 0
    assert check_ignore(paths["source"]).returncode == 1
    assert check_ignore(paths["test"]).returncode == 1
