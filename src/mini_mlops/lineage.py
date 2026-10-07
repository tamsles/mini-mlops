import hashlib
import subprocess
from pathlib import Path


def get_git_commit() -> str:
    """Return the current Git commit SHA."""

    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def is_git_dirty() -> bool:
    """Return whether the working tree has uncommitted changes."""

    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return bool(result.stdout.strip())


def sha256_file(path: Path) -> str:
    """Compute SHA256 fingerprint of a file."""

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()