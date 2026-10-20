"""Launch the desktop attendance app with its Conda environment."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
APP_SCRIPT = PROJECT_DIR / "attendance_system_with_blink.py"
CONDA_ENV_NAME = "attendance"


def find_conda_executable():
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        str(Path.home() / "miniconda3" / "Scripts" / "conda.exe"),
        str(Path.home() / "anaconda3" / "Scripts" / "conda.exe"),
        str(Path(sys.prefix) / "Scripts" / "conda.exe"),
    ]

    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs":
        candidates.append(str(prefix.parent.parent / "Scripts" / "conda.exe"))

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def main():
    if not APP_SCRIPT.is_file():
        print(f"Could not find the application: {APP_SCRIPT}", file=sys.stderr)
        return 1

    conda_executable = find_conda_executable()
    if conda_executable is None:
        print(
            "Could not find Conda. Install Miniconda/Anaconda or add Conda to PATH.",
            file=sys.stderr,
        )
        return 1

    if not (os.environ.get("ATTENDANCE_ADMIN_USERNAME") and
            os.environ.get("ATTENDANCE_ADMIN_PASSWORD")):
        if not (PROJECT_DIR / ".env").is_file():
            print(
                "Configure ATTENDANCE_ADMIN_USERNAME and "
                "ATTENDANCE_ADMIN_PASSWORD in a local .env file first.",
                file=sys.stderr,
            )
            print("Use .env.example as a template; do not commit .env.", file=sys.stderr)
            return 1

    command = [
        conda_executable,
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV_NAME,
        "python",
        str(APP_SCRIPT),
    ]
    return subprocess.call(command, cwd=PROJECT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
