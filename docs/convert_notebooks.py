"""Convert marimo apps in notebooks/ to Markdown for Zensical.

Run this script whenever a marimo app changes:
    python docs/convert_notebooks.py

Check that the committed Markdown exports are current:
    python docs/convert_notebooks.py --check

Produces a .md file for each .py app in docs/notebooks/ using marimo.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Input: marimo apps in repo root
APPS_DIR = Path(__file__).parent.parent / "notebooks"
# Output: Markdown files in docs/notebooks
OUTPUT_DIR = Path(__file__).parent / "notebooks"
REPO_ROOT = APPS_DIR.parent


def convert(app: Path) -> None:
    print(f"Converting {app.name} ...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "md",
            str(app),
            "--output",
            str(OUTPUT_DIR / f"{app.stem}.md"),
            "--force",
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if conversion changes a committed Markdown export",
    )
    args = parser.parse_args()

    apps = sorted(APPS_DIR.glob("*.py"))
    if not apps:
        print("No marimo apps found in", APPS_DIR)
        return 0

    for app in apps:
        convert(app)

    if args.check:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--exit-code",
                "--",
                str(OUTPUT_DIR.relative_to(REPO_ROOT)),
            ],
            check=False,
            cwd=REPO_ROOT,
        )
        if result.returncode:
            print(
                "Notebook Markdown exports are stale. "
                "Run `python docs/convert_notebooks.py` and commit the changes."
            )
            return result.returncode

    print("\nDone. Re-run this script whenever a marimo app changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
