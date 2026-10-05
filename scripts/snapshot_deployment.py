"""Generate a consistent migration seed; stop local workers before running."""

import shutil
import sqlite3
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    source, target = root / "server", root / ".deployment-seed"
    target.mkdir(exist_ok=True)
    if (source / "assessments.ndjson").exists():
        shutil.copyfile(source / "assessments.ndjson", target / "assessments.ndjson")
    for name in ("results.sqlite3", "prompt_revisions.sqlite3", "x_outbox.sqlite3"):
        if (source / name).exists():
            with sqlite3.connect(f"file:{source / name}?mode=ro", uri=True) as original:
                with sqlite3.connect(target / name) as snapshot:
                    original.backup(snapshot)
    print("Migration seed ready; no environment files or credentials included.")


if __name__ == "__main__":
    main()
