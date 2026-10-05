"""Initialize a fresh volume once, then run a single non-root research worker."""

import os
import shutil
import sys
from pathlib import Path

DATA_FILES = ("assessments.ndjson", "results.sqlite3", "prompt_revisions.sqlite3", "x_outbox.sqlite3")


def initialize_data(data_dir: Path, seed_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        source, target = seed_dir / name, data_dir / name
        if source.is_file() and not target.exists():
            temporary = target.with_suffix(target.suffix + ".initializing")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)


def main() -> None:
    data_dir = Path(os.environ.get("PLAGUESHIELD_DATA_DIR", "/data"))
    initialize_data(data_dir, Path("/opt/plagueshield-seed"))
    if os.getuid() == 0:
        os.chown(data_dir, 10001, 10001)
        for name in DATA_FILES:
            target = data_dir / name
            if target.exists():
                os.chown(target, 10001, 10001)
        os.setgroups([])
        os.setgid(10001)
        os.setuid(10001)
    os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "server.app:app",
                            "--host", "0.0.0.0", "--port", "8080", "--workers", "1",
                            "--timeout-graceful-shutdown", "20"])


if __name__ == "__main__":
    main()
