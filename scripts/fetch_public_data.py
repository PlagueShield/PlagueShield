"""Fetches public plague data snapshots into plagueshield/data/public/.

Run: python scripts/fetch_public_data.py

Safe to run offline -- failures are recorded in the snapshot rather than
raised, so the system keeps working with whatever it already has.
"""

from plagueshield.data.public_sources import fetch_all, save_snapshots


def main() -> None:
    results = fetch_all()
    for name, result in results.items():
        status = "OK " if result.ok else "FAIL"
        print(f"[{status}] {name}: {result.note}")
    for path in save_snapshots(results):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
