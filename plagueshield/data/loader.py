"""Loads bundled de-identified case records."""

from __future__ import annotations

import json
import pathlib

from ..models import CaseRecord

CASES_DIR = pathlib.Path(__file__).resolve().parent / "cases"
PUBLIC_DIR = pathlib.Path(__file__).resolve().parent / "public"


def _case_files() -> list[pathlib.Path]:
    if not CASES_DIR.exists():
        return []
    return sorted(p for p in CASES_DIR.glob("*.json") if p.name != "index.json")


def available_cases() -> list[dict]:
    """Lightweight index for listing in a UI without parsing every record."""
    index = CASES_DIR / "index.json"
    if index.exists():
        return json.loads(index.read_text())
    return [
        {
            "case_id": data["case_id"],
            "label": data["label"],
            "origin": data["provenance"]["origin"],
            "country": data.get("country"),
            "file": path.name,
        }
        for path in _case_files()
        for data in [json.loads(path.read_text())]
    ]


def load_case(case_id: str) -> CaseRecord:
    path = CASES_DIR / f"{case_id}.json"
    if not path.exists():
        for candidate in _case_files():
            data = json.loads(candidate.read_text())
            if data.get("case_id") == case_id:
                return CaseRecord.model_validate(data)
        raise FileNotFoundError(f"No bundled case with id {case_id!r}")
    return CaseRecord.model_validate(json.loads(path.read_text()))


def load_all_cases() -> list[CaseRecord]:
    return [
        CaseRecord.model_validate(json.loads(p.read_text())) for p in _case_files()
    ]


def load_public_snapshot(name: str) -> dict | None:
    """Reads a snapshot written by scripts/fetch_public_data.py, if present."""
    path = PUBLIC_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
