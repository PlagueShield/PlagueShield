"""Versioned research-focus revisions, without mutable safety instructions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .llm import ANALYSIS_INSTRUCTIONS

FOCUS_DIRECTIVES = {
    "citation_scope": "For each major finding, connect it to supplied evidence and distinguish general guidance from case-specific support; explicitly mark unsupported claims.",
    "quantitative_effects": "Report available baseline and ablation differences in percentage points, distinguish saturation from robustness, and state which quantities were not computed.",
    "missingness": "Explain how missing assays, susceptibility measurements and provenance limit each conclusion; do not treat missing evidence as negative evidence.",
    "confounding": "Identify potential correlated evidence, duplicated observations and untested independence assumptions; do not claim their effects were measured.",
    "validation_plan": "Propose a concrete held-out validation design with endpoints, comparator, failure criteria and provenance controls; label it as a proposal, not an executed experiment.",
    "cross_iteration": "Distinguish case-specific findings from repeatable methodological limitations; repeat runs on the same case are not independent validation samples.",
    "uncertainty": "Explain abstentions, disagreements and heuristic confidence caps, keeping model estimates distinct from calibrated clinical probabilities.",
}


def baseline() -> dict:
    return {"version": 0, "focus": [], "instructions": ANALYSIS_INSTRUCTIONS,
            "reason": "Original research prompt", "source_iteration": None}


def revised_instructions(focus: list[str]) -> str:
    if len(focus) > 4 or len(focus) != len(set(focus)) or any(key not in FOCUS_DIRECTIVES for key in focus):
        raise ValueError("Invalid research-focus selection")
    return ANALYSIS_INSTRUCTIONS + ("\n\nNext-iteration research focus:\n" + "\n".join(
        "- " + FOCUS_DIRECTIVES[key] for key in focus
    ) if focus else "")


class PromptStore:
    """Append-only revisions committed only by trusted worker publication."""

    def __init__(self, path: Path):
        self.path = path
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS revisions (version INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)")

    def history(self) -> list[dict]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT version, payload FROM revisions ORDER BY version DESC LIMIT 100").fetchall()
        return [{**json.loads(payload), "version": version} for version, payload in rows]

    def current(self) -> dict:
        history = self.history()
        return history[0] if history else baseline()

    def commit(self, record: dict) -> bool:
        if record.get("_publisher_source") != "research_worker":
            return False
        verdicts = record.get("verdicts", {})
        review = verdicts.get("meta_review", {})
        analysis = verdicts.get("analysis", {})
        if review.get("abstained", True) or analysis.get("abstained", True):
            return False
        data = review.get("data", {})
        proposal = data.get("review", {})
        focus = proposal.get("next_focus", [])
        instructions = revised_instructions(focus)
        source = f"{record['case_id']}:{record['assessed_at']}"
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT version, payload FROM revisions ORDER BY version DESC LIMIT 1").fetchone()
            current = {**json.loads(row[1]), "version": row[0]} if row else baseline()
            # Reject stale proposals, including concurrent public runs.
            if data.get("base_prompt_version") != current["version"] or focus == current["focus"]:
                return False
            payload = {"focus": focus, "instructions": instructions,
                       "reason": proposal.get("revision_reason", ""), "source_iteration": source,
                       "created_at": datetime.now(timezone.utc).isoformat(),
                       "parent_version": current["version"], "previous_instructions": current["instructions"],
                       "status": "Active for subsequent iterations; not validated as an improvement"}
            result = db.execute("INSERT OR IGNORE INTO revisions(source, payload) VALUES (?, ?)", (source, json.dumps(payload)))
            return result.rowcount == 1


def review_history(records: list[dict]) -> dict:
    """All-history counts plus bounded recent evidence, avoiding unbounded prompts."""
    trusted = [r for r in records if r.get("_publisher_source") == "research_worker"]
    counts: dict[str, dict] = {}
    for record in trusted:
        for name, verdict in record.get("verdicts", {}).items():
            count = counts.setdefault(name, {"runs": 0, "abstentions": 0, "failures": 0})
            count["runs"] += 1
            count["abstentions"] += bool(verdict.get("abstained"))
            count["failures"] += any(f.get("code") == "AGENT_FAILURE" for f in verdict.get("flags", []))
    recent = []
    for record in trusted[-8:]:
        recent.append({"case_id": record.get("case_id"), "assessed_at": record.get("assessed_at"),
                       "verdicts": {name: {"headline": v.get("headline"), "abstained": v.get("abstained"),
                                           "rationale": [str(s)[:2500] for s in v.get("rationale", [])[:3]],
                                           "prompt_version": v.get("data", {}).get("prompt_revision", {}).get("version")}
                                    for name, v in record.get("verdicts", {}).items()}})
    return {"total_iterations": len(trusted), "unique_cases": len({r.get("case_id") for r in trusted}),
            "agent_totals": counts, "recent_iterations": recent,
            "scope": "Counts cover all stored worker iterations. Detailed evidence covers the latest eight; this is not a full-text review of all historical outputs."}
