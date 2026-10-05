"""Shared agent protocol.

Every agent takes the case plus whatever upstream verdicts it depends on, and
returns a `Verdict`. Agents never mutate the case record and never call each
other directly -- the orchestrator owns sequencing. That keeps each agent
independently testable and makes the dependency graph explicit rather than
emergent.
"""

from __future__ import annotations

import abc
from typing import Any

from ..models import CaseRecord, Confidence, Flag, Severity, Verdict


class Agent(abc.ABC):
    """Base class for all PlagueShield agents."""

    name: str = "agent"
    description: str = ""
    #: Names of agents whose verdicts must be available before this one runs.
    depends_on: tuple[str, ...] = ()

    @abc.abstractmethod
    def run(
        self, case: CaseRecord, context: dict[str, Verdict]
    ) -> Verdict:  # pragma: no cover - interface
        ...

    # -- helpers ---------------------------------------------------------

    def verdict(
        self,
        headline: str,
        *,
        score: float | None = None,
        confidence: Confidence = Confidence.LOW,
        rationale: list[str] | None = None,
        flags: list[Flag] | None = None,
        citations: list[Any] | None = None,
        data: dict[str, Any] | None = None,
        abstained: bool = False,
        abstain_reason: str | None = None,
    ) -> Verdict:
        return Verdict(
            agent=self.name,
            headline=headline,
            score=score,
            confidence=confidence,
            rationale=rationale or [],
            flags=flags or [],
            citations=citations or [],
            data=data or {},
            abstained=abstained,
            abstain_reason=abstain_reason,
        )

    @staticmethod
    def flag(
        code: str, severity: Severity, message: str, detail: str | None = None
    ) -> Flag:
        return Flag(code=code, severity=severity, message=message, detail=detail)
