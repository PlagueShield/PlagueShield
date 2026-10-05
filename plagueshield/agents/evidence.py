"""Evidence Agent — retrieves current CDC/WHO guidance and relevant literature.

Retrieval is topic-driven, not keyword-driven: the agent decides which topics
this case actually raises (treatment, case definition, resistance reporting,
notification obligations) and pulls the corpus entries covering them.

Offline-first by design. The bundled corpus always answers; live refresh via
`plagueshield.data.public_sources` only annotates entries with a retrieval
timestamp and availability. A field deployment with no connectivity still
produces a correctly cited report, and the report says plainly whether its
guidance was verified live or served from the bundled snapshot.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..knowledge.guidance import CORPUS, GuidanceEntry, retrieve
from ..models import (
    AnomalyStatus,
    CaseRecord,
    ClinicalForm,
    Confidence,
    Severity,
    Verdict,
)
from .base import Agent


class EvidenceAgent(Agent):
    name = "evidence"
    description = "Retrieves current CDC/WHO guidance and relevant literature"

    def __init__(self, live_refresh: bool = False, timeout: float = 6.0) -> None:
        self.live_refresh = live_refresh
        self.timeout = timeout

    def run(self, case: CaseRecord, context: dict[str, Verdict]) -> Verdict:
        topics = self._topics(case, context)
        entries = retrieve(topics, limit=8)

        rationale = [f"Topics raised by this case: {', '.join(sorted(topics))}"]
        flags = []
        refreshed: dict[str, str] = {}

        if self.live_refresh:
            refreshed, errors = self._refresh(entries)
            if errors:
                flags.append(
                    self.flag(
                        "GUIDANCE_REFRESH_PARTIAL",
                        Severity.LOW,
                        "Some guidance sources could not be verified live",
                        "; ".join(errors[:3]),
                    )
                )
            rationale.append(
                f"Live verification: {len(refreshed)}/{len(entries)} sources "
                "reachable"
            )
        else:
            rationale.append(
                "Served from bundled guidance snapshot (no live refresh "
                "requested)"
            )

        for entry in entries:
            marker = "live" if entry.key in refreshed else "bundled"
            rationale.append(f"[{marker}] {entry.citation.short()} — {entry.summary}")

        citations = [e.citation for e in entries]

        return self.verdict(
            headline=f"{len(entries)} guidance sources retrieved",
            score=float(len(entries)),
            confidence=Confidence.HIGH if entries else Confidence.LOW,
            rationale=rationale,
            flags=flags,
            citations=citations,
            data={
                "topics": sorted(topics),
                "live_refresh": self.live_refresh,
                "entries": [
                    {
                        "key": e.key,
                        "source": e.citation.source,
                        "title": e.citation.title,
                        "url": e.citation.url,
                        "summary": e.summary,
                        "verified": e.key in refreshed,
                        "verified_at": refreshed.get(e.key),
                    }
                    for e in entries
                ],
            },
        )

    # -- helpers ---------------------------------------------------------

    def _topics(self, case: CaseRecord, context: dict[str, Verdict]) -> set[str]:
        topics = {"guidance", "case_definition", "diagnosis", "laboratory"}

        diagnostic = context.get("diagnostic")
        form = case.clinical.form
        if diagnostic:
            form_value = diagnostic.data.get("clinical_form")
            if form_value:
                try:
                    form = ClinicalForm(form_value)
                except ValueError:
                    pass

        if form == ClinicalForm.PNEUMONIC:
            topics |= {"notification", "escalation", "public_health", "treatment"}

        if diagnostic and (diagnostic.score or 0) >= 0.30:
            topics |= {"treatment", "prophylaxis"}

        resistance = context.get("resistance")
        if resistance:
            anomaly = resistance.data.get("anomaly")
            topics.add("susceptibility")
            if anomaly in (
                AnomalyStatus.SUSPECTED.value,
                AnomalyStatus.CONFIRMED.value,
            ):
                topics |= {"resistance", "reporting", "escalation", "breakpoints"}
            if resistance.abstained:
                topics |= {"resistance", "genomics", "tooling"}

        if case.genomic is not None:
            topics |= {"genomics", "resistance"}

        if case.exposure.laboratory_exposure:
            topics |= {"biosafety", "select_agent", "referral"}

        if case.exposure.cluster_size and case.exposure.cluster_size > 1:
            topics |= {"outbreak", "surveillance", "epidemiology"}

        return topics

    def _refresh(
        self, entries: list[GuidanceEntry]
    ) -> tuple[dict[str, str], list[str]]:
        """Best-effort liveness check. Never raises, never blocks a verdict."""
        refreshed: dict[str, str] = {}
        errors: list[str] = []
        try:
            import httpx
        except ImportError:  # pragma: no cover
            return refreshed, ["httpx unavailable"]

        now = datetime.now(timezone.utc).isoformat()
        targets = [e for e in entries if e.live_url]
        if not targets:
            return refreshed, errors

        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": "PlagueShield/0.1 (decision support)"},
            ) as client:
                for entry in targets:
                    try:
                        resp = client.head(entry.live_url)
                        if resp.status_code >= 400:
                            resp = client.get(entry.live_url)
                        if resp.status_code < 400:
                            refreshed[entry.key] = now
                        else:
                            errors.append(f"{entry.key}: HTTP {resp.status_code}")
                    except Exception as exc:  # noqa: BLE001 - liveness only
                        errors.append(f"{entry.key}: {type(exc).__name__}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"client: {type(exc).__name__}")

        return refreshed, errors
