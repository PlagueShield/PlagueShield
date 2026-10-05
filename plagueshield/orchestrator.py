"""Pipeline orchestration.

Owns sequencing so the agents don't have to. The dependency graph is declared
here as explicit stages rather than discovered at runtime, which makes the
data flow readable and keeps independent agents genuinely parallel:

    stage 1   diagnostic ‖ resistance        (independent evidence domains)
    stage 2   evidence   ‖ discordance       (both read stage 1)
    stage 3   uncertainty                    (reads everything; holds the veto)
    stage 4   next_test                      (needs posterior + abstention state)
    stage 5   code_analysis                  (Python stress tests)
    stage 6   analysis                       (GPT-5.5 research brief)
    stage 7   summary                        (renders)
    stage 8   meta_review                    (proposes next-iteration focus)

An agent that raises does not take the pipeline down. It yields a failure
verdict and the run continues degraded, because a partial assessment that
names its own gap is more useful to a clinician at 3am than a stack trace.
"""

from __future__ import annotations

import logging
import traceback
import time
from copy import copy
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .agents.base import Agent
from .agents.analysis import LLMAnalysisAgent
from .agents.meta_review import MetaReviewAgent, methodology_sources
from .prompt_evolution import baseline
from .agents.code_analysis import CodeAnalysisAgent
from .agents.diagnostic import DiagnosticAgent
from .agents.discordance import DiscordanceAgent
from .agents.evidence import EvidenceAgent
from .agents.next_test import NextTestAgent
from .agents.resistance import ResistanceAgent
from .agents.summary import ClinicalSummaryAgent
from .agents.uncertainty import UncertaintyAgent
from .models import (
    AnomalyStatus,
    CandidateTest,
    CaseAssessment,
    CaseClassification,
    CaseRecord,
    Citation,
    Confidence,
    Flag,
    Likelihood,
    Severity,
    SusceptibilityOutlook,
    Verdict,
)

logger = logging.getLogger("plagueshield.orchestrator")

PIPELINE_VERSION = "0.4.0"
ProgressCallback = Callable[[str, str, Verdict | None], None]


class Pipeline:
    """Runs ten research agents with an immutable per-iteration prompt snapshot."""

    def __init__(self, live_evidence: bool = False, max_workers: int = 4,
                 prompt_provider: Callable | None = None, history_provider: Callable | None = None) -> None:
        self.max_workers = max_workers
        self.prompt_provider = prompt_provider or baseline
        self.history_provider = history_provider or (lambda: {"total_iterations": 0, "recent_iterations": []})
        self.diagnostic = DiagnosticAgent()
        self.resistance = ResistanceAgent()
        self.evidence = EvidenceAgent(live_refresh=live_evidence)
        self.discordance = DiscordanceAgent()
        self.uncertainty = UncertaintyAgent()
        self.next_test = NextTestAgent()
        self.analysis = LLMAnalysisAgent()
        self.code_analysis = CodeAnalysisAgent()
        self.summary = ClinicalSummaryAgent()
        self.meta_review = MetaReviewAgent()

    # -- execution -------------------------------------------------------

    def _safe_run(
        self, agent: Agent, case: CaseRecord, context: dict[str, Verdict],
        on_progress: ProgressCallback | None = None,
    ) -> Verdict:
        self._notify(on_progress, agent.name, "running", None)
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        try:
            verdict = agent.run(case, dict(context))
        except Exception as exc:  # noqa: BLE001 - degrade, never crash
            logger.exception("agent %s failed on case %s", agent.name, case.case_id)
            verdict = Verdict(
                agent=agent.name,
                headline=f"{agent.name} agent failed",
                confidence=Confidence.LOW,
                abstained=True,
                abstain_reason=f"{type(exc).__name__}: {exc}",
                rationale=[
                    "This agent did not complete. Its contribution is absent "
                    "from the assessment below, which should therefore be "
                    "treated as incomplete."
                ],
                flags=[
                    Flag(
                        code="AGENT_FAILURE",
                        severity=Severity.HIGH,
                        message=f"The {agent.name} agent failed to run",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                ],
                data={"traceback": traceback.format_exc(limit=5)},
            )
        verdict.data["execution"] = {
            "started_at": started_at,
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
            "pipeline_version": PIPELINE_VERSION,
            "case_input": case.model_dump(mode="json"),
            "upstream": {
                name: {**value.model_dump(mode="json"),
                       "data": {key: item for key, item in value.data.items() if key not in {"execution", "llm_request"}}}
                for name, value in context.items() if name in agent.depends_on
            },
        }
        status = "failed" if any(f.code == "AGENT_FAILURE" for f in verdict.flags) else (
            "abstained" if verdict.abstained else "completed"
        )
        self._notify(on_progress, agent.name, status, verdict)
        return verdict

    @staticmethod
    def _notify(callback: ProgressCallback | None, name: str, status: str, verdict: Verdict | None) -> None:
        if callback is not None:
            try:
                callback(name, status, verdict)
            except Exception:
                logger.exception("Progress observer failed for %s", name)

    def _parallel(
        self, agents: list[Agent], case: CaseRecord, context: dict[str, Verdict],
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Verdict]:
        if len(agents) == 1:
            agent = agents[0]
            return {agent.name: self._safe_run(agent, case, context, on_progress)}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(agents))) as pool:
            futures = {
                agent.name: pool.submit(self._safe_run, agent, case, context, on_progress)
                for agent in agents
            }
            return {name: future.result() for name, future in futures.items()}

    def run(self, case: CaseRecord, on_progress: ProgressCallback | None = None) -> CaseAssessment:
        context: dict[str, Verdict] = {}
        revision = self.prompt_provider()
        history = self.history_provider()

        context.update(self._parallel([self.diagnostic, self.resistance], case, context, on_progress))
        context.update(self._parallel([self.evidence, self.discordance], case, context, on_progress))
        context.update(self._parallel([self.uncertainty], case, context, on_progress))
        context.update(self._parallel([self.next_test], case, context, on_progress))
        context.update(self._parallel([self.code_analysis], case, context, on_progress))
        analysis = copy(self.analysis)
        analysis.prompt_revision = revision
        context.update(self._parallel([analysis], case, context, on_progress))
        context.update(self._parallel([self.summary], case, context, on_progress))
        reviewer = MetaReviewAgent(history, methodology_sources([
            self.diagnostic, self.resistance, self.evidence, self.discordance, self.uncertainty,
            self.next_test, self.code_analysis, self.analysis, self.summary,
        ]))
        context.update(self._parallel([reviewer], case, context, on_progress))

        return self._assemble(case, context)

    # -- assembly --------------------------------------------------------

    def _assemble(
        self, case: CaseRecord, context: dict[str, Verdict]
    ) -> CaseAssessment:
        diagnostic = context.get("diagnostic")
        resistance = context.get("resistance")
        uncertainty = context.get("uncertainty")
        next_test = context.get("next_test")
        summary = context.get("summary")
        discordance = context.get("discordance")

        likelihood = Likelihood(
            diagnostic.data.get("likelihood", Likelihood.INDETERMINATE.value)
            if diagnostic and not diagnostic.abstained
            else Likelihood.INDETERMINATE.value
        )
        classification = CaseClassification(
            diagnostic.data.get("classification", CaseClassification.SUSPECTED.value)
            if diagnostic
            else CaseClassification.SUSPECTED.value
        )
        outlook = SusceptibilityOutlook(
            resistance.data.get(
                "outlook", SusceptibilityOutlook.INSUFFICIENT_EVIDENCE.value
            )
            if resistance
            else SusceptibilityOutlook.INSUFFICIENT_EVIDENCE.value
        )
        anomaly = AnomalyStatus(
            resistance.data.get("anomaly", AnomalyStatus.INSUFFICIENT_EVIDENCE.value)
            if resistance
            else AnomalyStatus.INSUFFICIENT_EVIDENCE.value
        )

        # The Uncertainty Agent caps reported confidence; it can only lower it.
        confidence = Confidence.LOW
        if uncertainty:
            confidence = uncertainty.confidence
        earned = min(
            (
                v.confidence
                for v in (diagnostic, resistance)
                if v is not None
            ),
            key=lambda c: ["Low", "Moderate", "High"].index(c.value),
            default=Confidence.LOW,
        )
        order = ["Low", "Moderate", "High"]
        confidence = Confidence(
            order[min(order.index(confidence.value), order.index(earned.value))]
        )

        ranked: list[CandidateTest] = []
        if next_test:
            ranked = [
                CandidateTest.model_validate(c)
                for c in next_test.data.get("ranked", [])
            ]

        flags: list[Flag] = []
        citations: list[Citation] = []
        seen_citations: set[tuple] = set()
        for verdict in context.values():
            flags.extend(verdict.flags)
            for cit in verdict.citations:
                key = (cit.source, cit.title)
                if key not in seen_citations:
                    seen_citations.add(key)
                    citations.append(cit)

        severity_rank = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MODERATE: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        flags.sort(key=lambda f: severity_rank.get(f.severity, 5))

        escalation = summary.data.get("escalation", []) if summary else []
        review_required = bool(
            summary.data.get("human_review_required") if summary else False
        )
        triggers = [
            f.message for f in flags if f.severity in (Severity.CRITICAL, Severity.HIGH)
        ]

        return CaseAssessment(
            case_id=case.case_id,
            case_label=case.label,
            assessed_at=datetime.now(timezone.utc),
            pipeline_version=PIPELINE_VERSION,
            plague_likelihood=likelihood,
            plague_probability=diagnostic.score if diagnostic else None,
            case_classification=classification,
            standard_treatment_susceptibility=outlook,
            resistance_anomaly=anomaly,
            confidence=confidence,
            most_valuable_next_result=(
                next_test.data.get("recommendation") if next_test else None
            ),
            escalation=escalation,
            human_review_required=review_required,
            review_triggers=triggers,
            verdicts=context,
            ranked_tests=ranked,
            flags=flags,
            citations=citations,
            report_markdown=summary.data.get("report_markdown", "") if summary else "",
            report_ascii=summary.data.get("report_ascii", "") if summary else "",
        )


# --------------------------------------------------------------------------
# Posting results to the dashboard
# --------------------------------------------------------------------------


class AssessmentPublisher:
    """Posts completed assessments to the PlagueShield web dashboard."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def publish(self, assessment: CaseAssessment) -> tuple[bool, str]:
        import httpx

        url = f"{self.base_url}/api/assessments"
        payload = assessment.model_dump(mode="json")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
            if resp.status_code >= 400:
                return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
            return True, f"posted {assessment.case_id} → {url}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"


def assess_case(case: CaseRecord, live_evidence: bool = False) -> CaseAssessment:
    """Convenience entry point for a single case."""
    return Pipeline(live_evidence=live_evidence).run(case)
