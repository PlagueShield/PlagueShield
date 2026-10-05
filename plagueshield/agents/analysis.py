"""LLM Analysis Agent."""

from __future__ import annotations

from ..llm import DEFAULT_ANALYSIS_MODEL, LLMUnavailable, analyze_with_openai, build_analysis_request, is_configured
from ..models import CaseRecord, Confidence, Severity, Verdict
from ..knowledge.case_sources import case_reference_info
from .base import Agent
from ..prompt_evolution import baseline


class LLMAnalysisAgent(Agent):
    name = "analysis"
    description = "GPT-5.5 research brief grounded in agent findings, citations, and Python experiments"
    depends_on = (
        "diagnostic",
        "resistance",
        "evidence",
        "discordance",
        "uncertainty",
        "next_test",
        "code_analysis",
    )

    def __init__(self, model: str = DEFAULT_ANALYSIS_MODEL, prompt_revision: dict | None = None) -> None:
        self.model = model
        self.prompt_revision = prompt_revision or baseline()

    def run(self, case: CaseRecord, context: dict[str, Verdict]) -> Verdict:
        compact = {
            name: {
                "headline": verdict.headline,
                "score": verdict.score,
                "confidence": verdict.confidence.value,
                "abstained": verdict.abstained,
                "abstain_reason": verdict.abstain_reason,
                "rationale": verdict.rationale,
                "citations": [citation.model_dump(mode="json") for citation in verdict.citations],
                "flags": [
                    {
                        "code": flag.code,
                        "severity": flag.severity.value,
                        "message": flag.message,
                    }
                    for flag in verdict.flags
                ],
                "data": {
                    key: value
                    for key, value in verdict.data.items()
                    if key not in {"execution", "llm_request", "traceback"}
                },
            }
            for name, verdict in context.items()
        }
        case_payload = {**case.model_dump(mode="json"), "source_context": case_reference_info(case)}
        request = build_analysis_request(case_payload, compact, self.model, self.prompt_revision["instructions"])
        try:
            synthesis = analyze_with_openai(
                case_payload=case_payload,
                verdicts=compact,
                model=self.model,
                instructions=self.prompt_revision["instructions"],
            )
        except LLMUnavailable as exc:
            return self.verdict(
                headline="GPT-5.5 analysis unavailable",
                confidence=Confidence.LOW,
                abstained=True,
                abstain_reason=str(exc),
                rationale=[
                    str(exc) if is_configured() else "Set OPENAI_API_KEY to enable GPT-5.5 analysis.",
                    f"Configured model: {self.model}.",
                ],
                flags=[
                    self.flag(
                        "LLM_ANALYSIS_UNAVAILABLE",
                        Severity.INFO,
                        "GPT-5.5 analysis layer did not run",
                        str(exc),
                    )
                ],
                data={"model": self.model, "configured": is_configured(), "llm_request": request, "prompt_revision": self.prompt_revision},
            )

        return self.verdict(
            headline="GPT-5.5 analysis complete",
            confidence=Confidence.MODERATE,
            rationale=[synthesis],
            data={"model": self.model, "configured": True, "synthesis": synthesis, "llm_request": request, "prompt_revision": self.prompt_revision},
        )
