"""Reproducible diagnostic model stress tests; no generated-code execution."""

from __future__ import annotations

from .base import Agent
from .diagnostic import DiagnosticAgent, family_of
from .next_test import entropy
from ..models import CaseRecord, Confidence, Verdict


class CodeAnalysisAgent(Agent):
    name = "code_analysis"
    description = "Executes Python evidence-ablation and posterior-odds sensitivity experiments"
    depends_on = ("diagnostic", "resistance", "uncertainty", "next_test")

    def run(self, case: CaseRecord, context: dict[str, Verdict]) -> Verdict:
        baseline = context["diagnostic"]
        if baseline.abstained or baseline.score is None:
            return self.verdict(
                headline="Numerical experiments unavailable: diagnostic baseline missing",
                abstained=True, abstain_reason="A completed diagnostic probability is required.",
            )
        probability = baseline.score
        experiments = []
        families = sorted({family_of(result.assay) for result in case.diagnostics})
        for family in [*families, "all_recorded_assays"] if families else []:
            retained = [] if family == "all_recorded_assays" else [
                result for result in case.diagnostics if family_of(result.assay) != family
            ]
            alternative = DiagnosticAgent().run(case.model_copy(update={"diagnostics": retained}), {})
            experiments.append({
                "removed_family": family,
                "removed_results": len(case.diagnostics) - len(retained),
                "probability": alternative.score,
                "change_percentage_points": round((alternative.score - probability) * 100, 4),
                "likelihood": alternative.data.get("likelihood"),
                "classification": alternative.data.get("classification"),
            })
        odds_sweep = [
            {"odds_multiplier": multiplier,
             "probability": multiplier * probability / (1 - probability + multiplier * probability)}
            for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0)
        ]
        max_change = max((abs(row["change_percentage_points"]) for row in experiments), default=0.0)
        rationale = [
            f"Baseline probability {probability:.2%}; binary entropy {entropy(probability):.4f} bits.",
            f"Re-executed DiagnosticAgent for {len(experiments)} evidence ablations, including removal of all recorded assays; largest absolute change {max_change:.2f} percentage points.",
            *[f"Remove {row['removed_family']}: {row['probability']:.2%} ({row['change_percentage_points']:+.2f} percentage points), {row['classification']}." for row in experiments],
            "Posterior-odds multipliers 0.25-4 are illustrative stress assumptions, not measured uncertainty or confidence intervals.",
            "Ablation removes recorded assay results only; clinical and genomic evidence remain fixed. Correlated sources and posterior caps can mask sensitivity. No clinical validation is implied.",
        ]
        return self.verdict(
            headline=f"Python audit: {len(experiments)} evidence ablations, maximum shift {max_change:.2f} pp",
            confidence=Confidence.MODERATE, rationale=rationale,
            citations=baseline.citations,
            data={"runtime": "Python", "baseline_probability": probability,
                  "entropy_bits": entropy(probability), "ablations": experiments,
                  "posterior_odds_sweep": odds_sweep,
                  "max_change_percentage_points": max_change,
                  "clinical_decisions_changed": False},
        )
