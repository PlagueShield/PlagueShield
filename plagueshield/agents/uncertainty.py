"""Uncertainty Agent — says plainly when the system cannot make a reliable call.

Every other agent is trying to answer. This one is trying to work out whether
answering is justified, and it has veto power: it can downgrade any upstream
confidence and can mark the whole assessment as non-reliable.

It looks at four things the individual agents cannot see on their own, because
each only sees its own slice:

  * **Coverage** — how much of the evidence we would want actually exists.
  * **Novelty** — whether the case resembles anything the system has grounds
    to reason about, inherited from the resistance OOD gate and extended to
    the identification side.
  * **Fragility** — whether the conclusion rests on a single result that, if
    wrong, would flip it. A high-probability call resting on one unconfirmed
    field RDT is fragile regardless of how high the number is.
  * **Internal consistency** — whether the agents are contradicting each
    other, inherited from the Discordance Agent.

Design commitment: this agent is allowed to make the system *less* confident
and never more. If it has nothing to say, confidence is whatever the upstream
agents earned.
"""

from __future__ import annotations

import math

from ..knowledge.assay_performance import profile_for
from ..knowledge.regimens import CORE_REGIMEN_CLASSES
from ..models import (
    CaseRecord,
    Confidence,
    DataOrigin,
    Flag,
    ResultValue,
    Severity,
    Verdict,
)
from .base import Agent

CONFIDENCE_ORDER = (Confidence.LOW, Confidence.MODERATE, Confidence.HIGH)


def _downgrade(level: Confidence, steps: int = 1) -> Confidence:
    idx = CONFIDENCE_ORDER.index(level)
    return CONFIDENCE_ORDER[max(0, idx - steps)]


class UncertaintyAgent(Agent):
    name = "uncertainty"
    description = "Detects out-of-distribution and weak-evidence cases"
    depends_on = ("diagnostic", "resistance", "discordance")

    def run(self, case: CaseRecord, context: dict[str, Verdict]) -> Verdict:
        flags: list[Flag] = []
        rationale: list[str] = []
        penalties: list[tuple[str, float]] = []

        coverage = self._coverage(case, rationale)
        novelty = self._novelty(case, context, rationale)
        fragility, fragility_flags = self._fragility(case, context, rationale)
        consistency = self._consistency(context, rationale)
        provenance_penalty = self._provenance(case, rationale, flags)

        flags.extend(fragility_flags)

        if coverage < 0.4:
            penalties.append(("sparse evidence", 0.4 - coverage))
            flags.append(
                self.flag(
                    "SPARSE_EVIDENCE",
                    Severity.HIGH if coverage < 0.25 else Severity.MODERATE,
                    f"Evidence coverage is {coverage:.0%} of what a complete "
                    "workup would provide",
                    "Conclusions drawn from this record are provisional.",
                )
            )

        if novelty > 0.6:
            penalties.append(("out of distribution", novelty - 0.6))
            flags.append(
                self.flag(
                    "OUT_OF_DISTRIBUTION",
                    Severity.HIGH,
                    "Case does not resemble the evidence patterns this system "
                    "has grounds to reason about",
                    f"Novelty score {novelty:.2f}. Treat quantitative outputs "
                    "as indicative only and escalate to specialist review.",
                )
            )

        if fragility > 0.6:
            penalties.append(("fragile conclusion", fragility - 0.6))

        if consistency < 0.5:
            penalties.append(("internal contradiction", 0.5 - consistency))

        reliability = max(
            0.0, 1.0 - sum(weight for _, weight in penalties)
        )
        reliability *= 1.0 - provenance_penalty

        reliable = reliability >= 0.45 and coverage >= 0.25
        recommended = self._recommended_confidence(
            reliability, coverage, novelty, fragility, consistency
        )

        if not reliable:
            flags.append(
                self.flag(
                    "UNRELIABLE_ASSESSMENT",
                    Severity.CRITICAL,
                    "System cannot make a reliable prediction for this case",
                    "The assessment below is shown for transparency but should "
                    "not be used as a basis for clinical decisions without "
                    "specialist review and further testing.",
                )
            )

        headline = (
            f"Reliability: {reliability:.0%} — "
            + ("reliable" if reliable else "NOT reliable")
        )
        rationale.append(
            "Reliability "
            + (
                f"{reliability:.0%} after penalties: "
                + ", ".join(f"{name} (−{w:.2f})" for name, w in penalties)
                if penalties
                else f"{reliability:.0%}, no penalties applied"
            )
        )
        rationale.append(f"Recommended ceiling on reported confidence: {recommended.value}")

        return self.verdict(
            headline=headline,
            score=reliability,
            confidence=recommended,
            abstained=not reliable,
            abstain_reason=None
            if reliable
            else "Insufficient or out-of-distribution evidence for a reliable call",
            rationale=rationale,
            flags=flags,
            data={
                "reliability": reliability,
                "reliable": reliable,
                "coverage": coverage,
                "novelty": novelty,
                "fragility": fragility,
                "consistency": consistency,
                "recommended_confidence": recommended.value,
                "penalties": [{"reason": n, "weight": w} for n, w in penalties],
            },
        )

    # -- components ------------------------------------------------------

    def _coverage(self, case: CaseRecord, rationale: list[str]) -> float:
        have = 0.0
        want = 0.0
        missing: list[str] = []

        checks = [
            ("clinical syndrome", bool(case.clinical.symptoms or case.clinical.form.value != "unknown"), 1.0),
            ("exposure history", any(
                v is not None
                for v in case.exposure.model_dump().values()
            ), 1.0),
            ("any diagnostic result", bool(case.tested_assays()), 2.0),
            ("confirmatory-grade assay", any(
                (p := profile_for(d.assay)) and p.confirmatory
                and d.result in (ResultValue.POSITIVE, ResultValue.NEGATIVE)
                for d in case.diagnostics
            ), 2.0),
            ("genomic AMR screen", bool(case.genomic and case.genomic.amr_screen_performed), 1.5),
            ("phenotypic susceptibility", bool(case.susceptibility), 2.0),
        ]
        for label, present, weight in checks:
            want += weight
            if present:
                have += weight
            else:
                missing.append(label)

        # Partial credit for core-class phenotypic breadth.
        tested = {r.drug_class for r in case.susceptibility}
        want += 1.5
        have += 1.5 * (
            len(tested & set(CORE_REGIMEN_CLASSES)) / len(CORE_REGIMEN_CLASSES)
        )

        coverage = have / want if want else 0.0
        if missing:
            rationale.append(f"Missing evidence: {', '.join(missing)}")
        rationale.append(f"Evidence coverage: {coverage:.0%}")
        return coverage

    def _novelty(
        self, case: CaseRecord, context: dict[str, Verdict], rationale: list[str]
    ) -> float:
        resistance = context.get("resistance")
        novelty = 0.0
        if resistance:
            novelty = float(resistance.data.get("ood", {}).get("novelty", 0.0))

        # Identification-side novelty: a case with a positive result on an
        # assay combination that is internally odd, or no endemic link at all.
        if case.exposure.plague_endemic_area is False and not (
            case.exposure.laboratory_exposure
            or case.exposure.contact_with_confirmed_case
        ):
            novelty = max(novelty, 0.55)
            rationale.append(
                "No endemic, laboratory or contact exposure — unusual context "
                "for plague; raises identification novelty."
            )

        if case.genomic and case.genomic.ani_identity_pct is not None:
            if case.genomic.ani_identity_pct < 97.0:
                novelty = max(novelty, 0.8)
                rationale.append(
                    f"Genomic identity to reference is "
                    f"{case.genomic.ani_identity_pct:.1f}% — below the range "
                    "expected for a confident species assignment."
                )

        rationale.append(f"Novelty score: {novelty:.2f}")
        return novelty

    def _fragility(
        self, case: CaseRecord, context: dict[str, Verdict], rationale: list[str]
    ) -> tuple[float, list[Flag]]:
        """How much the conclusion depends on a single result."""
        flags: list[Flag] = []
        diagnostic = context.get("diagnostic")
        if not diagnostic:
            return 0.0, flags

        informative = [
            d
            for d in case.diagnostics
            if d.result in (ResultValue.POSITIVE, ResultValue.NEGATIVE)
            and profile_for(d.assay)
        ]
        if not informative:
            rationale.append("Fragility: no informative diagnostic results at all.")
            return 1.0, flags

        probability = diagnostic.score or 0.0
        confirmatory = [d for d in informative if profile_for(d.assay).confirmatory]

        fragility = 0.0
        if len(informative) == 1:
            fragility = 0.85
            only = informative[0]
            flags.append(
                self.flag(
                    "SINGLE_RESULT_DEPENDENCY",
                    Severity.HIGH,
                    f"Assessment rests on a single result "
                    f"({profile_for(only.assay).display_name})",
                    "If that result is wrong, the conclusion inverts. Obtain an "
                    "independent second result before acting.",
                )
            )
            rationale.append("Fragility: single informative result.")
        elif not confirmatory and probability >= 0.6:
            fragility = 0.65
            flags.append(
                self.flag(
                    "NO_CONFIRMATORY_EVIDENCE",
                    Severity.MODERATE,
                    "Elevated probability without any confirmatory-grade result",
                    "Screening and presumptive assays alone cannot confirm a "
                    "case under the surveillance definition.",
                )
            )
            rationale.append("Fragility: no confirmatory-grade assay.")
        elif len(informative) == 2:
            fragility = 0.45
            rationale.append("Fragility: only two informative results.")
        else:
            fragility = max(0.0, 0.4 - 0.05 * (len(informative) - 2))
            rationale.append(
                f"Fragility: {len(informative)} independent informative results."
            )

        # A probability sitting near the decision boundary is inherently fragile.
        if 0.25 <= probability <= 0.55:
            fragility = max(fragility, 0.55)
            rationale.append(
                f"Probability {probability:.0%} sits in the indeterminate band "
                "where small evidence changes flip the call."
            )

        return fragility, flags

    def _consistency(self, context: dict[str, Verdict], rationale: list[str]) -> float:
        discordance = context.get("discordance")
        if not discordance:
            return 1.0
        n_critical = int(discordance.data.get("n_critical", 0))
        n_high = int(discordance.data.get("n_high", 0))
        score = 1.0 - min(1.0, 0.4 * n_critical + 0.15 * n_high)
        if n_critical or n_high:
            rationale.append(
                f"Internal consistency {score:.2f} "
                f"({n_critical} critical, {n_high} significant conflicts)"
            )
        return score

    def _provenance(
        self, case: CaseRecord, rationale: list[str], flags: list[Flag]
    ) -> float:
        """Penalty reflecting how the record itself was obtained."""
        origin = case.provenance.origin
        if origin == DataOrigin.SYNTHETIC:
            rationale.append(
                "Record is synthetic — suitable for validating system "
                "behaviour, not for drawing epidemiological conclusions."
            )
            return 0.0
        if origin == DataOrigin.PUBLISHED_REPORT:
            rationale.append(
                "Record reconstructed from a published report; granularity is "
                "limited to what the publication described."
            )
            return 0.08
        if origin == DataOrigin.SURVEILLANCE_SUMMARY:
            rationale.append(
                "Record derived from aggregate surveillance reporting; "
                "individual-level detail is approximate."
            )
            return 0.12
        return 0.0

    @staticmethod
    def _recommended_confidence(
        reliability: float,
        coverage: float,
        novelty: float,
        fragility: float,
        consistency: float,
    ) -> Confidence:
        if (
            reliability >= 0.8
            and coverage >= 0.7
            and novelty <= 0.35
            and fragility <= 0.4
            and consistency >= 0.85
        ):
            return Confidence.HIGH
        if reliability >= 0.55 and coverage >= 0.4 and novelty <= 0.6:
            return Confidence.MODERATE
        return Confidence.LOW
