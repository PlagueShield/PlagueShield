"""Next-Test Agent — ranks what to measure next by expected information gain.

The question is not "what tests exist" but "which single result, obtained now,
would most reduce what we do not know". That is an information-theoretic
question with a real answer, so it gets computed rather than heuristically
guessed.

For a binary hypothesis H with current probability p and a test with
sensitivity s and specificity t:

    P(+)   = p·s + (1−p)·(1−t)
    p|+    = p·s / P(+)
    p|−    = p·(1−s) / P(−)
    EIG    = H(p) − [ P(+)·H(p|+) + P(−)·H(p|−) ]     (bits)

EIG peaks where uncertainty is highest and the test is discriminating. It is
correctly near zero when we are already certain — which is why a test that
"confirms what we know" ranks low, as it should.

Information alone is not the whole decision, so EIG is then traded against:

  * **turnaround** — a marginally better answer in two weeks loses to a good
    answer in four hours, because the patient is treated in the meantime;
  * **actionability** — whether the result would actually change management;
  * **feasibility** — a test needing an isolate is not available without one.

Both hypotheses are scored — identification and resistance — and a test is
credited for whichever it informs. The weighting between them shifts with the
case: when plague is already near-certain, identification information is worth
little and resistance information is worth a great deal.
"""

from __future__ import annotations

import math

from ..knowledge.assay_performance import ASSAY_PROFILES, profile_for
from ..knowledge.regimens import CORE_REGIMEN_CLASSES, role_for
from ..models import (
    Assay,
    CandidateTest,
    CaseRecord,
    Confidence,
    ResultValue,
    Severity,
    Verdict,
)
from .base import Agent


def entropy(p: float) -> float:
    """Shannon entropy of a Bernoulli(p), in bits."""
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def expected_information_gain(p: float, sensitivity: float, specificity: float) -> float:
    """Expected reduction in entropy from running a binary test."""
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    p_pos = p * sensitivity + (1.0 - p) * (1.0 - specificity)
    p_neg = 1.0 - p_pos
    if p_pos <= 1e-12 or p_neg <= 1e-12:
        return 0.0

    post_pos = (p * sensitivity) / p_pos
    post_neg = (p * (1.0 - sensitivity)) / p_neg
    expected_posterior = p_pos * entropy(post_pos) + p_neg * entropy(post_neg)
    return max(0.0, entropy(p) - expected_posterior)


class NextTestAgent(Agent):
    name = "next_test"
    description = (
        "Ranks the next diagnostic measurement by expected information gain, "
        "turnaround and clinical relevance"
    )
    depends_on = ("diagnostic", "resistance", "uncertainty")

    def run(self, case: CaseRecord, context: dict[str, Verdict]) -> Verdict:
        diagnostic = context.get("diagnostic")
        resistance = context.get("resistance")

        p_plague = (diagnostic.score if diagnostic else 0.5) or 0.5
        p_resistance = (resistance.score if resistance else 0.02) or 0.02

        # When the Resistance Agent abstained, its posterior is not a
        # calibrated belief -- it is the base-rate prior, barely moved, because
        # nothing informative was measured. Feeding that straight into an EIG
        # calculation produces a near-zero entropy and therefore the conclusion
        # that susceptibility testing is not worth doing. That is exactly
        # backwards: the entropy is low *because* of the prior, not because we
        # know anything about this isolate.
        #
        # So for ranking purposes we evaluate resistance EIG at a probability
        # pulled toward maximum entropy in proportion to our acknowledged
        # ignorance. With complete evidence this is a no-op; with none, it
        # correctly reports that the first measurement is highly informative.
        completeness = 1.0
        if resistance:
            completeness = float(
                resistance.data.get("ood", {}).get("completeness", 1.0)
            )
        ignorance = max(0.0, min(1.0, 1.0 - completeness))
        p_resistance_eff = p_resistance + (0.5 - p_resistance) * ignorance

        has_isolate = case.has_positive(Assay.CULTURE)
        candidates = self._candidates(case, has_isolate)

        # How much each hypothesis is worth resolving right now.
        id_weight, res_weight = self._hypothesis_weights(
            p_plague, p_resistance_eff, case, resistance
        )

        rationale = [
            f"Current plague probability {p_plague:.1%} "
            f"(entropy {entropy(p_plague):.2f} bits)",
            f"Current resistance probability {p_resistance:.2%} "
            f"(entropy {entropy(p_resistance):.3f} bits)",
        ]
        if ignorance > 0.01:
            rationale.append(
                f"Resistance evidence is {completeness:.0%} complete; EIG "
                f"evaluated at {p_resistance_eff:.1%} rather than "
                f"{p_resistance:.2%} to avoid treating an uninformed prior as "
                f"a confident posterior (entropy "
                f"{entropy(p_resistance_eff):.2f} bits)"
            )
        rationale.append(
            f"Hypothesis weighting — identification {id_weight:.2f}, "
            f"resistance {res_weight:.2f}"
        )

        for cand in candidates:
            eig_id = (
                expected_information_gain(p_plague, cand.sensitivity, cand.specificity)
                if cand.target_question in ("identification", "both")
                else 0.0
            )
            eig_res = (
                expected_information_gain(
                    p_resistance_eff, cand.sensitivity, cand.specificity
                )
                if cand.target_question in ("resistance", "both")
                else 0.0
            )
            cand.expected_information_gain_bits = (
                id_weight * eig_id + res_weight * eig_res
            )

            time_factor = 1.0 / (1.0 + cand.turnaround_hours / 24.0)
            action_factor = 0.4 + 0.6 * cand.clinical_actionability
            feasibility = 1.0
            if cand.requires_isolate and not has_isolate:
                feasibility = 0.15
            if cand.already_done:
                feasibility *= 0.25

            cand.utility = (
                cand.expected_information_gain_bits
                * time_factor
                * action_factor
                * feasibility
            )
            cand.rationale = (
                f"{cand.expected_information_gain_bits:.3f} bits × "
                f"time {time_factor:.2f} × actionability {action_factor:.2f}"
                + (f" × feasibility {feasibility:.2f}" if feasibility < 1.0 else "")
            )

        ranked = sorted(candidates, key=lambda c: -(c.utility or 0.0))
        for i, cand in enumerate(ranked, start=1):
            cand.rank = i

        top = ranked[0] if ranked else None
        flags = []

        if top and (top.utility or 0.0) < 0.02:
            flags.append(
                self.flag(
                    "LOW_INFORMATION_YIELD",
                    Severity.LOW,
                    "No remaining test offers substantial information gain",
                    "Either the case is already well characterised, or the "
                    "informative tests are not currently feasible.",
                )
            )

        if not has_isolate and any(
            c.requires_isolate and (c.expected_information_gain_bits or 0) > 0.1
            for c in ranked
        ):
            flags.append(
                self.flag(
                    "ISOLATE_IS_BOTTLENECK",
                    Severity.MODERATE,
                    "The highest-information resistance tests require an isolate "
                    "that is not available",
                    "Recovering a viable isolate — before further antibiotic "
                    "exposure where clinically safe — unlocks phenotypic "
                    "susceptibility testing, which no molecular result "
                    "substitutes for.",
                )
            )

        for cand in ranked[:3]:
            rationale.append(
                f"#{cand.rank} {cand.name}: utility {cand.utility:.3f} "
                f"({cand.rationale}), TAT {cand.turnaround_hours:.0f}h"
            )

        headline = (
            f"Most valuable next result: {top.name}" if top else "No test recommended"
        )

        return self.verdict(
            headline=headline,
            score=top.utility if top else None,
            confidence=Confidence.MODERATE if top else Confidence.LOW,
            rationale=rationale,
            flags=flags,
            data={
                "p_plague": p_plague,
                "p_resistance": p_resistance,
                "p_resistance_effective": p_resistance_eff,
                "evidence_completeness": completeness,
                "id_weight": id_weight,
                "res_weight": res_weight,
                "has_isolate": has_isolate,
                "recommendation": top.name if top else None,
                "ranked": [c.model_dump() for c in ranked],
            },
        )

    # -- helpers ---------------------------------------------------------

    def _hypothesis_weights(
        self,
        p_plague: float,
        p_resistance: float,
        case: CaseRecord,
        resistance: Verdict | None,
    ) -> tuple[float, float]:
        """Decide how much resolving each question is worth right now.

        Identification dominates while the diagnosis is open; once plague is
        near-certain, the open question becomes whether treatment will work.
        """
        id_weight = entropy(p_plague)

        # Resistance only matters if this is plausibly plague at all.
        res_weight = entropy(p_resistance) * min(1.0, p_plague / 0.3)

        # If the resistance agent abstained for lack of evidence, resolving
        # that is worth more than the raw entropy suggests, because the
        # entropy is low only because the prior is low.
        if resistance and resistance.abstained:
            res_weight = max(res_weight, 0.45 * min(1.0, p_plague / 0.3))

        # A confirmed or suspected anomaly makes resistance the live question.
        if resistance and resistance.data.get("anomaly") in (
            "Suspected",
            "Confirmed",
        ):
            res_weight = max(res_weight, 0.9)

        total = id_weight + res_weight
        if total <= 1e-9:
            return 0.5, 0.5
        return id_weight / total, res_weight / total

    def _candidates(self, case: CaseRecord, has_isolate: bool) -> list[CandidateTest]:
        done = case.tested_assays()
        pending = case.pending_assays()
        candidates: list[CandidateTest] = []

        # Identification assays, drawn from the published performance table.
        for assay, profile in ASSAY_PROFILES.items():
            if assay in (Assay.SEROLOGY_F1_SINGLE,):
                continue  # dominated by the paired test; never the best choice
            candidates.append(
                CandidateTest(
                    key=assay.value,
                    name=profile.display_name,
                    target_question="identification",
                    sensitivity=profile.sensitivity,
                    specificity=profile.specificity,
                    turnaround_hours=profile.turnaround_hours,
                    requires_isolate=profile.requires_isolate,
                    requires_bsl3=profile.requires_bsl3,
                    clinical_actionability=self._actionability(assay, profile),
                    availability_note=profile.note or None,
                    already_done=assay in done or assay in pending,
                )
            )

        # Resistance-side measurements, which the assay table does not cover.
        tested_classes = {r.drug_class for r in case.susceptibility}
        untested_core = [c for c in CORE_REGIMEN_CLASSES if c not in tested_classes]

        candidates.append(
            CandidateTest(
                key="phenotypic_ast_full",
                name="Phenotypic susceptibility testing (full core panel)",
                target_question="resistance",
                sensitivity=0.97,
                specificity=0.95,
                turnaround_hours=24.0,
                requires_isolate=True,
                requires_bsl3=True,
                clinical_actionability=1.0,
                availability_note=(
                    "Broth microdilution against CLSI M45 breakpoints. The "
                    "definitive answer on whether treatment will work; nothing "
                    "else substitutes for it."
                ),
                already_done=not untested_core,
            )
        )

        if untested_core:
            names = ", ".join(role_for(c).display_name for c in untested_core)
            candidates.append(
                CandidateTest(
                    key="phenotypic_ast_gap",
                    name=f"Phenotypic susceptibility — untested classes ({names})",
                    target_question="resistance",
                    sensitivity=0.97,
                    specificity=0.95,
                    turnaround_hours=20.0,
                    requires_isolate=True,
                    requires_bsl3=True,
                    clinical_actionability=1.0,
                    availability_note="Completes coverage of the core regimen.",
                )
            )

        nonsusceptible_unconfirmed = [
            r
            for r in case.susceptibility
            if r.is_nonsusceptible and not r.confirmed_by_repeat
        ]
        if nonsusceptible_unconfirmed:
            candidates.append(
                CandidateTest(
                    key="ast_repeat_confirm",
                    name="Repeat susceptibility testing on fresh subculture "
                    "(reference laboratory)",
                    target_question="resistance",
                    sensitivity=0.98,
                    specificity=0.98,
                    turnaround_hours=30.0,
                    requires_isolate=True,
                    requires_bsl3=True,
                    clinical_actionability=1.0,
                    availability_note=(
                        "An unconfirmed non-susceptible result is the single "
                        "highest-value thing to adjudicate: it is either a "
                        "testing artefact or a reportable public health event, "
                        "and those demand opposite responses."
                    ),
                )
            )

        screened = bool(case.genomic and case.genomic.amr_screen_performed)
        candidates.append(
            CandidateTest(
                key="wgs_amr_screen",
                name="Whole-genome sequencing with AMR determinant screen",
                target_question="both",
                sensitivity=0.90,
                specificity=0.97,
                turnaround_hours=48.0,
                requires_isolate=False,
                requires_bsl3=False,
                clinical_actionability=0.65,
                availability_note=(
                    "Confirms species and screens for catalogued determinants "
                    "via AMRFinderPlus. Cannot detect an uncatalogued "
                    "mechanism, so a clean screen does not replace phenotypic "
                    "testing."
                ),
                already_done=screened,
            )
        )

        return candidates

    @staticmethod
    def _actionability(assay: Assay, profile) -> float:
        """How much a result from this assay would change management."""
        if assay == Assay.CULTURE:
            # Uniquely high: it is also the gateway to susceptibility testing.
            return 1.0
        if profile.confirmatory:
            return 0.85
        if assay in (Assay.F1_RDT, Assay.PCR_PLA, Assay.PCR_CAF1):
            return 0.75  # fast enough to affect the first treatment decision
        if assay in (Assay.SEROLOGY_F1_PAIRED,):
            return 0.2  # retrospective; no bearing on acute management
        return 0.5
