"""Diagnostic Agent — "Is this actually plague?"

A transparent Bayesian log-odds model rather than an opaque classifier, for
two reasons: every term can be shown to a microbiologist and argued with, and
the pre-test probability can be set from epidemiology instead of being baked
into learned weights.

    posterior log-odds = prior log-odds
                       + Σ log LR(syndrome features)
                       + Σ log LR(assay results, weighted by lab tier)

Separately, and *not* by thresholding that probability, the agent assigns a
surveillance case classification (suspect / presumptive / confirmed) by
applying the published criteria directly. The two answers are different
questions -- "how likely is this" versus "what does the case definition call
it" -- and conflating them is a common failure. A confirmed case definitionally
has confirmatory laboratory evidence; a high posterior probability does not.
"""

from __future__ import annotations

import math

from ..fewshot.prior import inv_logit, logit
from ..knowledge.assay_performance import (
    ALL_ASSAY_CITATIONS,
    CIT_CDC_CASE_DEF,
    CIT_CDC_PLAGUE_DX,
    CIT_WHO_LAB_MANUAL,
    lab_tier_weight,
    profile_for,
)
from ..models import (
    Assay,
    CaseClassification,
    CaseRecord,
    ClinicalForm,
    Confidence,
    Likelihood,
    ResultValue,
    Severity,
    SpecimenType,
    Verdict,
)
from .base import Agent

#: Pre-test probability of plague among patients whose presentation prompted
#: plague testing at all. Deliberately not the population prevalence -- the
#: case has already been selected for suspicion by a clinician.
BASE_PRETEST_PROBABILITY = 0.04

# ---------------------------------------------------------------------------
# Correlation damping
# ---------------------------------------------------------------------------
# A naive product of likelihood ratios assumes every piece of evidence is
# conditionally independent given disease status. That assumption is badly
# wrong here and fails in the dangerous direction -- towards false certainty.
#
#   * PCR, culture and antigen detection run on the *same bubo aspirate* are
#     not three independent observations; they are three views of one sample.
#   * "Endemic area" + "contact with a confirmed case" + "part of a cluster"
#     are largely one underlying fact: the patient is in an outbreak.
#
# Multiplying them out produces posteriors of 1.000, which is both wrong and
# corrosive to trust. Instead each evidence group is combined with geometric
# damping: the strongest term counts in full, the next at `damping`, the next
# at `damping²`, and so on, with a ceiling on the group's total contribution.
# Positive and negative terms are damped separately so that contradicting
# evidence is never suppressed merely because it arrived second.
EPI_DAMPING, EPI_CAP = 0.45, math.log(30.0)
SYNDROME_DAMPING, SYNDROME_CAP = 0.50, math.log(25.0)
WITHIN_FAMILY_DAMPING = 0.35
ACROSS_FAMILY_DAMPING = 0.85
ASSAY_CAP = math.log(500.0)

#: Assay families. Results inside a family are strongly correlated (shared
#: specimen, shared failure modes); results across families much less so.
ASSAY_FAMILIES: dict[str, set[Assay]] = {
    "antigen": {Assay.F1_RDT, Assay.IMMUNOFLUORESCENCE},
    "molecular": {
        Assay.PCR_PLA,
        Assay.PCR_CAF1,
        Assay.PCR_MULTIPLEX,
        Assay.WGS_SPECIES_ID,
    },
    "culture": {Assay.CULTURE, Assay.PHAGE_LYSIS, Assay.MALDI_TOF},
    "microscopy": {Assay.MICROSCOPY_BIPOLAR},
    "serology": {Assay.SEROLOGY_F1_PAIRED, Assay.SEROLOGY_F1_SINGLE},
}

#: No finite quantity of evidence justifies reporting certainty. The ceiling
#: is a statement about the model's limits, not about the organism.
PROBABILITY_CEILING = 0.99
PROBABILITY_FLOOR = 0.001


def family_of(assay: Assay) -> str:
    for family, members in ASSAY_FAMILIES.items():
        if assay in members:
            return family
    return "other"


def damped_sum(terms: list[float], damping: float, cap: float) -> float:
    """Combine correlated log-LR terms with geometric discounting.

    Positive and negative terms are ordered by magnitude and damped within
    their own sign, so a single strong negative still registers against a
    stack of positives.
    """
    if not terms:
        return 0.0
    positives = sorted((t for t in terms if t > 0), reverse=True)
    negatives = sorted((t for t in terms if t < 0))

    total = 0.0
    for i, term in enumerate(positives):
        total += term * (damping**i)
    for i, term in enumerate(negatives):
        total += term * (damping**i)
    return max(-cap, min(cap, total))

#: Specimens appropriate to each clinical form. Testing the wrong compartment
#: is a recognised source of false reassurance.
APPROPRIATE_SPECIMENS: dict[ClinicalForm, set[SpecimenType]] = {
    ClinicalForm.PNEUMONIC: {
        SpecimenType.SPUTUM,
        SpecimenType.BRONCHOALVEOLAR_LAVAGE,
        SpecimenType.BLOOD,
    },
    ClinicalForm.BUBONIC: {SpecimenType.BUBO_ASPIRATE, SpecimenType.BLOOD},
    ClinicalForm.SEPTICEMIC: {SpecimenType.BLOOD},
    ClinicalForm.PHARYNGEAL: {SpecimenType.THROAT_SWAB},
    ClinicalForm.MENINGEAL: {SpecimenType.CSF, SpecimenType.BLOOD},
}


class DiagnosticAgent(Agent):
    name = "diagnostic"
    description = "Scores whether the evidence supports Y. pestis, and typed form"

    def run(self, case: CaseRecord, context: dict[str, Verdict]) -> Verdict:
        rationale: list[str] = []
        flags = []

        log_odds = logit(BASE_PRETEST_PROBABILITY)
        rationale.append(
            f"Pre-test probability among patients tested for plague: "
            f"{BASE_PRETEST_PROBABILITY:.0%}"
        )

        log_odds += self._epi_contribution(case, rationale)
        log_odds += self._syndrome_contribution(case, rationale)
        log_odds += self._assay_contribution(case, rationale, flags)

        raw_probability = inv_logit(log_odds)
        probability = min(max(raw_probability, PROBABILITY_FLOOR), PROBABILITY_CEILING)
        if raw_probability > PROBABILITY_CEILING:
            rationale.append(
                f"Raw posterior {raw_probability:.4f} capped at "
                f"{PROBABILITY_CEILING:.2f} — the model does not express "
                "certainty, because residual risks (specimen mix-up, "
                "misidentification, data-entry error) are not in it."
            )

        classification = self._classify(case)
        form_call, form_note = self._infer_form(case)

        flags.extend(self._specimen_flags(case))

        if case.treatment_started and not case.has_positive(Assay.CULTURE):
            flags.append(
                self.flag(
                    "CULTURE_AFTER_ANTIBIOTICS",
                    Severity.MODERATE,
                    "Antibiotics started before or without culture recovery",
                    "Culture sensitivity falls sharply after therapy begins; a "
                    "negative culture here is weak evidence against plague, and "
                    "no isolate means no phenotypic susceptibility testing.",
                )
            )

        likelihood = self._to_likelihood(probability)
        confidence = self._confidence(case, probability)

        if form_note:
            rationale.append(form_note)
        rationale.append(
            f"Posterior probability of plague: {probability:.1%} "
            f"→ {likelihood.value}"
        )
        rationale.append(
            f"Surveillance case classification applied independently: "
            f"{classification.value}"
        )

        return self.verdict(
            headline=f"Plague likelihood: {likelihood.value}",
            score=probability,
            confidence=confidence,
            rationale=rationale,
            flags=flags,
            citations=[CIT_CDC_CASE_DEF, CIT_WHO_LAB_MANUAL, CIT_CDC_PLAGUE_DX],
            data={
                "probability": probability,
                "log_odds": log_odds,
                "likelihood": likelihood.value,
                "classification": classification.value,
                "clinical_form": form_call.value,
                "assays_tested": sorted(a.value for a in case.tested_assays()),
                "assays_pending": sorted(a.value for a in case.pending_assays()),
            },
        )

    # -- model terms -----------------------------------------------------

    def _epi_contribution(self, case: CaseRecord, rationale: list[str]) -> float:
        exp = case.exposure
        terms: list[float] = []

        def add(lr: float, note: str) -> None:
            terms.append(math.log(lr))
            rationale.append(f"{note} (×{lr:.1f} odds, before damping)")

        if exp.plague_endemic_area:
            add(
                4.0,
                "Endemic-area exposure"
                + (f": {exp.region_note}" if exp.region_note else ""),
            )
        elif exp.plague_endemic_area is False:
            add(0.35, "No endemic-area exposure reported")

        if exp.contact_with_confirmed_case:
            add(6.0, "Contact with a confirmed case")
        if exp.rodent_or_flea_contact:
            add(2.5, "Rodent or flea contact")
        if exp.hunting_or_skinning:
            add(3.0, "Hunting/skinning exposure")
        if exp.contact_with_sick_animal:
            add(2.0, "Contact with a sick or dead animal")
        if exp.laboratory_exposure:
            add(5.0, "Laboratory exposure reported")
        if exp.cluster_size and exp.cluster_size > 1:
            add(
                min(1.0 + 0.6 * exp.cluster_size, 5.0),
                f"Epidemiologically linked cluster of {exp.cluster_size} cases",
            )

        total = damped_sum(terms, EPI_DAMPING, EPI_CAP)
        if len(terms) > 1:
            rationale.append(
                f"Epidemiological evidence damped for correlation "
                f"({len(terms)} overlapping factors) → ×{math.exp(total):.2f} odds"
            )
        return total

    def _syndrome_contribution(
        self, case: CaseRecord, rationale: list[str]
    ) -> float:
        c = case.clinical
        terms: list[float] = []

        def add(lr: float, note: str) -> None:
            terms.append(math.log(lr))
            rationale.append(f"{note} (×{lr:.1f} odds, before damping)")

        if c.bubo_present:
            add(5.0, "Bubo present")
        elif c.lymphadenopathy:
            add(1.8, "Regional lymphadenopathy")

        if c.fever_c is not None and c.fever_c >= 38.0:
            add(1.6, f"Febrile at {c.fever_c:.1f}°C")
        elif c.fever_c is not None and c.fever_c < 37.5:
            add(0.5, "Afebrile")

        if c.hemoptysis and c.form == ClinicalForm.PNEUMONIC:
            add(3.0, "Haemoptysis with pneumonic presentation")
        elif c.hemoptysis:
            add(2.0, "Haemoptysis")

        if c.rapid_deterioration:
            add(2.2, "Fulminant course")
        if c.septic_shock:
            add(1.7, "Septic shock")

        # Plague's incubation is short; a long prodrome argues against it.
        if (
            c.days_symptomatic_at_presentation is not None
            and c.days_symptomatic_at_presentation > 10
        ):
            add(
                0.35,
                f"Symptomatic {c.days_symptomatic_at_presentation} days before "
                "presentation — longer than the typical course",
            )

        total = damped_sum(terms, SYNDROME_DAMPING, SYNDROME_CAP)
        if len(terms) > 1:
            rationale.append(
                f"Syndrome features damped for correlation ({len(terms)} "
                f"features) → ×{math.exp(total):.2f} odds"
            )
        return total

    def _assay_contribution(
        self, case: CaseRecord, rationale: list[str], flags: list
    ) -> float:
        by_family: dict[str, list[float]] = {}

        for result in case.diagnostics:
            profile = profile_for(result.assay)
            if profile is None:
                continue
            if result.result == ResultValue.POSITIVE:
                lr = profile.lr_positive
            elif result.result == ResultValue.NEGATIVE:
                lr = profile.lr_negative
            else:
                continue

            weight = lab_tier_weight(result.lab_tier) * profile.field_tier_discount
            contribution = weight * math.log(lr)
            by_family.setdefault(family_of(result.assay), []).append(contribution)

            rationale.append(
                f"{profile.display_name} {result.result.value} "
                f"(LR {lr:.1f}, lab weight {weight:.2f}) "
                f"→ ×{math.exp(contribution):.2f} odds, before damping"
            )

        # Damp hard within a family (same specimen, shared failure modes),
        # lightly across families (closer to genuinely independent evidence).
        family_totals: list[float] = []
        for family, terms in by_family.items():
            subtotal = damped_sum(terms, WITHIN_FAMILY_DAMPING, ASSAY_CAP)
            family_totals.append(subtotal)
            if len(terms) > 1:
                rationale.append(
                    f"{len(terms)} correlated {family} results combined "
                    f"→ ×{math.exp(subtotal):.2f} odds"
                )

        total = damped_sum(family_totals, ACROSS_FAMILY_DAMPING, ASSAY_CAP)
        if len(family_totals) > 1:
            rationale.append(
                f"{len(family_totals)} independent assay families combined "
                f"→ ×{math.exp(total):.2f} odds"
            )
        return total

    # -- classification --------------------------------------------------

    def _classify(self, case: CaseRecord) -> CaseClassification:
        """Apply surveillance case-definition criteria directly."""
        confirmatory_positive = any(
            d.result == ResultValue.POSITIVE
            and (p := profile_for(d.assay))
            and p.confirmatory
            for d in case.diagnostics
        )
        if confirmatory_positive:
            return CaseClassification.CONFIRMED

        presumptive_assays = {
            Assay.F1_RDT,
            Assay.PCR_PLA,
            Assay.PCR_CAF1,
            Assay.IMMUNOFLUORESCENCE,
            Assay.SEROLOGY_F1_SINGLE,
            Assay.MALDI_TOF,
        }
        if any(
            d.assay in presumptive_assays and d.result == ResultValue.POSITIVE
            for d in case.diagnostics
        ):
            return CaseClassification.PRESUMPTIVE

        c = case.clinical
        compatible = bool(
            c.bubo_present
            or c.form in (ClinicalForm.PNEUMONIC, ClinicalForm.SEPTICEMIC)
            or (c.fever_c and c.fever_c >= 38.0 and c.symptoms)
        )
        epi_link = bool(
            case.exposure.plague_endemic_area
            or case.exposure.contact_with_confirmed_case
            or case.exposure.rodent_or_flea_contact
            or case.exposure.laboratory_exposure
        )
        if compatible and epi_link:
            return CaseClassification.SUSPECTED

        if any(d.result == ResultValue.POSITIVE for d in case.diagnostics):
            return CaseClassification.SUSPECTED

        return CaseClassification.NOT_A_CASE

    def _infer_form(self, case: CaseRecord) -> tuple[ClinicalForm, str | None]:
        c = case.clinical
        if c.form != ClinicalForm.UNKNOWN:
            return c.form, None

        respiratory = bool(c.cough or c.hemoptysis or c.dyspnea or c.chest_imaging)
        if respiratory and c.bubo_present:
            return (
                ClinicalForm.PNEUMONIC,
                "Form not stated; respiratory involvement alongside a bubo is "
                "consistent with secondary pneumonic plague — treat as "
                "transmissible pending clarification.",
            )
        if respiratory:
            return (
                ClinicalForm.PNEUMONIC,
                "Form not stated; respiratory features present, so handled as "
                "pneumonic for infection-control purposes.",
            )
        if c.bubo_present:
            return ClinicalForm.BUBONIC, "Form inferred from bubo."
        return ClinicalForm.UNKNOWN, None

    def _specimen_flags(self, case: CaseRecord) -> list:
        form, _ = self._infer_form(case)
        wanted = APPROPRIATE_SPECIMENS.get(form)
        if not wanted:
            return []
        sampled = {
            d.specimen
            for d in case.diagnostics
            if d.specimen and d.result != ResultValue.NOT_DONE
        }
        if sampled and not (sampled & wanted):
            return [
                self.flag(
                    "SPECIMEN_MISMATCH",
                    Severity.HIGH,
                    f"No specimen appropriate to {form.value} plague has been tested",
                    f"Tested: {', '.join(sorted(s.value for s in sampled))}. "
                    f"Expected one of: "
                    f"{', '.join(sorted(s.value for s in wanted))}. "
                    "Negative results from the wrong compartment do not "
                    "exclude disease.",
                )
            ]
        return []

    # -- presentation ----------------------------------------------------

    @staticmethod
    def _to_likelihood(p: float) -> Likelihood:
        if p >= 0.85:
            return Likelihood.VERY_HIGH
        if p >= 0.60:
            return Likelihood.HIGH
        if p >= 0.30:
            return Likelihood.MODERATE
        if p >= 0.08:
            return Likelihood.LOW
        return Likelihood.VERY_LOW

    @staticmethod
    def _confidence(case: CaseRecord, probability: float) -> Confidence:
        n_informative = len(case.tested_assays())
        has_confirmatory = any(
            (p := profile_for(d.assay))
            and p.confirmatory
            and d.result in (ResultValue.POSITIVE, ResultValue.NEGATIVE)
            for d in case.diagnostics
        )
        decisive = probability >= 0.85 or probability <= 0.05

        if has_confirmatory and n_informative >= 2 and decisive:
            return Confidence.HIGH
        if n_informative >= 2:
            return Confidence.MODERATE
        return Confidence.LOW
