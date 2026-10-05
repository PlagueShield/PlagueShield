"""Resistance Agent — "Is the isolate showing unusual antimicrobial resistance?"

Combines three sources, in this order of authority:

  1. **Phenotype.** A confirmed non-susceptible result in a core therapeutic
     class is the governing finding. Nothing downstream overrides it.
  2. **Genotype, at drug-class level.** Determinant calls come from an
     external tool (see knowledge/regimens.py for why the mechanism catalogue
     lives there and not here). A class-level call raises concern for that
     class; it does not by itself establish resistance.
  3. **Few-shot pattern match.** Similarity to the labelled exemplar bank,
     which carries the rare positives that no fitted model could learn.

The agent reports per-class outlooks and an overall position, and defers
entirely to the OOD gate: if the evidence cannot support a call, it abstains
and says so rather than defaulting to "susceptible". Defaulting to susceptible
on absent evidence is the specific failure this agent exists to prevent.
"""

from __future__ import annotations

import math

from ..fewshot.exemplars import extract_features, few_shot_vote
from ..fewshot.ood import assess as ood_assess
from ..fewshot.prior import EvidenceUpdate, ResistancePosterior
from ..knowledge.guidance import CIT_AMRFINDER, CIT_GUIYOULE_1997, CIT_NCBI_PATHOGEN
from ..knowledge.regimens import (
    CIT_CLSI_M45,
    CORE_REGIMEN_CLASSES,
    CORE_REGIMEN_WEIGHT,
    TreatmentRole,
    clinically_meaningful,
    role_for,
)
from ..models import (
    AnomalyStatus,
    CaseRecord,
    Confidence,
    DrugClass,
    Interpretation,
    Severity,
    SusceptibilityOutlook,
    Verdict,
)
from .base import Agent

#: Log-likelihood-ratio contributions. Magnitudes encode the authority order
#: above: a confirmed phenotype moves the posterior far more than a genotype
#: call, and a clean complete screen is only modestly reassuring because
#: absence of a known determinant does not exclude an unknown one.
LLR_PHENOTYPE_NONSUSC_CONFIRMED = math.log(60.0)
LLR_PHENOTYPE_NONSUSC_UNCONFIRMED = math.log(12.0)
LLR_PHENOTYPE_INTERMEDIATE = math.log(4.0)
LLR_DETERMINANT_CORE_CLASS = math.log(15.0)
LLR_DETERMINANT_NONCORE_CLASS = math.log(1.6)
LLR_CLEAN_SCREEN_GOOD_QUALITY = math.log(0.45)
LLR_FULL_PHENOTYPE_SUSCEPTIBLE = math.log(0.25)


class ResistanceAgent(Agent):
    name = "resistance"
    description = (
        "Few-shot resistance inference over genomic and phenotypic evidence"
    )

    def run(self, case: CaseRecord, context: dict[str, Verdict]) -> Verdict:
        genomic = case.genomic
        susceptibility = case.susceptibility

        vector = extract_features(genomic, susceptibility)
        few_shot = few_shot_vote(vector)
        ood = ood_assess(few_shot, genomic, susceptibility)

        posterior = ResistancePosterior()
        rationale: list[str] = []
        flags = []

        per_class = self._per_class_outlook(case, posterior, flags)
        self._apply_global_evidence(case, posterior)

        # The few-shot vote enters as evidence in its own right, scaled by how
        # much support the bank actually offers. A weak match contributes
        # little rather than contributing noise.
        if few_shot.support_mass >= 0.35:
            centred = (few_shot.score - 0.5) * 2.0
            scale = min(few_shot.support_mass / 2.0, 1.0)
            llr = centred * scale * math.log(8.0)
            posterior.apply(
                EvidenceUpdate(
                    source="Few-shot pattern match",
                    log_likelihood_ratio=llr,
                    description=(
                        f"vote {few_shot.score:.2f} over {len(few_shot.neighbours[:5])} "
                        f"nearest patterns (support mass {few_shot.support_mass:.2f})"
                    ),
                )
            )

        probability = posterior.probability
        rationale.extend(posterior.explain())
        rationale.extend(few_shot.explain())

        # -- abstention -------------------------------------------------
        if ood.abstain:
            flags.append(
                self.flag(
                    "RESISTANCE_ABSTAIN",
                    Severity.MODERATE,
                    "Resistance call withheld — evidence insufficient or "
                    "out of distribution",
                    ood.summary(),
                )
            )
            return self.verdict(
                headline="Resistance anomaly: Insufficient evidence",
                score=probability,
                confidence=Confidence.LOW,
                abstained=True,
                abstain_reason=ood.summary(),
                rationale=rationale
                + [f"Abstained: {m}" for m in ood.messages],
                flags=flags,
                citations=[CIT_AMRFINDER, CIT_CLSI_M45, CIT_NCBI_PATHOGEN],
                data=self._data(
                    probability,
                    per_class,
                    few_shot,
                    ood,
                    SusceptibilityOutlook.INSUFFICIENT_EVIDENCE,
                    AnomalyStatus.INSUFFICIENT_EVIDENCE,
                ),
            )

        outlook = self._overall_outlook(per_class, ood)
        anomaly = self._anomaly_status(case, per_class, probability)
        confidence = self._confidence(ood, few_shot, per_class)

        if anomaly in (AnomalyStatus.SUSPECTED, AnomalyStatus.CONFIRMED):
            threatened = [
                name
                for name, info in per_class.items()
                if info["concern"] and info["core"]
            ]
            flags.append(
                self.flag(
                    "RESISTANCE_ANOMALY",
                    Severity.CRITICAL
                    if anomaly == AnomalyStatus.CONFIRMED
                    else Severity.HIGH,
                    f"Resistance anomaly {anomaly.value.lower()} in a core "
                    f"therapeutic class",
                    f"Classes implicated: {', '.join(threatened) or 'none core'}. "
                    "Resistance in Y. pestis is rare and reportable; notify the "
                    "reference laboratory and public health authority.",
                )
            )

        return self.verdict(
            headline=f"Standard-treatment susceptibility: {outlook.value}",
            score=probability,
            confidence=confidence,
            rationale=rationale,
            flags=flags,
            citations=[
                CIT_AMRFINDER,
                CIT_CLSI_M45,
                CIT_NCBI_PATHOGEN,
                CIT_GUIYOULE_1997,
            ],
            data=self._data(
                probability, per_class, few_shot, ood, outlook, anomaly
            ),
        )

    # -- evidence --------------------------------------------------------

    def _per_class_outlook(
        self, case: CaseRecord, posterior: ResistancePosterior, flags: list
    ) -> dict[str, dict]:
        """Build a per-drug-class picture and accumulate class-level evidence."""
        genomic = case.genomic
        determinant_classes = (
            genomic.classes_with_determinants() if genomic else set()
        )
        screened = bool(genomic and genomic.amr_screen_performed)

        by_class: dict[DrugClass, list] = {}
        for result in case.susceptibility:
            by_class.setdefault(result.drug_class, []).append(result)

        out: dict[str, dict] = {}
        considered = set(CORE_REGIMEN_CLASSES) | determinant_classes | set(by_class)

        for cls in considered:
            role = role_for(cls)
            results = by_class.get(cls, [])
            has_determinant = cls in determinant_classes

            nonsusc = [r for r in results if r.is_nonsusceptible]
            resistant = [
                r
                for r in results
                if r.interpretation
                in (Interpretation.RESISTANT, Interpretation.NONSUSCEPTIBLE)
            ]
            confirmed = any(r.confirmed_by_repeat for r in nonsusc)

            # Phenotype evidence.
            if resistant and clinically_meaningful(cls):
                llr = (
                    LLR_PHENOTYPE_NONSUSC_CONFIRMED
                    if confirmed
                    else LLR_PHENOTYPE_NONSUSC_UNCONFIRMED
                )
                weight = role.regimen_weight
                posterior.apply(
                    EvidenceUpdate(
                        source=f"Phenotype — {role.display_name}",
                        log_likelihood_ratio=llr * max(weight, 0.4),
                        description=(
                            f"{len(resistant)} non-susceptible result(s)"
                            + (" , confirmed by repeat" if confirmed else
                               ", not yet confirmed by repeat")
                        ),
                    )
                )
            elif nonsusc and clinically_meaningful(cls):
                posterior.apply(
                    EvidenceUpdate(
                        source=f"Phenotype — {role.display_name}",
                        log_likelihood_ratio=LLR_PHENOTYPE_INTERMEDIATE
                        * max(role.regimen_weight, 0.4),
                        description="intermediate-category result",
                    )
                )

            # Genotype evidence, class level only.
            if has_determinant:
                core = cls in CORE_REGIMEN_CLASSES
                posterior.apply(
                    EvidenceUpdate(
                        source=f"Genotype — {role.display_name}",
                        log_likelihood_ratio=(
                            LLR_DETERMINANT_CORE_CLASS
                            if core
                            else LLR_DETERMINANT_NONCORE_CLASS
                        ),
                        description=(
                            f"external tool called {sum(1 for d in genomic.amr_determinants if d.drug_class == cls)} "
                            f"determinant(s) affecting this class"
                            + ("" if core else " (not a core plague class)")
                        ),
                    )
                )

            concern = bool(resistant) or (has_determinant and cls in CORE_REGIMEN_CLASSES)

            if cls in CORE_REGIMEN_CLASSES and not results and not screened:
                status = "Untested"
            elif cls in CORE_REGIMEN_CLASSES and not results and screened:
                status = "Genotype only"
            elif resistant:
                status = "Non-susceptible"
            elif nonsusc:
                status = "Intermediate"
            elif results:
                status = "Susceptible"
            elif has_determinant:
                status = "Determinant called, untested"
            else:
                status = "No evidence"

            out[role.display_name] = {
                "drug_class": cls.value,
                "role": role.role.value,
                "core": cls in CORE_REGIMEN_CLASSES,
                "status": status,
                "determinant_called": has_determinant,
                "phenotype_tested": bool(results),
                "nonsusceptible": bool(nonsusc),
                "resistant": bool(resistant),
                "confirmed_by_repeat": confirmed,
                "concern": concern,
                "mics": [
                    {
                        "antimicrobial": r.antimicrobial,
                        "mic_mg_l": r.mic_mg_l,
                        "operator": r.mic_operator,
                        "interpretation": r.interpretation.value,
                        "method": r.method.value,
                        "breakpoint_source": r.breakpoint_source,
                    }
                    for r in results
                ],
                "note": role.note,
            }

            if role.role == TreatmentRole.NOT_RECOMMENDED and results:
                flags.append(
                    self.flag(
                        "NONTHERAPEUTIC_CLASS_REPORTED",
                        Severity.LOW,
                        f"{role.display_name} susceptibility reported but the "
                        "class has no role in plague therapy",
                        role.note,
                    )
                )

        return out

    def _apply_global_evidence(
        self, case: CaseRecord, posterior: ResistancePosterior
    ) -> None:
        genomic = case.genomic
        if (
            genomic
            and genomic.amr_screen_performed
            and genomic.quality_sufficient
            and not genomic.classes_with_determinants() & set(CORE_REGIMEN_CLASSES)
        ):
            posterior.apply(
                EvidenceUpdate(
                    source="Genomic screen",
                    log_likelihood_ratio=LLR_CLEAN_SCREEN_GOOD_QUALITY,
                    description=(
                        "adequate-quality assembly with no determinants called "
                        "in any core class — modestly reassuring only, since a "
                        "screen cannot detect an uncatalogued mechanism"
                    ),
                )
            )

        tested = {r.drug_class for r in case.susceptibility}
        all_core_tested = set(CORE_REGIMEN_CLASSES).issubset(tested)
        all_core_susceptible = all_core_tested and not any(
            r.is_nonsusceptible
            for r in case.susceptibility
            if r.drug_class in CORE_REGIMEN_CLASSES
        )
        if all_core_susceptible:
            posterior.apply(
                EvidenceUpdate(
                    source="Phenotype panel",
                    log_likelihood_ratio=LLR_FULL_PHENOTYPE_SUSCEPTIBLE,
                    description="every core therapeutic class tested and "
                    "susceptible",
                )
            )

    # -- conclusions -----------------------------------------------------

    def _overall_outlook(
        self, per_class: dict[str, dict], ood
    ) -> SusceptibilityOutlook:
        core = [i for i in per_class.values() if i["core"]]
        if not core:
            return SusceptibilityOutlook.INSUFFICIENT_EVIDENCE

        threatened_weight = sum(
            role_for(DrugClass(i["drug_class"])).regimen_weight
            for i in core
            if i["resistant"]
        )
        tested = [i for i in core if i["phenotype_tested"]]
        susceptible = [i for i in tested if not i["nonsusceptible"]]

        if threatened_weight >= CORE_REGIMEN_WEIGHT * 0.66:
            return SusceptibilityOutlook.COMPROMISED
        if threatened_weight > 0:
            return SusceptibilityOutlook.POSSIBLY_COMPROMISED

        determinant_in_core = any(i["determinant_called"] for i in core)
        if determinant_in_core:
            return SusceptibilityOutlook.UNCERTAIN

        if len(susceptible) == len(core) and len(tested) == len(core):
            return (
                SusceptibilityOutlook.PRESERVED
                if ood.reliability > 0.85
                else SusceptibilityOutlook.PROBABLY_PRESERVED
            )
        if susceptible and ood.completeness >= 0.5:
            return SusceptibilityOutlook.PROBABLY_PRESERVED
        return SusceptibilityOutlook.INSUFFICIENT_EVIDENCE

    def _anomaly_status(
        self, case: CaseRecord, per_class: dict[str, dict], probability: float
    ) -> AnomalyStatus:
        core = [i for i in per_class.values() if i["core"]]

        confirmed_resistant_core = any(
            i["resistant"] and i["confirmed_by_repeat"] for i in core
        )
        if confirmed_resistant_core:
            return AnomalyStatus.CONFIRMED

        unconfirmed_resistant_core = any(i["resistant"] for i in core)
        determinant_core = any(i["determinant_called"] for i in core)
        if unconfirmed_resistant_core or determinant_core:
            return AnomalyStatus.SUSPECTED

        if probability >= 0.25:
            return AnomalyStatus.SUSPECTED

        tested_core = [i for i in core if i["phenotype_tested"]]
        if len(tested_core) >= 2:
            return AnomalyStatus.NONE_DETECTED
        return AnomalyStatus.INSUFFICIENT_EVIDENCE

    @staticmethod
    def _confidence(ood, few_shot, per_class: dict[str, dict]) -> Confidence:
        core_tested = sum(
            1 for i in per_class.values() if i["core"] and i["phenotype_tested"]
        )
        if (
            ood.completeness >= 0.8
            and ood.reliability >= 0.85
            and few_shot.max_similarity >= 0.75
            and core_tested >= 3
        ):
            return Confidence.HIGH
        if ood.completeness >= 0.5 and ood.reliability >= 0.7:
            return Confidence.MODERATE
        return Confidence.LOW

    @staticmethod
    def _data(
        probability: float,
        per_class: dict[str, dict],
        few_shot,
        ood,
        outlook: SusceptibilityOutlook,
        anomaly: AnomalyStatus,
    ) -> dict:
        return {
            "resistance_probability": probability,
            "outlook": outlook.value,
            "anomaly": anomaly.value,
            "per_class": per_class,
            "few_shot": {
                "score": few_shot.score,
                "max_similarity": few_shot.max_similarity,
                "support_mass": few_shot.support_mass,
                "nearest": few_shot.nearest.exemplar.key
                if few_shot.nearest
                else None,
                "neighbours": [
                    {
                        "key": n.exemplar.key,
                        "label": n.exemplar.label,
                        "similarity": round(n.similarity, 4),
                        "description": n.exemplar.description,
                    }
                    for n in few_shot.neighbours[:5]
                ],
            },
            "ood": {
                "abstain": ood.abstain,
                "novelty": ood.novelty,
                "threshold": ood.support_threshold,
                "completeness": ood.completeness,
                "reliability": ood.reliability,
                "reasons": [r.value for r in ood.reasons],
                "messages": ood.messages,
            },
        }
