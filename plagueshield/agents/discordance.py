"""Discordance Agent — flags contradictions between independent evidence streams.

This is the agent that earns the system its keep. Any single stream can be
wrong quietly; two streams disagreeing is *loud*, and the disagreement itself
carries information that neither stream has alone.

The canonical case, and the one the brief names: a genome that predicts
susceptibility while the laboratory reports resistance. The correct response is
never to pick the more convenient answer. It is to surface the conflict, name
which result governs clinically in the interim, and route to a human.

Rules are grouped by the pair of streams they compare:

  genotype × phenotype   — the resistance-critical pair
  molecular × culture    — identification consistency
  screening × confirmatory
  clinical × laboratory  — syndrome versus what was actually tested
  temporal               — sequences that cannot have happened

Each rule is independent and additive. None of them silences another.
"""

from __future__ import annotations

from ..knowledge.regimens import CORE_REGIMEN_CLASSES, role_for
from ..models import (
    Assay,
    CaseRecord,
    ClinicalForm,
    Confidence,
    DrugClass,
    Flag,
    Interpretation,
    ResultValue,
    Severity,
    SpecimenType,
    Verdict,
)
from .base import Agent


class DiscordanceAgent(Agent):
    name = "discordance"
    description = "Flags contradictions between genomic, phenotypic and clinical evidence"
    depends_on = ("diagnostic", "resistance")

    def run(self, case: CaseRecord, context: dict[str, Verdict]) -> Verdict:
        flags: list[Flag] = []
        rationale: list[str] = []

        flags += self._genotype_vs_phenotype(case, rationale)
        flags += self._molecular_vs_culture(case, rationale)
        flags += self._screening_vs_confirmatory(case, rationale)
        flags += self._clinical_vs_laboratory(case, rationale)
        flags += self._temporal(case, rationale)
        flags += self._agent_cross_check(case, context, rationale)

        critical = [f for f in flags if f.severity == Severity.CRITICAL]
        high = [f for f in flags if f.severity == Severity.HIGH]

        if critical:
            headline = f"Discordance: {len(critical)} critical conflict(s)"
        elif high:
            headline = f"Discordance: {len(high)} significant conflict(s)"
        elif flags:
            headline = f"Discordance: {len(flags)} minor inconsistency(ies)"
        else:
            headline = "Discordance: none detected"
            rationale.append(
                "Evidence streams are mutually consistent across all "
                "applicable checks."
            )

        return self.verdict(
            headline=headline,
            score=float(len(critical) * 3 + len(high)),
            confidence=Confidence.HIGH if case.susceptibility else Confidence.MODERATE,
            rationale=rationale,
            flags=flags,
            data={
                "n_flags": len(flags),
                "n_critical": len(critical),
                "n_high": len(high),
                "requires_human_review": bool(critical or high),
                "conflicts": [
                    {
                        "code": f.code,
                        "severity": f.severity.value,
                        "message": f.message,
                        "detail": f.detail,
                    }
                    for f in flags
                ],
            },
        )

    # -- genotype × phenotype -------------------------------------------

    def _genotype_vs_phenotype(
        self, case: CaseRecord, rationale: list[str]
    ) -> list[Flag]:
        genomic = case.genomic
        if not genomic or not genomic.amr_screen_performed or not case.susceptibility:
            if case.susceptibility and not (genomic and genomic.amr_screen_performed):
                rationale.append(
                    "Genotype/phenotype comparison not possible: phenotypic "
                    "results exist but no genomic AMR screen was performed."
                )
            return []

        determinant_classes = genomic.classes_with_determinants()
        flags: list[Flag] = []

        by_class: dict[DrugClass, list] = {}
        for r in case.susceptibility:
            by_class.setdefault(r.drug_class, []).append(r)

        for cls, results in by_class.items():
            role = role_for(cls)
            has_determinant = cls in determinant_classes
            resistant = [
                r
                for r in results
                if r.interpretation
                in (Interpretation.RESISTANT, Interpretation.NONSUSCEPTIBLE)
            ]
            core = cls in CORE_REGIMEN_CLASSES

            # The headline conflict from the brief.
            if resistant and not has_determinant:
                confirmed = any(r.confirmed_by_repeat for r in resistant)
                flags.append(
                    self.flag(
                        "GENOTYPE_SUSCEPTIBLE_PHENOTYPE_RESISTANT",
                        Severity.CRITICAL if core else Severity.MODERATE,
                        f"{role.display_name}: laboratory reports "
                        f"non-susceptible, genomic screen called no determinant "
                        f"for this class",
                        "Phenotype governs clinical decisions in the interim. "
                        "A screen only detects catalogued determinants, so this "
                        "pattern is expected if the mechanism is uncatalogued — "
                        "but it is equally consistent with a testing artefact. "
                        + (
                            "Result is confirmed by repeat testing, which "
                            "argues against artefact."
                            if confirmed
                            else "Result is NOT confirmed by repeat testing; "
                            "repeat on a fresh subculture at a reference "
                            "laboratory before acting on it."
                        ),
                    )
                )
                rationale.append(
                    f"CONFLICT — {role.display_name}: phenotype non-susceptible, "
                    f"genotype silent."
                )

            # The inverse: determinant present, isolate tests susceptible.
            if has_determinant and results and not resistant:
                flags.append(
                    self.flag(
                        "GENOTYPE_RESISTANT_PHENOTYPE_SUSCEPTIBLE",
                        Severity.HIGH if core else Severity.LOW,
                        f"{role.display_name}: determinant called but isolate "
                        f"tests susceptible",
                        "A detected determinant is not necessarily expressed. "
                        "Phenotype governs current management, but the "
                        "determinant is a genuine finding: it may be "
                        "inducible, may be silent, or may be carried on a "
                        "mobile element. Retain the isolate and report to the "
                        "reference laboratory.",
                    )
                )
                rationale.append(
                    f"CONFLICT — {role.display_name}: genotype positive, "
                    f"phenotype susceptible."
                )

        # Core classes with a determinant but never phenotypically tested.
        untested_with_determinant = [
            cls
            for cls in determinant_classes & set(CORE_REGIMEN_CLASSES)
            if cls not in by_class
        ]
        if untested_with_determinant:
            names = ", ".join(role_for(c).display_name for c in untested_with_determinant)
            flags.append(
                self.flag(
                    "DETERMINANT_WITHOUT_PHENOTYPE",
                    Severity.HIGH,
                    f"Determinant called in {names} but the class was never "
                    f"phenotypically tested",
                    "The genotypic signal cannot be adjudicated without a "
                    "phenotypic result in the same class.",
                )
            )
            rationale.append(f"GAP — determinant in {names} with no phenotype.")

        return flags

    # -- molecular × culture --------------------------------------------

    def _molecular_vs_culture(
        self, case: CaseRecord, rationale: list[str]
    ) -> list[Flag]:
        flags: list[Flag] = []
        molecular_positive = any(
            case.has_positive(a)
            for a in (Assay.PCR_PLA, Assay.PCR_CAF1, Assay.PCR_MULTIPLEX)
        )
        culture_results = case.results_for(Assay.CULTURE)
        culture_negative = any(
            d.result == ResultValue.NEGATIVE for d in culture_results
        )

        if molecular_positive and culture_negative:
            severity = Severity.LOW if case.treatment_started else Severity.MODERATE
            flags.append(
                self.flag(
                    "PCR_POSITIVE_CULTURE_NEGATIVE",
                    severity,
                    "PCR positive but culture did not grow",
                    (
                        "Expected when antibiotics preceded sampling — PCR "
                        "detects non-viable organism. "
                        if case.treatment_started
                        else "No antibiotic exposure recorded, so this "
                        "combination is less readily explained. Review "
                        "specimen handling and transport conditions. "
                    )
                    + "Clinically important either way: without an isolate "
                    "there can be no phenotypic susceptibility testing.",
                )
            )
            rationale.append("CONFLICT — molecular positive, culture negative.")

        if case.has_positive(Assay.CULTURE) and any(
            d.result == ResultValue.NEGATIVE
            for a in (Assay.PCR_PLA, Assay.PCR_CAF1, Assay.PCR_MULTIPLEX)
            for d in case.results_for(a)
        ):
            flags.append(
                self.flag(
                    "CULTURE_POSITIVE_PCR_NEGATIVE",
                    Severity.HIGH,
                    "Culture positive but a molecular target was negative",
                    "Raises the possibility of an atypical isolate, a variant "
                    "affecting the primer target, or a misidentified organism. "
                    "Confirm species identification at a reference laboratory.",
                )
            )
            rationale.append("CONFLICT — culture positive, PCR negative.")

        return flags

    def _screening_vs_confirmatory(
        self, case: CaseRecord, rationale: list[str]
    ) -> list[Flag]:
        flags: list[Flag] = []
        rdt_positive = case.has_positive(Assay.F1_RDT)
        confirmatory_negative = any(
            d.result == ResultValue.NEGATIVE
            for a in (Assay.PCR_MULTIPLEX, Assay.CULTURE, Assay.WGS_SPECIES_ID)
            for d in case.results_for(a)
        )
        if rdt_positive and confirmatory_negative:
            flags.append(
                self.flag(
                    "RDT_POSITIVE_CONFIRMATORY_NEGATIVE",
                    Severity.MODERATE,
                    "Rapid test positive but confirmatory testing negative",
                    "The dipstick's specificity is moderate, so false "
                    "positives occur. Do not discard the case on this alone — "
                    "reconcile against the clinical picture and consider "
                    "repeat sampling.",
                )
            )
            rationale.append("CONFLICT — RDT positive, confirmatory negative.")

        if case.genomic and case.genomic.species_call:
            call = case.genomic.species_call.lower()
            if "pseudotuberculosis" in call and (
                rdt_positive or case.has_positive(Assay.PCR_PLA)
            ):
                flags.append(
                    self.flag(
                        "SPECIES_CALL_CONFLICT",
                        Severity.CRITICAL,
                        "Sequencing calls a species other than Y. pestis while "
                        "plague-targeted assays are positive",
                        f"Genomic species call: {case.genomic.species_call}. "
                        "These organisms are closely related and are a known "
                        "identification pitfall in both directions. Resolve at "
                        "a reference laboratory before the case is classified.",
                    )
                )
                rationale.append("CONFLICT — species call contradicts assays.")

        return flags

    def _clinical_vs_laboratory(
        self, case: CaseRecord, rationale: list[str]
    ) -> list[Flag]:
        flags: list[Flag] = []
        c = case.clinical

        respiratory = bool(c.cough or c.hemoptysis or c.dyspnea)
        if (c.form == ClinicalForm.PNEUMONIC or respiratory) and not any(
            d.specimen
            in (SpecimenType.SPUTUM, SpecimenType.BRONCHOALVEOLAR_LAVAGE)
            for d in case.diagnostics
        ):
            flags.append(
                self.flag(
                    "PNEUMONIC_WITHOUT_RESPIRATORY_SPECIMEN",
                    Severity.HIGH,
                    "Respiratory presentation but no respiratory specimen tested",
                    "Pneumonic plague is person-to-person transmissible and "
                    "drives infection-control and prophylaxis decisions for "
                    "contacts. Obtain sputum or BAL.",
                )
            )
            rationale.append("GAP — respiratory syndrome, no respiratory specimen.")

        if c.bubo_present and not any(
            d.specimen == SpecimenType.BUBO_ASPIRATE for d in case.diagnostics
        ):
            flags.append(
                self.flag(
                    "BUBO_NOT_ASPIRATED",
                    Severity.MODERATE,
                    "Bubo documented but no bubo aspirate tested",
                    "Aspirate is the highest-yield specimen in bubonic "
                    "presentation and the most likely route to an isolate.",
                )
            )
            rationale.append("GAP — bubo present, not aspirated.")

        if case.susceptibility and not case.has_positive(Assay.CULTURE):
            flags.append(
                self.flag(
                    "AST_WITHOUT_ISOLATE",
                    Severity.HIGH,
                    "Susceptibility results reported without a positive culture "
                    "on record",
                    "Phenotypic susceptibility testing requires a viable "
                    "isolate. Either the culture result is missing from this "
                    "record or the results belong to a different specimen — "
                    "reconcile before relying on them.",
                )
            )
            rationale.append("CONFLICT — AST present without recorded isolate.")

        return flags

    def _temporal(self, case: CaseRecord, rationale: list[str]) -> list[Flag]:
        flags: list[Flag] = []
        for d in case.diagnostics:
            if d.collected_at and d.reported_at and d.reported_at < d.collected_at:
                flags.append(
                    self.flag(
                        "IMPOSSIBLE_TIMELINE",
                        Severity.MODERATE,
                        f"{d.assay.value} reported before it was collected",
                        f"Collected {d.collected_at.isoformat()}, reported "
                        f"{d.reported_at.isoformat()}. Data-entry error likely; "
                        "the record cannot be audited until corrected.",
                    )
                )
                rationale.append(f"DATA ERROR — {d.assay.value} timeline inverted.")

        c = case.clinical
        if c.onset_date and c.presentation_date and c.presentation_date < c.onset_date:
            flags.append(
                self.flag(
                    "IMPOSSIBLE_TIMELINE",
                    Severity.MODERATE,
                    "Presentation recorded before symptom onset",
                    f"Onset {c.onset_date}, presentation {c.presentation_date}.",
                )
            )
            rationale.append("DATA ERROR — presentation precedes onset.")

        return flags

    def _agent_cross_check(
        self, case: CaseRecord, context: dict[str, Verdict], rationale: list[str]
    ) -> list[Flag]:
        """Contradictions between what the upstream agents concluded."""
        flags: list[Flag] = []
        diagnostic = context.get("diagnostic")
        resistance = context.get("resistance")
        if not diagnostic:
            return flags

        probability = diagnostic.score or 0.0
        classification = diagnostic.data.get("classification")

        if probability >= 0.75 and classification in ("not_a_case", "suspected"):
            flags.append(
                self.flag(
                    "PROBABILITY_CLASSIFICATION_GAP",
                    Severity.MODERATE,
                    "High modelled probability but the case definition is not "
                    "yet met",
                    f"Model probability {probability:.0%} versus surveillance "
                    f"classification '{classification}'. This is not an error — "
                    "it means confirmatory laboratory evidence is the missing "
                    "piece, and it should be pursued rather than assumed.",
                )
            )
            rationale.append("TENSION — probability outruns case classification.")

        if probability < 0.15 and resistance and not resistance.abstained:
            anomaly = resistance.data.get("anomaly")
            if anomaly in ("Suspected", "Confirmed"):
                flags.append(
                    self.flag(
                        "RESISTANCE_CALL_ON_UNLIKELY_PLAGUE",
                        Severity.HIGH,
                        "Resistance anomaly reported on a case unlikely to be "
                        "plague",
                        f"Plague probability is {probability:.0%}. Confirm the "
                        "isolate's identity before interpreting its "
                        "susceptibility profile as a plague finding — the "
                        "result may belong to a different organism entirely.",
                    )
                )
                rationale.append("TENSION — resistance call on low-probability case.")

        return flags
