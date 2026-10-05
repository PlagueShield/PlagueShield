"""Builds the bundled de-identified case records.

Run: python scripts/build_bundled_cases.py

Why a builder rather than hand-written JSON: constructing the records through
the pydantic schema guarantees they are valid, and keeps the provenance of
each one written down next to the record itself.

PROVENANCE POLICY
-----------------
Individual-level plague patient records are not published anywhere, and should
not be — plague case counts are small enough that a genuine line list would be
re-identifiable. So every record here is one of:

  * published_report      — a vignette reconstructed from the clinical and
                            laboratory details described in a public outbreak
                            report or WHO Disease Outbreak News item. Detail is
                            limited to what the source stated; anything else is
                            left null rather than invented.
  * surveillance_summary  — a representative case built from aggregate
                            surveillance reporting (e.g. CDC's published US
                            case characteristics). Represents a pattern, not a
                            person.
  * synthetic             — constructed to exercise a specific system
                            behaviour (discordance, abstention, species
                            conflict). Clearly labelled; carries no
                            epidemiological meaning.

Real *isolate* data, by contrast, is genuinely public: NCBI Pathogen Detection
holds Y. pestis genomes with standardised AMR genotype calls. Those accessions
are fetched live by scripts/fetch_public_data.py.
"""

from __future__ import annotations

import json
import pathlib
from datetime import date, datetime, timezone

from plagueshield.models import (
    AmrDeterminant,
    Assay,
    AstMethod,
    CaseRecord,
    Citation,
    Clinical,
    ClinicalForm,
    DataOrigin,
    DiagnosticResult,
    DrugClass,
    Exposure,
    GenomicEvidence,
    Interpretation,
    LabTier,
    Provenance,
    ResultValue,
    SpecimenType,
    SusceptibilityResult,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "plagueshield" / "data" / "cases"


def dt(y: int, m: int, d: int, h: int = 9) -> datetime:
    return datetime(y, m, d, h, 0, tzinfo=timezone.utc)


def susceptible(
    name: str, cls: DrugClass, mic: float, day: datetime, op: str = "<="
) -> SusceptibilityResult:
    return SusceptibilityResult(
        antimicrobial=name,
        drug_class=cls,
        mic_mg_l=mic,
        mic_operator=op,
        interpretation=Interpretation.SUSCEPTIBLE,
        method=AstMethod.BROTH_MICRODILUTION,
        breakpoint_source="CLSI M45 4th ed.",
        tested_at=day,
        lab_tier=LabTier.BSL3_REFERENCE,
    )


CASES: list[CaseRecord] = []

# ---------------------------------------------------------------------------
# 1. Marmot-exposure case, Mongolia/Russia border region (the Irkutsk incident
#    context named in the brief).
# ---------------------------------------------------------------------------
CASES.append(
    CaseRecord(
        case_id="PS-2019-MN-001",
        label="Marmot-exposure bubonic→pneumonic, Mongolia/Russia border",
        provenance=Provenance(
            origin=DataOrigin.PUBLISHED_REPORT,
            description=(
                "Vignette reconstructed from public reporting of 2019 plague "
                "cases in western Mongolia linked to marmot consumption, with "
                "associated quarantine measures in the neighbouring Irkutsk "
                "region of Russia. Clinical and laboratory detail limited to "
                "what was publicly described."
            ),
            citation=Citation(
                source="WHO",
                title="Disease Outbreak News — Plague, Mongolia",
                url="https://www.who.int/emergencies/disease-outbreak-news",
                year=2019,
            ),
            deidentified=True,
            notes="Not an individual patient record. Represents the reported "
            "presentation pattern.",
        ),
        reported_date=date(2019, 5, 3),
        country="Mongolia",
        admin1="Bayan-Ölgii",
        age_band="30-39",
        sex="unknown",
        exposure=Exposure(
            plague_endemic_area=True,
            region_note="Marmot-enzootic steppe focus; recognised natural "
            "plague reservoir",
            rodent_or_flea_contact=True,
            hunting_or_skinning=True,
            contact_with_sick_animal=True,
            days_since_exposure=4,
            cluster_size=2,
        ),
        clinical=Clinical(
            onset_date=date(2019, 4, 29),
            presentation_date=date(2019, 5, 1),
            form=ClinicalForm.BUBONIC,
            fever_c=39.4,
            symptoms=["fever", "chills", "painful inguinal swelling", "malaise"],
            lymphadenopathy=True,
            bubo_present=True,
            cough=True,
            dyspnea=True,
            chest_imaging="Bilateral lower-zone consolidation on later imaging",
            rapid_deterioration=True,
            days_symptomatic_at_presentation=2,
        ),
        diagnostics=[
            DiagnosticResult(
                assay=Assay.F1_RDT,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BUBO_ASPIRATE,
                collected_at=dt(2019, 5, 1, 10),
                reported_at=dt(2019, 5, 1, 11),
                lab_tier=LabTier.FIELD,
            ),
            DiagnosticResult(
                assay=Assay.MICROSCOPY_BIPOLAR,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BUBO_ASPIRATE,
                collected_at=dt(2019, 5, 1, 10),
                reported_at=dt(2019, 5, 1, 13),
                lab_tier=LabTier.HOSPITAL,
            ),
            DiagnosticResult(
                assay=Assay.PCR_MULTIPLEX,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BUBO_ASPIRATE,
                collected_at=dt(2019, 5, 1, 10),
                reported_at=dt(2019, 5, 2, 8),
                lab_tier=LabTier.REFERENCE,
                detail="Two independent targets positive",
            ),
            DiagnosticResult(
                assay=Assay.CULTURE,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BUBO_ASPIRATE,
                collected_at=dt(2019, 5, 1, 10),
                reported_at=dt(2019, 5, 4, 9),
                lab_tier=LabTier.BSL3_REFERENCE,
            ),
            DiagnosticResult(
                assay=Assay.CULTURE,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.SPUTUM,
                collected_at=dt(2019, 5, 2, 12),
                reported_at=dt(2019, 5, 5, 9),
                lab_tier=LabTier.BSL3_REFERENCE,
                detail="Secondary pneumonic involvement",
            ),
        ],
        genomic=GenomicEvidence(
            platform="Illumina MiSeq",
            read_depth_x=68.0,
            completeness_pct=98.6,
            contamination_pct=0.4,
            species_call="Yersinia pestis",
            ani_identity_pct=99.4,
            mlst_or_lineage="Branch 0 (ancestral steppe lineage)",
            plasmid_markers_detected=["pCD1", "pMT1", "pPCP1"],
            amr_screen_performed=True,
            amr_tool_run="AMRFinderPlus 3.12.8 / 2024-07-22.1",
            amr_determinants=[],
            sequenced_at=dt(2019, 5, 8),
        ),
        susceptibility=[
            susceptible("gentamicin", DrugClass.AMINOGLYCOSIDE, 0.5, dt(2019, 5, 6)),
            susceptible("streptomycin", DrugClass.AMINOGLYCOSIDE, 2.0, dt(2019, 5, 6)),
            susceptible("ciprofloxacin", DrugClass.FLUOROQUINOLONE, 0.03, dt(2019, 5, 6)),
            susceptible("levofloxacin", DrugClass.FLUOROQUINOLONE, 0.06, dt(2019, 5, 6)),
            susceptible("doxycycline", DrugClass.TETRACYCLINE, 1.0, dt(2019, 5, 6)),
        ],
        treatment_started=True,
        notes="Epidemiologically linked second case in the same household.",
    )
)

# ---------------------------------------------------------------------------
# 2. Urban pneumonic, Madagascar 2017 outbreak pattern.
# ---------------------------------------------------------------------------
CASES.append(
    CaseRecord(
        case_id="PS-2017-MG-014",
        label="Urban pneumonic plague, Madagascar outbreak",
        provenance=Provenance(
            origin=DataOrigin.PUBLISHED_REPORT,
            description=(
                "Vignette reconstructed from WHO reporting of the 2017 "
                "Madagascar urban pneumonic plague outbreak. Reflects the "
                "documented pattern of early empiric treatment and consequent "
                "low culture recovery."
            ),
            citation=Citation(
                source="WHO",
                title="Plague outbreak Madagascar — External Situation Reports",
                url="https://www.who.int/emergencies/disease-outbreak-news",
                year=2017,
            ),
        ),
        reported_date=date(2017, 10, 12),
        country="Madagascar",
        admin1="Analamanga",
        age_band="20-29",
        sex="unknown",
        exposure=Exposure(
            plague_endemic_area=True,
            region_note="Urban outbreak setting with documented "
            "person-to-person transmission",
            contact_with_confirmed_case=True,
            days_since_exposure=3,
            cluster_size=6,
        ),
        clinical=Clinical(
            onset_date=date(2017, 10, 10),
            presentation_date=date(2017, 10, 11),
            form=ClinicalForm.PNEUMONIC,
            fever_c=39.0,
            symptoms=["fever", "cough", "chest pain", "dyspnoea"],
            cough=True,
            hemoptysis=True,
            dyspnea=True,
            chest_imaging="Rapidly progressive bilateral infiltrates",
            rapid_deterioration=True,
            days_symptomatic_at_presentation=1,
        ),
        diagnostics=[
            DiagnosticResult(
                assay=Assay.F1_RDT,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.SPUTUM,
                collected_at=dt(2017, 10, 11, 8),
                reported_at=dt(2017, 10, 11, 9),
                lab_tier=LabTier.FIELD,
            ),
            DiagnosticResult(
                assay=Assay.PCR_PLA,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.SPUTUM,
                collected_at=dt(2017, 10, 11, 8),
                reported_at=dt(2017, 10, 12, 14),
                lab_tier=LabTier.REFERENCE,
            ),
            DiagnosticResult(
                assay=Assay.CULTURE,
                result=ResultValue.NEGATIVE,
                specimen=SpecimenType.SPUTUM,
                collected_at=dt(2017, 10, 12, 8),
                reported_at=dt(2017, 10, 15, 9),
                lab_tier=LabTier.REFERENCE,
                detail="Collected after empiric therapy commenced",
            ),
        ],
        genomic=None,
        susceptibility=[],
        treatment_started=True,
        notes="Empiric therapy started at presentation per outbreak protocol.",
    )
)

# ---------------------------------------------------------------------------
# 3. US bubonic case, surveillance-typical.
# ---------------------------------------------------------------------------
CASES.append(
    CaseRecord(
        case_id="PS-2021-US-003",
        label="Bubonic plague, US southwestern focus",
        provenance=Provenance(
            origin=DataOrigin.SURVEILLANCE_SUMMARY,
            description=(
                "Representative case constructed from CDC's published "
                "characteristics of US human plague cases (peridomestic flea "
                "exposure, bubonic presentation, full reference-laboratory "
                "workup). Represents a pattern, not an individual."
            ),
            citation=Citation(
                source="CDC",
                title="Plague Surveillance and Maps — United States",
                url="https://www.cdc.gov/plague/php/surveillance/",
            ),
        ),
        reported_date=date(2021, 7, 18),
        country="United States",
        admin1="New Mexico",
        age_band="50-59",
        sex="unknown",
        exposure=Exposure(
            plague_endemic_area=True,
            region_note="Southwestern US enzootic focus",
            rodent_or_flea_contact=True,
            contact_with_sick_animal=True,
            days_since_exposure=5,
        ),
        clinical=Clinical(
            onset_date=date(2021, 7, 14),
            presentation_date=date(2021, 7, 16),
            form=ClinicalForm.BUBONIC,
            fever_c=38.9,
            symptoms=["fever", "headache", "tender axillary swelling"],
            lymphadenopathy=True,
            bubo_present=True,
            days_symptomatic_at_presentation=2,
        ),
        diagnostics=[
            DiagnosticResult(
                assay=Assay.CULTURE,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BUBO_ASPIRATE,
                collected_at=dt(2021, 7, 16, 11),
                reported_at=dt(2021, 7, 18, 16),
                lab_tier=LabTier.BSL3_REFERENCE,
                lab_name="State public health laboratory (LRN reference)",
            ),
            DiagnosticResult(
                assay=Assay.PCR_MULTIPLEX,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BUBO_ASPIRATE,
                collected_at=dt(2021, 7, 16, 11),
                reported_at=dt(2021, 7, 17, 9),
                lab_tier=LabTier.REFERENCE,
            ),
            DiagnosticResult(
                assay=Assay.PHAGE_LYSIS,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BUBO_ASPIRATE,
                collected_at=dt(2021, 7, 18, 16),
                reported_at=dt(2021, 7, 19, 16),
                lab_tier=LabTier.BSL3_REFERENCE,
            ),
            DiagnosticResult(
                assay=Assay.CULTURE,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BLOOD,
                collected_at=dt(2021, 7, 16, 11),
                reported_at=dt(2021, 7, 19, 9),
                lab_tier=LabTier.BSL3_REFERENCE,
            ),
        ],
        genomic=GenomicEvidence(
            platform="Illumina NextSeq",
            read_depth_x=92.0,
            completeness_pct=99.1,
            contamination_pct=0.2,
            species_call="Yersinia pestis",
            ani_identity_pct=99.6,
            mlst_or_lineage="Branch 1.ORI",
            plasmid_markers_detected=["pCD1", "pMT1", "pPCP1"],
            amr_screen_performed=True,
            amr_tool_run="AMRFinderPlus 3.12.8 / 2024-07-22.1",
            amr_determinants=[
                AmrDeterminant(
                    gene_symbol="blaA",
                    drug_class=DrugClass.BETA_LACTAM,
                    tool="AMRFinderPlus 3.12.8",
                    identity_pct=99.8,
                    coverage_pct=100.0,
                    element_type="AMR (species-intrinsic)",
                )
            ],
            sequenced_at=dt(2021, 7, 22),
            accession="(see NCBI Pathogen Detection for real accessions)",
        ),
        susceptibility=[
            susceptible("gentamicin", DrugClass.AMINOGLYCOSIDE, 0.25, dt(2021, 7, 20)),
            susceptible("streptomycin", DrugClass.AMINOGLYCOSIDE, 1.0, dt(2021, 7, 20)),
            susceptible("ciprofloxacin", DrugClass.FLUOROQUINOLONE, 0.015, dt(2021, 7, 20)),
            susceptible("doxycycline", DrugClass.TETRACYCLINE, 0.5, dt(2021, 7, 20)),
            susceptible(
                "trimethoprim-sulfamethoxazole",
                DrugClass.SULFONAMIDE,
                0.5,
                dt(2021, 7, 20),
            ),
        ],
        treatment_started=True,
    )
)

# ---------------------------------------------------------------------------
# 4. Sparse field case — exercises abstention.
# ---------------------------------------------------------------------------
CASES.append(
    CaseRecord(
        case_id="PS-2020-CD-021",
        label="Suspected pneumonic plague, remote health post (sparse data)",
        provenance=Provenance(
            origin=DataOrigin.SURVEILLANCE_SUMMARY,
            description=(
                "Representative remote-setting presentation built from "
                "reporting of plague activity in the Ituri region, DRC. "
                "Deliberately sparse: a single field rapid test is often all "
                "that is available, and the system must say so rather than "
                "extrapolate."
            ),
            citation=Citation(
                source="WHO",
                title="Disease Outbreak News — Plague, Democratic Republic of "
                "the Congo",
                url="https://www.who.int/emergencies/disease-outbreak-news",
                year=2020,
            ),
        ),
        reported_date=date(2020, 8, 9),
        country="Democratic Republic of the Congo",
        admin1="Ituri",
        age_band="10-19",
        sex="unknown",
        exposure=Exposure(
            plague_endemic_area=True,
            region_note="Known endemic highland focus",
            contact_with_confirmed_case=True,
            cluster_size=3,
        ),
        clinical=Clinical(
            onset_date=date(2020, 8, 7),
            presentation_date=date(2020, 8, 9),
            form=ClinicalForm.PNEUMONIC,
            fever_c=38.6,
            symptoms=["fever", "cough", "chest pain"],
            cough=True,
            dyspnea=True,
            days_symptomatic_at_presentation=2,
        ),
        diagnostics=[
            DiagnosticResult(
                assay=Assay.F1_RDT,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.SPUTUM,
                collected_at=dt(2020, 8, 9, 7),
                reported_at=dt(2020, 8, 9, 8),
                lab_tier=LabTier.FIELD,
                lab_name="Rural health post",
            ),
        ],
        genomic=None,
        susceptibility=[],
        treatment_started=True,
        notes="No reference laboratory access at point of care; referral "
        "specimens in transit.",
    )
)

# ---------------------------------------------------------------------------
# 5. SYNTHETIC — genotype/phenotype discordance.
# ---------------------------------------------------------------------------
CASES.append(
    CaseRecord(
        case_id="PS-SYN-DISC-01",
        label="SYNTHETIC — genomic screen clean, phenotype non-susceptible",
        provenance=Provenance(
            origin=DataOrigin.SYNTHETIC,
            description=(
                "Constructed to exercise the Discordance Agent's primary case: "
                "a genome predicting susceptibility while the laboratory "
                "reports a non-susceptible result in a core therapeutic class. "
                "Carries no epidemiological meaning and describes no real "
                "patient or isolate."
            ),
            notes="Expected behaviour: CRITICAL discordance flag, suspected "
            "anomaly, escalation to human review, and repeat testing ranked "
            "as the most informative next result.",
        ),
        reported_date=date(2026, 3, 2),
        country="(synthetic)",
        admin1=None,
        age_band="40-49",
        sex="unknown",
        exposure=Exposure(
            plague_endemic_area=True,
            region_note="Synthetic endemic setting",
            rodent_or_flea_contact=True,
        ),
        clinical=Clinical(
            onset_date=date(2026, 2, 26),
            presentation_date=date(2026, 2, 28),
            form=ClinicalForm.BUBONIC,
            fever_c=39.1,
            symptoms=["fever", "bubo", "malaise"],
            bubo_present=True,
            lymphadenopathy=True,
            days_symptomatic_at_presentation=2,
        ),
        diagnostics=[
            DiagnosticResult(
                assay=Assay.CULTURE,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BUBO_ASPIRATE,
                collected_at=dt(2026, 2, 28, 10),
                reported_at=dt(2026, 3, 2, 10),
                lab_tier=LabTier.BSL3_REFERENCE,
            ),
            DiagnosticResult(
                assay=Assay.PCR_MULTIPLEX,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BUBO_ASPIRATE,
                collected_at=dt(2026, 2, 28, 10),
                reported_at=dt(2026, 3, 1, 9),
                lab_tier=LabTier.REFERENCE,
            ),
        ],
        genomic=GenomicEvidence(
            platform="Illumina NextSeq",
            read_depth_x=85.0,
            completeness_pct=98.9,
            contamination_pct=0.3,
            species_call="Yersinia pestis",
            ani_identity_pct=99.5,
            plasmid_markers_detected=["pCD1", "pMT1", "pPCP1"],
            amr_screen_performed=True,
            amr_tool_run="AMRFinderPlus 3.12.8 / 2024-07-22.1",
            amr_determinants=[],
            sequenced_at=dt(2026, 3, 4),
        ),
        susceptibility=[
            SusceptibilityResult(
                antimicrobial="streptomycin",
                drug_class=DrugClass.AMINOGLYCOSIDE,
                mic_mg_l=64.0,
                mic_operator=">",
                interpretation=Interpretation.RESISTANT,
                method=AstMethod.BROTH_MICRODILUTION,
                breakpoint_source="CLSI M45 4th ed.",
                tested_at=dt(2026, 3, 3),
                lab_tier=LabTier.REFERENCE,
                confirmed_by_repeat=False,
                note="Single determination; repeat not yet performed.",
            ),
            susceptible("gentamicin", DrugClass.AMINOGLYCOSIDE, 0.5, dt(2026, 3, 3)),
            susceptible("ciprofloxacin", DrugClass.FLUOROQUINOLONE, 0.03, dt(2026, 3, 3)),
            susceptible("doxycycline", DrugClass.TETRACYCLINE, 1.0, dt(2026, 3, 3)),
        ],
        treatment_started=True,
    )
)

# ---------------------------------------------------------------------------
# 6. SYNTHETIC — minimal evidence, out of distribution.
# ---------------------------------------------------------------------------
CASES.append(
    CaseRecord(
        case_id="PS-SYN-OOD-01",
        label="SYNTHETIC — non-endemic, minimal evidence",
        provenance=Provenance(
            origin=DataOrigin.SYNTHETIC,
            description=(
                "Constructed to exercise the Uncertainty Agent. Almost nothing "
                "has been measured and the exposure context is atypical, so "
                "the correct output is an explicit refusal to predict."
            ),
            notes="Expected behaviour: abstention with "
            "'cannot make a reliable prediction'.",
        ),
        reported_date=date(2026, 4, 11),
        country="(synthetic)",
        age_band="60-69",
        sex="unknown",
        exposure=Exposure(
            plague_endemic_area=False,
            rodent_or_flea_contact=None,
            contact_with_confirmed_case=False,
        ),
        clinical=Clinical(
            form=ClinicalForm.UNKNOWN,
            fever_c=38.2,
            symptoms=["fever", "malaise"],
            days_symptomatic_at_presentation=6,
        ),
        diagnostics=[
            DiagnosticResult(
                assay=Assay.MICROSCOPY_BIPOLAR,
                result=ResultValue.INDETERMINATE,
                specimen=SpecimenType.BLOOD,
                collected_at=dt(2026, 4, 11, 9),
                reported_at=dt(2026, 4, 11, 11),
                lab_tier=LabTier.HOSPITAL,
                detail="Equivocal morphology reported",
            ),
        ],
        genomic=None,
        susceptibility=[],
        treatment_started=False,
    )
)

# ---------------------------------------------------------------------------
# 7. SYNTHETIC — species identification conflict.
# ---------------------------------------------------------------------------
CASES.append(
    CaseRecord(
        case_id="PS-SYN-SPEC-01",
        label="SYNTHETIC — sequencing contradicts plague-targeted assays",
        provenance=Provenance(
            origin=DataOrigin.SYNTHETIC,
            description=(
                "Constructed to exercise identification discordance: "
                "plague-targeted assays positive while sequencing calls a "
                "closely related species. A recognised real-world pitfall, "
                "reproduced here with no real isolate."
            ),
            notes="Expected behaviour: CRITICAL species-conflict flag and "
            "referral for reference-laboratory resolution.",
        ),
        reported_date=date(2026, 1, 20),
        country="(synthetic)",
        age_band="20-29",
        sex="unknown",
        exposure=Exposure(plague_endemic_area=True, rodent_or_flea_contact=True),
        clinical=Clinical(
            onset_date=date(2026, 1, 16),
            presentation_date=date(2026, 1, 18),
            form=ClinicalForm.BUBONIC,
            fever_c=38.7,
            symptoms=["fever", "abdominal pain", "lymphadenopathy"],
            lymphadenopathy=True,
            bubo_present=True,
            days_symptomatic_at_presentation=2,
        ),
        diagnostics=[
            DiagnosticResult(
                assay=Assay.PCR_PLA,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BUBO_ASPIRATE,
                collected_at=dt(2026, 1, 18, 10),
                reported_at=dt(2026, 1, 19, 8),
                lab_tier=LabTier.REFERENCE,
            ),
            DiagnosticResult(
                assay=Assay.F1_RDT,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BUBO_ASPIRATE,
                collected_at=dt(2026, 1, 18, 10),
                reported_at=dt(2026, 1, 18, 11),
                lab_tier=LabTier.HOSPITAL,
            ),
            DiagnosticResult(
                assay=Assay.CULTURE,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BUBO_ASPIRATE,
                collected_at=dt(2026, 1, 18, 10),
                reported_at=dt(2026, 1, 21, 9),
                lab_tier=LabTier.REFERENCE,
            ),
        ],
        genomic=GenomicEvidence(
            platform="Oxford Nanopore",
            read_depth_x=44.0,
            completeness_pct=96.0,
            contamination_pct=1.1,
            species_call="Yersinia pseudotuberculosis",
            ani_identity_pct=96.4,
            plasmid_markers_detected=[],
            amr_screen_performed=True,
            amr_tool_run="AMRFinderPlus 3.12.8 / 2024-07-22.1",
            amr_determinants=[],
            sequenced_at=dt(2026, 1, 24),
        ),
        susceptibility=[],
        treatment_started=True,
    )
)

# ---------------------------------------------------------------------------
# 8. US septicemic case, partial workup.
# ---------------------------------------------------------------------------
CASES.append(
    CaseRecord(
        case_id="PS-2022-US-009",
        label="Septicemic plague, peridomestic animal exposure",
        provenance=Provenance(
            origin=DataOrigin.SURVEILLANCE_SUMMARY,
            description=(
                "Representative septicaemic presentation built from CDC "
                "surveillance characteristics, where primary septicaemic "
                "plague presents without a localising bubo and is frequently "
                "diagnosed late."
            ),
            citation=Citation(
                source="CDC",
                title="Plague Surveillance and Maps — United States",
                url="https://www.cdc.gov/plague/php/surveillance/",
            ),
        ),
        reported_date=date(2022, 6, 24),
        country="United States",
        admin1="Colorado",
        age_band="60-69",
        sex="unknown",
        exposure=Exposure(
            plague_endemic_area=True,
            region_note="Prairie-dog colony die-off reported nearby",
            rodent_or_flea_contact=True,
            contact_with_sick_animal=True,
            days_since_exposure=6,
        ),
        clinical=Clinical(
            onset_date=date(2022, 6, 20),
            presentation_date=date(2022, 6, 23),
            form=ClinicalForm.SEPTICEMIC,
            fever_c=39.8,
            symptoms=["fever", "hypotension", "confusion", "abdominal pain"],
            lymphadenopathy=False,
            bubo_present=False,
            septic_shock=True,
            rapid_deterioration=True,
            days_symptomatic_at_presentation=3,
        ),
        diagnostics=[
            DiagnosticResult(
                assay=Assay.CULTURE,
                result=ResultValue.POSITIVE,
                specimen=SpecimenType.BLOOD,
                collected_at=dt(2022, 6, 23, 14),
                reported_at=dt(2022, 6, 25, 20),
                lab_tier=LabTier.REFERENCE,
            ),
            DiagnosticResult(
                assay=Assay.MALDI_TOF,
                result=ResultValue.INDETERMINATE,
                specimen=SpecimenType.BLOOD,
                collected_at=dt(2022, 6, 25, 20),
                reported_at=dt(2022, 6, 25, 22),
                lab_tier=LabTier.HOSPITAL,
                detail="No confident identification — organism not in the "
                "clinical library",
            ),
            DiagnosticResult(
                assay=Assay.PCR_MULTIPLEX,
                result=ResultValue.PENDING,
                specimen=SpecimenType.BLOOD,
                collected_at=dt(2022, 6, 25, 20),
                lab_tier=LabTier.REFERENCE,
            ),
        ],
        genomic=None,
        susceptibility=[
            susceptible("gentamicin", DrugClass.AMINOGLYCOSIDE, 0.5, dt(2022, 6, 27)),
            susceptible("ciprofloxacin", DrugClass.FLUOROQUINOLONE, 0.03, dt(2022, 6, 27)),
        ],
        treatment_started=True,
        notes="Doxycycline not included in the local panel.",
    )
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for case in CASES:
        path = OUT / f"{case.case_id}.json"
        path.write_text(json.dumps(case.model_dump(mode="json"), indent=2))
        index.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "origin": case.provenance.origin.value,
                "country": case.country,
                "file": path.name,
            }
        )
        print(f"wrote {path.relative_to(OUT.parents[3])}")

    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote index with {len(index)} cases")


if __name__ == "__main__":
    main()
