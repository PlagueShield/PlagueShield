"""Published diagnostic performance for plague assays.

Each entry carries the sensitivity/specificity used by the Diagnostic Agent's
likelihood-ratio model, plus the citation it came from. Numbers are rounded
central estimates from the cited evaluations; they are deliberately
conservative where studies disagree.

These are *diagnostic* performance figures -- the kind published in evaluation
studies and reproduced in WHO/CDC laboratory guidance. Nothing here is
operationally sensitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Assay, Citation, LabTier

# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------

CIT_CHANTEAU_RDT = Citation(
    source="Chanteau S, et al. (The Lancet)",
    title="Development and testing of a rapid diagnostic test for bubonic and "
    "pneumonic plague",
    identifier="PMID:12531581",
    year=2003,
    note="F1 antigen immunochromatographic dipstick field evaluation, Madagascar.",
)

CIT_WHO_LAB_MANUAL = Citation(
    source="WHO",
    title="Plague Manual: Epidemiology, Distribution, Surveillance and Control",
    identifier="WHO/CDS/CSR/EDC/99.2",
    url="https://www.who.int/publications/i/item/WHO-CDS-CSR-EDC-99.2",
    year=1999,
    note="Reference laboratory confirmation criteria and assay roles.",
)

CIT_WHO_IG_2021 = Citation(
    source="WHO",
    title="Interim guidance on the clinical management of plague",
    url="https://www.who.int/publications/i/item/9789240034884",
    year=2021,
)

CIT_CDC_PLAGUE_DX = Citation(
    source="CDC",
    title="Plague: Diagnosis and Laboratory Testing / Resources for Clinicians",
    url="https://www.cdc.gov/plague/hcp/diagnosis-testing/",
    note="Confirmatory testing pathway and LRN referral.",
)

CIT_CDC_CASE_DEF = Citation(
    source="CDC / CSTE",
    title="Plague (Yersinia pestis) 2020 Case Definition",
    url="https://ndc.services.cdc.gov/case-definitions/plague-2020/",
    year=2020,
    note="Suspect / probable / confirmed classification criteria.",
)

CIT_RILEY_PCR = Citation(
    source="Riehm JM, et al. (Clin Microbiol Infect)",
    title="Detection of Yersinia pestis using real-time PCR in patients with "
    "suspected bubonic plague",
    identifier="PMID:21951463",
    year=2011,
    note="Real-time PCR targeting pla and caf1 in clinical specimens.",
)

CIT_SPLETTSTOESSER = Citation(
    source="Splettstoesser WD, et al. (J Med Microbiol)",
    title="Evaluation of a standardised direct immunofluorescence assay for "
    "Yersinia pestis",
    identifier="PMID:14663086",
    year=2004,
)

CIT_LRN = Citation(
    source="CDC",
    title="Laboratory Response Network (LRN) — sentinel laboratory guidance for "
    "Yersinia pestis",
    url="https://www.cdc.gov/laboratory-response-network/php/about/",
    note="Rule-out/refer protocol for sentinel clinical laboratories.",
)

ALL_ASSAY_CITATIONS = [
    CIT_CHANTEAU_RDT,
    CIT_WHO_LAB_MANUAL,
    CIT_WHO_IG_2021,
    CIT_CDC_PLAGUE_DX,
    CIT_CDC_CASE_DEF,
    CIT_RILEY_PCR,
    CIT_SPLETTSTOESSER,
    CIT_LRN,
]


# --------------------------------------------------------------------------
# Performance table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AssayProfile:
    assay: Assay
    display_name: str
    sensitivity: float
    specificity: float
    turnaround_hours: float
    requires_isolate: bool
    requires_bsl3: bool
    confirmatory: bool
    citation: Citation
    note: str = ""
    # Multiplier applied to the log-LR when the producing lab is below
    # reference tier. Field-read RDTs are the main case.
    field_tier_discount: float = 1.0

    @property
    def lr_positive(self) -> float:
        """Likelihood ratio for a positive result."""
        fp = max(1.0 - self.specificity, 1e-3)
        return self.sensitivity / fp

    @property
    def lr_negative(self) -> float:
        """Likelihood ratio for a negative result."""
        fn = max(1.0 - self.sensitivity, 1e-3)
        return fn / max(self.specificity, 1e-3)


ASSAY_PROFILES: dict[Assay, AssayProfile] = {
    Assay.F1_RDT: AssayProfile(
        assay=Assay.F1_RDT,
        display_name="F1 antigen rapid dipstick",
        sensitivity=0.95,
        specificity=0.87,
        turnaround_hours=0.25,
        requires_isolate=False,
        requires_bsl3=False,
        confirmatory=False,
        citation=CIT_CHANTEAU_RDT,
        note="Point-of-care screening. High sensitivity, moderate specificity; "
        "a positive RDT alone does not confirm a case.",
        field_tier_discount=0.85,
    ),
    Assay.PCR_PLA: AssayProfile(
        assay=Assay.PCR_PLA,
        display_name="Real-time PCR (pla target)",
        sensitivity=0.86,
        specificity=0.99,
        turnaround_hours=4.0,
        requires_isolate=False,
        requires_bsl3=False,
        confirmatory=False,
        citation=CIT_RILEY_PCR,
        note="pla is plasmid-borne; single-target positives are presumptive. "
        "Related species can rarely carry homologues.",
    ),
    Assay.PCR_CAF1: AssayProfile(
        assay=Assay.PCR_CAF1,
        display_name="Real-time PCR (caf1/F1 target)",
        sensitivity=0.84,
        specificity=0.99,
        turnaround_hours=4.0,
        requires_isolate=False,
        requires_bsl3=False,
        confirmatory=False,
        citation=CIT_RILEY_PCR,
    ),
    Assay.PCR_MULTIPLEX: AssayProfile(
        assay=Assay.PCR_MULTIPLEX,
        display_name="Multiplex PCR (≥2 independent targets)",
        sensitivity=0.90,
        specificity=0.998,
        turnaround_hours=5.0,
        requires_isolate=False,
        requires_bsl3=False,
        confirmatory=True,
        citation=CIT_WHO_LAB_MANUAL,
        note="Two or more independent targets is a recognised confirmatory "
        "criterion in reference laboratories.",
    ),
    Assay.CULTURE: AssayProfile(
        assay=Assay.CULTURE,
        display_name="Culture isolation",
        sensitivity=0.70,
        specificity=0.999,
        turnaround_hours=72.0,
        requires_isolate=False,
        requires_bsl3=True,
        confirmatory=True,
        citation=CIT_WHO_LAB_MANUAL,
        note="Gold standard for confirmation and the prerequisite for "
        "phenotypic susceptibility testing. Sensitivity falls sharply once "
        "antibiotics have been started.",
    ),
    Assay.MICROSCOPY_BIPOLAR: AssayProfile(
        assay=Assay.MICROSCOPY_BIPOLAR,
        display_name="Microscopy, bipolar-staining coccobacilli",
        sensitivity=0.55,
        specificity=0.80,
        turnaround_hours=1.0,
        requires_isolate=False,
        requires_bsl3=False,
        confirmatory=False,
        citation=CIT_WHO_LAB_MANUAL,
        note="Suggestive only; morphology overlaps with other organisms.",
    ),
    Assay.IMMUNOFLUORESCENCE: AssayProfile(
        assay=Assay.IMMUNOFLUORESCENCE,
        display_name="Direct fluorescent antibody (F1)",
        sensitivity=0.75,
        specificity=0.98,
        turnaround_hours=2.0,
        requires_isolate=False,
        requires_bsl3=False,
        confirmatory=False,
        citation=CIT_SPLETTSTOESSER,
    ),
    Assay.MALDI_TOF: AssayProfile(
        assay=Assay.MALDI_TOF,
        display_name="MALDI-TOF identification",
        sensitivity=0.92,
        specificity=0.97,
        turnaround_hours=2.0,
        requires_isolate=True,
        requires_bsl3=True,
        confirmatory=False,
        citation=CIT_CDC_PLAGUE_DX,
        note="Requires an isolate and a database containing Y. pestis; many "
        "clinical libraries exclude select agents, causing misidentification.",
    ),
    Assay.PHAGE_LYSIS: AssayProfile(
        assay=Assay.PHAGE_LYSIS,
        display_name="Bacteriophage lysis",
        sensitivity=0.95,
        specificity=0.99,
        turnaround_hours=24.0,
        requires_isolate=True,
        requires_bsl3=True,
        confirmatory=True,
        citation=CIT_WHO_LAB_MANUAL,
        note="Classical confirmatory test performed at reference level.",
    ),
    Assay.SEROLOGY_F1_PAIRED: AssayProfile(
        assay=Assay.SEROLOGY_F1_PAIRED,
        display_name="Paired serology, ≥4-fold F1 titre rise",
        sensitivity=0.90,
        specificity=0.99,
        turnaround_hours=336.0,
        requires_isolate=False,
        requires_bsl3=False,
        confirmatory=True,
        citation=CIT_CDC_CASE_DEF,
        note="Retrospective confirmation; requires convalescent serum ~2-3 "
        "weeks after onset. No use for acute management.",
    ),
    Assay.SEROLOGY_F1_SINGLE: AssayProfile(
        assay=Assay.SEROLOGY_F1_SINGLE,
        display_name="Single-serum F1 antibody titre",
        sensitivity=0.70,
        specificity=0.92,
        turnaround_hours=48.0,
        requires_isolate=False,
        requires_bsl3=False,
        confirmatory=False,
        citation=CIT_CDC_CASE_DEF,
        note="Early-acute sera are frequently negative; cannot exclude plague.",
    ),
    Assay.WGS_SPECIES_ID: AssayProfile(
        assay=Assay.WGS_SPECIES_ID,
        display_name="Whole-genome sequencing species identification",
        sensitivity=0.98,
        specificity=0.999,
        turnaround_hours=48.0,
        requires_isolate=False,
        requires_bsl3=False,
        confirmatory=True,
        citation=CIT_CDC_PLAGUE_DX,
        note="Also distinguishes Y. pestis from Y. pseudotuberculosis, the "
        "main identification pitfall.",
    ),
}


def profile_for(assay: Assay) -> AssayProfile | None:
    return ASSAY_PROFILES.get(assay)


def lab_tier_weight(tier: LabTier) -> float:
    """How much to trust a result given the producing laboratory's tier."""
    return {
        LabTier.FIELD: 0.70,
        LabTier.HOSPITAL: 0.85,
        LabTier.REFERENCE: 1.0,
        LabTier.BSL3_REFERENCE: 1.0,
    }.get(tier, 0.85)
