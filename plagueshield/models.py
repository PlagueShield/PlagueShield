"""Domain models for PlagueShield.

Everything the agents consume or emit is defined here so that a case record is
auditable end to end: every assertion carries provenance, every verdict carries
its rationale and the citations behind it.

Design note on patient data: records are deliberately de-identified by
construction. There is no name, no exact date of birth, no address, no record
locator. Age is banded, geography is at admin-1 (province/state) granularity.
Public plague case data does not exist at individual level for privacy reasons,
so bundled cases are either (a) vignettes reconstructed from published outbreak
reports and surveillance summaries, or (b) real isolate metadata from public
sequence archives. The `provenance` field on every record says which.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Controlled vocabularies
# --------------------------------------------------------------------------


class DataOrigin(str, Enum):
    """Where a case record came from. Shown in the UI; never inferred."""

    PUBLISHED_REPORT = "published_report"
    SURVEILLANCE_SUMMARY = "surveillance_summary"
    PUBLIC_SEQUENCE_ARCHIVE = "public_sequence_archive"
    SYNTHETIC = "synthetic"


class ClinicalForm(str, Enum):
    BUBONIC = "bubonic"
    PNEUMONIC = "pneumonic"
    SEPTICEMIC = "septicemic"
    PHARYNGEAL = "pharyngeal"
    MENINGEAL = "meningeal"
    UNKNOWN = "unknown"


class SpecimenType(str, Enum):
    BLOOD = "blood"
    SPUTUM = "sputum"
    BUBO_ASPIRATE = "bubo_aspirate"
    BRONCHOALVEOLAR_LAVAGE = "bronchoalveolar_lavage"
    THROAT_SWAB = "throat_swab"
    CSF = "csf"
    TISSUE = "tissue"
    POSTMORTEM = "postmortem"


class Assay(str, Enum):
    """Diagnostic assays recognised by the Diagnostic Agent.

    Performance characteristics live in knowledge/assay_performance.py with
    citations; this enum is only the vocabulary.
    """

    MICROSCOPY_BIPOLAR = "microscopy_bipolar"
    CULTURE = "culture"
    F1_RDT = "f1_rdt"
    PCR_PLA = "pcr_pla"
    PCR_CAF1 = "pcr_caf1"
    PCR_MULTIPLEX = "pcr_multiplex"
    MALDI_TOF = "maldi_tof"
    IMMUNOFLUORESCENCE = "immunofluorescence"
    SEROLOGY_F1_PAIRED = "serology_f1_paired"
    SEROLOGY_F1_SINGLE = "serology_f1_single"
    PHAGE_LYSIS = "phage_lysis"
    WGS_SPECIES_ID = "wgs_species_id"


class ResultValue(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    INDETERMINATE = "indeterminate"
    PENDING = "pending"
    NOT_DONE = "not_done"


class LabTier(str, Enum):
    """Capability of the lab that produced a result. Affects weighting."""

    FIELD = "field"
    HOSPITAL = "hospital"
    REFERENCE = "reference"
    BSL3_REFERENCE = "bsl3_reference"


class DrugClass(str, Enum):
    """Drug classes relevant to plague therapy and prophylaxis.

    PlagueShield reasons at this level only. It does not model or catalogue
    resistance mechanisms -- see knowledge/regimens.py for the rationale.
    """

    AMINOGLYCOSIDE = "aminoglycoside"
    TETRACYCLINE = "tetracycline"
    FLUOROQUINOLONE = "fluoroquinolone"
    SULFONAMIDE = "sulfonamide"
    PHENICOL = "phenicol"
    BETA_LACTAM = "beta_lactam"
    MACROLIDE = "macrolide"
    OTHER = "other"


class Interpretation(str, Enum):
    SUSCEPTIBLE = "S"
    INTERMEDIATE = "I"
    RESISTANT = "R"
    NONSUSCEPTIBLE = "NS"
    NO_BREAKPOINT = "no_breakpoint"


class AstMethod(str, Enum):
    BROTH_MICRODILUTION = "broth_microdilution"
    AGAR_DILUTION = "agar_dilution"
    GRADIENT_STRIP = "gradient_strip"
    DISK_DIFFUSION = "disk_diffusion"


class Likelihood(str, Enum):
    VERY_LOW = "Very low"
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very high"
    INDETERMINATE = "Indeterminate"


class Confidence(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


class SusceptibilityOutlook(str, Enum):
    PRESERVED = "Preserved"
    PROBABLY_PRESERVED = "Probably preserved"
    UNCERTAIN = "Uncertain"
    POSSIBLY_COMPROMISED = "Possibly compromised"
    COMPROMISED = "Compromised"
    INSUFFICIENT_EVIDENCE = "Insufficient evidence"


class AnomalyStatus(str, Enum):
    NONE_DETECTED = "None detected"
    INSUFFICIENT_EVIDENCE = "Insufficient evidence"
    SUSPECTED = "Suspected"
    CONFIRMED = "Confirmed"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class CaseClassification(str, Enum):
    """WHO/CDC-style case classification tiers."""

    NOT_A_CASE = "not_a_case"
    SUSPECTED = "suspected"
    PRESUMPTIVE = "presumptive"
    CONFIRMED = "confirmed"


# --------------------------------------------------------------------------
# Evidence primitives
# --------------------------------------------------------------------------


class Citation(BaseModel):
    """A pointer to guidance or literature backing an assertion."""

    source: str = Field(description="Issuing body or journal, e.g. 'CDC', 'WHO'")
    title: str
    identifier: str | None = Field(
        default=None, description="DOI, PMID, or document number"
    )
    url: str | None = None
    year: int | None = None
    retrieved_at: datetime | None = None
    note: str | None = None

    def short(self) -> str:
        bits = [self.source]
        if self.year:
            bits.append(str(self.year))
        return f"{' '.join(bits)}: {self.title}"


class Provenance(BaseModel):
    """Where a case record came from, and how much to trust it."""

    origin: DataOrigin
    description: str
    citation: Citation | None = None
    deidentified: bool = True
    notes: str | None = None


class Exposure(BaseModel):
    """Epidemiological risk factors. All optional -- absence is not negation."""

    plague_endemic_area: bool | None = None
    region_note: str | None = None
    rodent_or_flea_contact: bool | None = None
    contact_with_confirmed_case: bool | None = None
    contact_with_sick_animal: bool | None = None
    hunting_or_skinning: bool | None = None
    laboratory_exposure: bool | None = None
    days_since_exposure: int | None = None
    cluster_size: int | None = Field(
        default=None, description="Epi-linked cases in the same cluster, if any"
    )


class Clinical(BaseModel):
    onset_date: date | None = None
    presentation_date: date | None = None
    form: ClinicalForm = ClinicalForm.UNKNOWN
    fever_c: float | None = None
    symptoms: list[str] = Field(default_factory=list)
    lymphadenopathy: bool | None = None
    bubo_present: bool | None = None
    cough: bool | None = None
    hemoptysis: bool | None = None
    dyspnea: bool | None = None
    chest_imaging: str | None = None
    septic_shock: bool | None = None
    rapid_deterioration: bool | None = None
    days_symptomatic_at_presentation: int | None = None

    @field_validator("fever_c")
    @classmethod
    def _plausible_temp(cls, v: float | None) -> float | None:
        if v is not None and not (30.0 <= v <= 45.0):
            raise ValueError(f"implausible body temperature: {v}")
        return v


class DiagnosticResult(BaseModel):
    assay: Assay
    result: ResultValue
    specimen: SpecimenType | None = None
    collected_at: datetime | None = None
    reported_at: datetime | None = None
    lab_tier: LabTier = LabTier.HOSPITAL
    lab_name: str | None = None
    detail: str | None = None
    ct_value: float | None = Field(
        default=None, description="PCR cycle threshold, if reported"
    )
    titer: str | None = None
    fold_rise: float | None = Field(
        default=None, description="Paired serology fold rise"
    )

    @property
    def turnaround_hours(self) -> float | None:
        if self.collected_at and self.reported_at:
            return (self.reported_at - self.collected_at).total_seconds() / 3600.0
        return None


class AmrDeterminant(BaseModel):
    """An AMR determinant **as called by an external tool**.

    PlagueShield does not contain a catalogue of resistance mechanisms and does
    not attempt to predict resistance from raw sequence. It consumes the output
    of standard, separately maintained tools (NCBI AMRFinderPlus, ResFinder,
    CARD/RGI), each of which reports a determinant together with the drug class
    it affects. Reasoning downstream happens at the drug-class level.
    """

    gene_symbol: str = Field(description="Symbol exactly as emitted by the tool")
    drug_class: DrugClass
    tool: str = Field(description="e.g. 'AMRFinderPlus 3.12.8'")
    identity_pct: float | None = None
    coverage_pct: float | None = None
    contig: str | None = None
    element_type: str | None = Field(
        default=None, description="Tool's own element type label, passed through"
    )

    @field_validator("identity_pct", "coverage_pct")
    @classmethod
    def _pct(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 100.0):
            raise ValueError("percentage must be 0-100")
        return v


class GenomicEvidence(BaseModel):
    """Sequencing-derived evidence for identification and AMR screening."""

    platform: str | None = None
    read_depth_x: float | None = None
    assembly_n50: int | None = None
    completeness_pct: float | None = None
    contamination_pct: float | None = None

    species_call: str | None = None
    ani_identity_pct: float | None = Field(
        default=None, description="Average nucleotide identity to Y. pestis reference"
    )
    mlst_or_lineage: str | None = None

    # Virulence-plasmid detection is used here strictly for species/subspecies
    # identification -- these are the canonical markers that distinguish
    # Y. pestis from Y. pseudotuberculosis in a diagnostic workflow.
    plasmid_markers_detected: list[str] = Field(default_factory=list)

    amr_determinants: list[AmrDeterminant] = Field(default_factory=list)
    amr_tool_run: str | None = Field(
        default=None, description="Tool+DB version string for the AMR screen"
    )
    amr_screen_performed: bool = False

    sequenced_at: datetime | None = None
    accession: str | None = None

    @property
    def quality_sufficient(self) -> bool:
        """Whether the assembly is good enough to trust a negative AMR screen."""
        if self.read_depth_x is not None and self.read_depth_x < 20:
            return False
        if self.completeness_pct is not None and self.completeness_pct < 90:
            return False
        if self.contamination_pct is not None and self.contamination_pct > 5:
            return False
        return True

    def classes_with_determinants(self) -> set[DrugClass]:
        return {d.drug_class for d in self.amr_determinants}


class SusceptibilityResult(BaseModel):
    """A phenotypic antimicrobial susceptibility test result."""

    antimicrobial: str
    drug_class: DrugClass
    mic_mg_l: float | None = None
    mic_operator: Literal["<=", "<", "=", ">", ">="] | None = None
    interpretation: Interpretation
    method: AstMethod = AstMethod.BROTH_MICRODILUTION
    breakpoint_source: str | None = Field(
        default=None, description="e.g. 'CLSI M45 4th ed.'"
    )
    tested_at: datetime | None = None
    lab_tier: LabTier = LabTier.REFERENCE
    lab_name: str | None = None
    confirmed_by_repeat: bool = False
    note: str | None = None

    @property
    def is_nonsusceptible(self) -> bool:
        return self.interpretation in (
            Interpretation.RESISTANT,
            Interpretation.NONSUSCEPTIBLE,
            Interpretation.INTERMEDIATE,
        )


class CaseRecord(BaseModel):
    """A de-identified suspected-plague case, as fed to the agent pipeline."""

    case_id: str
    label: str = Field(description="Short human-readable handle for the UI")
    provenance: Provenance

    reported_date: date | None = None
    country: str | None = None
    admin1: str | None = Field(default=None, description="Province/state, no finer")
    age_band: str | None = Field(default=None, description="e.g. '30-39'")
    sex: Literal["male", "female", "other", "unknown"] | None = None

    exposure: Exposure = Field(default_factory=Exposure)
    clinical: Clinical = Field(default_factory=Clinical)
    diagnostics: list[DiagnosticResult] = Field(default_factory=list)
    genomic: GenomicEvidence | None = None
    susceptibility: list[SusceptibilityResult] = Field(default_factory=list)

    prior_antibiotics: list[str] = Field(default_factory=list)
    treatment_started: bool | None = None
    notes: str | None = None

    def results_for(self, assay: Assay) -> list[DiagnosticResult]:
        return [d for d in self.diagnostics if d.assay == assay]

    def has_positive(self, assay: Assay) -> bool:
        return any(
            d.result == ResultValue.POSITIVE for d in self.results_for(assay)
        )

    def tested_assays(self) -> set[Assay]:
        return {
            d.assay
            for d in self.diagnostics
            if d.result
            in (ResultValue.POSITIVE, ResultValue.NEGATIVE, ResultValue.INDETERMINATE)
        }

    def pending_assays(self) -> set[Assay]:
        return {d.assay for d in self.diagnostics if d.result == ResultValue.PENDING}


# --------------------------------------------------------------------------
# Agent outputs
# --------------------------------------------------------------------------


class Flag(BaseModel):
    code: str
    severity: Severity
    message: str
    detail: str | None = None


class Verdict(BaseModel):
    """Uniform envelope every agent returns."""

    agent: str
    headline: str
    score: float | None = Field(
        default=None, description="Primary numeric output, agent-specific"
    )
    confidence: Confidence = Confidence.LOW
    abstained: bool = False
    abstain_reason: str | None = None
    rationale: list[str] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    produced_at: datetime = Field(default_factory=_utcnow)


class CandidateTest(BaseModel):
    """A test the Next-Test Agent can recommend, with its ranking inputs."""

    key: str
    name: str
    target_question: Literal["identification", "resistance", "both"]
    sensitivity: float
    specificity: float
    turnaround_hours: float
    requires_isolate: bool = False
    requires_bsl3: bool = False
    clinical_actionability: float = Field(
        ge=0.0, le=1.0, description="How much the result changes management"
    )
    availability_note: str | None = None

    # populated by the ranker
    expected_information_gain_bits: float | None = None
    utility: float | None = None
    rank: int | None = None
    already_done: bool = False
    rationale: str | None = None


class CaseAssessment(BaseModel):
    """The full, auditable output of one pipeline run over one case."""

    case_id: str
    case_label: str
    assessed_at: datetime = Field(default_factory=_utcnow)
    pipeline_version: str = "0.1.0"

    # The headline answers, matching the operator-facing summary shape.
    plague_likelihood: Likelihood
    plague_probability: float | None = None
    case_classification: CaseClassification
    standard_treatment_susceptibility: SusceptibilityOutlook
    resistance_anomaly: AnomalyStatus
    confidence: Confidence
    most_valuable_next_result: str | None = None
    escalation: list[str] = Field(default_factory=list)

    human_review_required: bool = False
    review_triggers: list[str] = Field(default_factory=list)

    verdicts: dict[str, Verdict] = Field(default_factory=dict)
    ranked_tests: list[CandidateTest] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    report_markdown: str = ""
    report_ascii: str = ""

    disclaimer: str = (
        "Decision support only. Does not prescribe or withhold treatment. "
        "All outputs require review by a qualified infectious-disease "
        "clinician or clinical microbiologist before any clinical action."
    )
