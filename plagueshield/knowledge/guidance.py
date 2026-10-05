"""Bundled CDC/WHO guidance and literature corpus for the Evidence Agent.

The Evidence Agent retrieves from this corpus offline, and optionally refreshes
live URLs via `plagueshield.data.public_sources`. Bundling means the system
still cites correctly with no network, which matters for field deployment.

Each entry is tagged so the agent can retrieve by topic rather than by keyword
soup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Citation

from .assay_performance import (
    CIT_CDC_CASE_DEF,
    CIT_CDC_PLAGUE_DX,
    CIT_CHANTEAU_RDT,
    CIT_LRN,
    CIT_RILEY_PCR,
    CIT_WHO_IG_2021,
    CIT_WHO_LAB_MANUAL,
)
from .regimens import (
    CIT_CDC_TREATMENT,
    CIT_CLSI_M45,
    CIT_NELSON_MMWR,
    CIT_WHO_CLINICAL_2021,
)


@dataclass(frozen=True)
class GuidanceEntry:
    key: str
    citation: Citation
    topics: frozenset[str]
    summary: str
    live_url: str | None = None


CIT_WHO_DON = Citation(
    source="WHO",
    title="Disease Outbreak News — Plague",
    url="https://www.who.int/emergencies/disease-outbreak-news",
    note="Primary source for current international plague event reporting.",
)

CIT_CDC_SURVEILLANCE = Citation(
    source="CDC",
    title="Plague Surveillance and Maps — United States",
    url="https://www.cdc.gov/plague/php/surveillance/",
    note="US human plague case counts and geographic distribution.",
)

CIT_IHR = Citation(
    source="WHO",
    title="International Health Regulations (2005), Annex 2 decision instrument",
    url="https://www.who.int/publications/i/item/9789241580496",
    year=2005,
    note="Pneumonic plague is notifiable; triggers IHR assessment for "
    "potential public health emergency of international concern.",
)

CIT_NCBI_PATHOGEN = Citation(
    source="NCBI",
    title="NCBI Pathogen Detection — Yersinia pestis isolates browser",
    url="https://www.ncbi.nlm.nih.gov/pathogens/organisms/",
    note="Public isolate metadata and AMR genotype calls produced by "
    "AMRFinderPlus.",
)

CIT_AMRFINDER = Citation(
    source="Feldgarden M, et al. (Sci Rep)",
    title="AMRFinderPlus and the Reference Gene Catalog facilitate examination "
    "of the genomic links among antimicrobial resistance, stress response, "
    "and virulence",
    identifier="PMID:34135355",
    year=2021,
    note="The external determinant-calling tool whose class-level output "
    "PlagueShield consumes.",
)

CIT_GUIYOULE_1997 = Citation(
    source="Guiyoule A, et al. (Emerg Infect Dis / N Engl J Med)",
    title="Recognition of multidrug-resistant Yersinia pestis in Madagascar",
    year=1997,
    note="Documents that transferable multiresistance has been observed in "
    "Y. pestis. Cited here only to establish that the phenomenon exists and "
    "is rare; mechanism detail is intentionally not reproduced.",
)

CIT_WHO_AST = Citation(
    source="WHO",
    title="Global Antimicrobial Resistance and Use Surveillance System (GLASS)",
    url="https://www.who.int/initiatives/glass",
    note="Framework for reporting confirmed resistance findings onward.",
)


CORPUS: tuple[GuidanceEntry, ...] = (
    GuidanceEntry(
        key="who_clinical_2021",
        citation=CIT_WHO_CLINICAL_2021,
        topics=frozenset({"treatment", "clinical", "prophylaxis", "guidance"}),
        summary="WHO interim guidance on clinical management of bubonic, "
        "pneumonic and septicaemic plague, including recognised therapeutic "
        "classes and supportive care.",
        live_url=CIT_WHO_CLINICAL_2021.url,
    ),
    GuidanceEntry(
        key="cdc_treatment",
        citation=CIT_CDC_TREATMENT,
        topics=frozenset({"treatment", "prophylaxis", "clinical", "guidance"}),
        summary="CDC clinical guidance covering treatment and post-exposure "
        "prophylaxis for plague.",
        live_url=CIT_CDC_TREATMENT.url,
    ),
    GuidanceEntry(
        key="mmwr_2021",
        citation=CIT_NELSON_MMWR,
        topics=frozenset({"treatment", "prophylaxis", "guidance", "preparedness"}),
        summary="MMWR recommendations for antimicrobial treatment and "
        "prophylaxis of plague, for naturally acquired infection and "
        "deliberate-release response.",
        live_url=CIT_NELSON_MMWR.url,
    ),
    GuidanceEntry(
        key="cdc_case_definition",
        citation=CIT_CDC_CASE_DEF,
        topics=frozenset({"case_definition", "classification", "surveillance"}),
        summary="CSTE/CDC surveillance case definition: suspect, probable and "
        "confirmed criteria for human plague.",
        live_url=CIT_CDC_CASE_DEF.url,
    ),
    GuidanceEntry(
        key="who_lab_manual",
        citation=CIT_WHO_LAB_MANUAL,
        topics=frozenset({"laboratory", "diagnosis", "confirmation", "surveillance"}),
        summary="WHO plague manual: laboratory confirmation criteria, assay "
        "roles, specimen handling and surveillance structure.",
        live_url=CIT_WHO_LAB_MANUAL.url,
    ),
    GuidanceEntry(
        key="cdc_diagnosis",
        citation=CIT_CDC_PLAGUE_DX,
        topics=frozenset({"laboratory", "diagnosis", "testing"}),
        summary="CDC diagnosis and laboratory testing pathway for suspected "
        "plague, including specimen selection by clinical form.",
        live_url=CIT_CDC_PLAGUE_DX.url,
    ),
    GuidanceEntry(
        key="lrn_referral",
        citation=CIT_LRN,
        topics=frozenset({"laboratory", "referral", "biosafety", "select_agent"}),
        summary="Laboratory Response Network sentinel-laboratory rule-out and "
        "referral protocol; sentinel labs refer suspected Y. pestis rather "
        "than fully characterising it.",
        live_url=CIT_LRN.url,
    ),
    GuidanceEntry(
        key="f1_rdt",
        citation=CIT_CHANTEAU_RDT,
        topics=frozenset({"laboratory", "diagnosis", "rapid_test"}),
        summary="Field evaluation of the F1 antigen dipstick for bubonic and "
        "pneumonic plague, the basis for its screening role.",
    ),
    GuidanceEntry(
        key="pcr_performance",
        citation=CIT_RILEY_PCR,
        topics=frozenset({"laboratory", "diagnosis", "pcr"}),
        summary="Real-time PCR performance in clinical specimens from "
        "suspected plague patients.",
    ),
    GuidanceEntry(
        key="clsi_m45",
        citation=CIT_CLSI_M45,
        topics=frozenset({"susceptibility", "breakpoints", "laboratory"}),
        summary="CLSI M45 interpretive breakpoints, the standard applied when "
        "reporting Y. pestis susceptibility results.",
    ),
    GuidanceEntry(
        key="amr_rarity",
        citation=CIT_GUIYOULE_1997,
        topics=frozenset({"resistance", "epidemiology", "prior"}),
        summary="Establishes that transferable multiresistance has been "
        "documented in Y. pestis on rare occasions, which is the basis for a "
        "low but non-zero prior on resistance.",
    ),
    GuidanceEntry(
        key="amrfinderplus",
        citation=CIT_AMRFINDER,
        topics=frozenset({"resistance", "genomics", "tooling"}),
        summary="AMRFinderPlus and the NCBI Reference Gene Catalog — the "
        "external determinant-calling pipeline PlagueShield consumes.",
        live_url=CIT_AMRFINDER.url,
    ),
    GuidanceEntry(
        key="ncbi_pathogen",
        citation=CIT_NCBI_PATHOGEN,
        topics=frozenset({"resistance", "genomics", "surveillance", "public_data"}),
        summary="Public Y. pestis isolate metadata with standardised AMR "
        "genotype calls; the empirical basis for the resistance prior.",
        live_url=CIT_NCBI_PATHOGEN.url,
    ),
    GuidanceEntry(
        key="who_don",
        citation=CIT_WHO_DON,
        topics=frozenset({"surveillance", "outbreak", "public_data", "epidemiology"}),
        summary="WHO Disease Outbreak News, the authoritative feed for "
        "ongoing international plague events.",
        live_url=CIT_WHO_DON.url,
    ),
    GuidanceEntry(
        key="cdc_surveillance",
        citation=CIT_CDC_SURVEILLANCE,
        topics=frozenset({"surveillance", "epidemiology", "public_data"}),
        summary="US human plague surveillance counts and endemic geography, "
        "used to inform the pre-test probability in US cases.",
        live_url=CIT_CDC_SURVEILLANCE.url,
    ),
    GuidanceEntry(
        key="ihr_notification",
        citation=CIT_IHR,
        topics=frozenset({"notification", "escalation", "public_health"}),
        summary="IHR (2005) notification obligations; pneumonic plague "
        "requires immediate public health assessment and notification.",
        live_url=CIT_IHR.url,
    ),
    GuidanceEntry(
        key="glass",
        citation=CIT_WHO_AST,
        topics=frozenset({"resistance", "reporting", "escalation"}),
        summary="WHO GLASS, the route for reporting confirmed resistance "
        "findings into global AMR surveillance.",
        live_url=CIT_WHO_AST.url,
    ),
)


def retrieve(topics: set[str], limit: int = 6) -> list[GuidanceEntry]:
    """Return corpus entries ranked by topic overlap."""
    scored: list[tuple[int, GuidanceEntry]] = []
    for entry in CORPUS:
        overlap = len(entry.topics & topics)
        if overlap:
            scored.append((overlap, entry))
    scored.sort(key=lambda pair: (-pair[0], pair[1].key))
    return [entry for _, entry in scored[:limit]]


def by_key(key: str) -> GuidanceEntry | None:
    for entry in CORPUS:
        if entry.key == key:
            return entry
    return None
