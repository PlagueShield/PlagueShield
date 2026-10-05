"""Drug-class → treatment-role mapping for plague.

SCOPE BOUNDARY (deliberate, do not "improve" this away)
-------------------------------------------------------
This module maps *drug classes* to their role in recognised plague treatment
and prophylaxis guidance. It does **not** contain, and must not be extended to
contain:

  * a catalogue of resistance genes, alleles, mutations or plasmids;
  * any mapping from a genetic determinant to the degree of resistance it
    confers;
  * any ranking of which determinants would defeat which plague regimens.

That catalogue is exactly the artifact that would function as a design sheet
for an untreatable strain, and PlagueShield does not need it. Determinant-level
calling is delegated to externally maintained, access-controlled tools
(NCBI AMRFinderPlus, ResFinder, CARD/RGI). Those tools emit a determinant plus
the drug class it affects; PlagueShield consumes the class and reasons from
there. The clinical question an ID physician actually asks -- "is a class in my
regimen under threat, and does the phenotype agree?" -- is fully answerable at
class level.

Treatment roles below follow WHO interim clinical-management guidance (2021)
and CDC clinical guidance. They describe which classes guidance recognises for
plague; dosing and regimen selection are explicitly out of scope, because
PlagueShield does not prescribe.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..models import Citation, DrugClass

CIT_WHO_CLINICAL_2021 = Citation(
    source="WHO",
    title="Interim guidance on the clinical management of plague",
    url="https://www.who.int/publications/i/item/9789240034884",
    year=2021,
    note="Recognised therapeutic and prophylactic classes for plague.",
)

CIT_CDC_TREATMENT = Citation(
    source="CDC",
    title="Clinical Guidance for Plague — Treatment and Post-Exposure "
    "Prophylaxis",
    url="https://www.cdc.gov/plague/hcp/clinical-guidance/",
)

CIT_NELSON_MMWR = Citation(
    source="CDC (MMWR Recomm Rep)",
    title="Antimicrobial Treatment and Prophylaxis of Plague: Recommendations "
    "for Naturally Acquired Infections and Bioterrorism Response",
    identifier="MMWR 2021;70(RR-3)",
    url="https://www.cdc.gov/mmwr/volumes/70/rr/rr7003a1.htm",
    year=2021,
)

CIT_CLSI_M45 = Citation(
    source="CLSI",
    title="M45: Methods for Antimicrobial Dilution and Disk Susceptibility "
    "Testing of Infrequently Isolated or Fastidious Bacteria",
    note="Source of interpretive breakpoints applied to Y. pestis isolates.",
)

ALL_REGIMEN_CITATIONS = [
    CIT_WHO_CLINICAL_2021,
    CIT_CDC_TREATMENT,
    CIT_NELSON_MMWR,
    CIT_CLSI_M45,
]


class TreatmentRole(str, Enum):
    FIRST_LINE = "first_line"
    ALTERNATIVE = "alternative"
    PROPHYLAXIS = "prophylaxis"
    ADJUNCT_SPECIAL_SITE = "adjunct_special_site"
    NOT_RECOMMENDED = "not_recommended"


@dataclass(frozen=True)
class ClassRole:
    drug_class: DrugClass
    role: TreatmentRole
    # Weight of this class when scoring whether the overall standard-treatment
    # position is threatened. First-line classes dominate.
    regimen_weight: float
    display_name: str
    note: str


CLASS_ROLES: dict[DrugClass, ClassRole] = {
    DrugClass.AMINOGLYCOSIDE: ClassRole(
        drug_class=DrugClass.AMINOGLYCOSIDE,
        role=TreatmentRole.FIRST_LINE,
        regimen_weight=1.0,
        display_name="Aminoglycosides",
        note="Long-standing first-line class for treatment of plague in "
        "recognised guidance.",
    ),
    DrugClass.FLUOROQUINOLONE: ClassRole(
        drug_class=DrugClass.FLUOROQUINOLONE,
        role=TreatmentRole.FIRST_LINE,
        regimen_weight=1.0,
        display_name="Fluoroquinolones",
        note="First-line class; also carries a prophylaxis role.",
    ),
    DrugClass.TETRACYCLINE: ClassRole(
        drug_class=DrugClass.TETRACYCLINE,
        role=TreatmentRole.FIRST_LINE,
        regimen_weight=0.9,
        display_name="Tetracyclines",
        note="First-line class in current guidance and a principal "
        "post-exposure prophylaxis class.",
    ),
    DrugClass.PHENICOL: ClassRole(
        drug_class=DrugClass.PHENICOL,
        role=TreatmentRole.ADJUNCT_SPECIAL_SITE,
        regimen_weight=0.4,
        display_name="Phenicols",
        note="Historically reserved for specific deep-site presentations; "
        "limited contemporary use.",
    ),
    DrugClass.SULFONAMIDE: ClassRole(
        drug_class=DrugClass.SULFONAMIDE,
        role=TreatmentRole.ALTERNATIVE,
        regimen_weight=0.3,
        display_name="Sulfonamides",
        note="Alternative/limited role; inferior outcomes relative to "
        "first-line classes.",
    ),
    DrugClass.BETA_LACTAM: ClassRole(
        drug_class=DrugClass.BETA_LACTAM,
        role=TreatmentRole.NOT_RECOMMENDED,
        regimen_weight=0.0,
        display_name="Beta-lactams",
        note="Not a recommended class for plague irrespective of in vitro "
        "results; a susceptible beta-lactam result does not imply clinical "
        "utility.",
    ),
    DrugClass.MACROLIDE: ClassRole(
        drug_class=DrugClass.MACROLIDE,
        role=TreatmentRole.NOT_RECOMMENDED,
        regimen_weight=0.0,
        display_name="Macrolides",
        note="Not an established class for plague therapy.",
    ),
    DrugClass.OTHER: ClassRole(
        drug_class=DrugClass.OTHER,
        role=TreatmentRole.NOT_RECOMMENDED,
        regimen_weight=0.0,
        display_name="Other",
        note="Unclassified by the calling tool.",
    ),
}

#: Classes whose compromise would materially change management.
CORE_REGIMEN_CLASSES: tuple[DrugClass, ...] = (
    DrugClass.AMINOGLYCOSIDE,
    DrugClass.FLUOROQUINOLONE,
    DrugClass.TETRACYCLINE,
)

#: Total weight of the core regimen, used to normalise threat scores.
CORE_REGIMEN_WEIGHT: float = sum(
    CLASS_ROLES[c].regimen_weight for c in CORE_REGIMEN_CLASSES
)


def role_for(drug_class: DrugClass) -> ClassRole:
    return CLASS_ROLES.get(drug_class, CLASS_ROLES[DrugClass.OTHER])


def is_core_regimen(drug_class: DrugClass) -> bool:
    return drug_class in CORE_REGIMEN_CLASSES


def clinically_meaningful(drug_class: DrugClass) -> bool:
    """Whether a result in this class should influence the headline outlook."""
    return role_for(drug_class).regimen_weight > 0.0
