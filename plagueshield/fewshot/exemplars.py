"""Few-shot resistance inference over a small, explicitly-labelled exemplar bank.

Why few-shot rather than a trained model: there are not enough public
resistant Y. pestis isolates to fit a supervised classifier that generalises.
Any model trained on this distribution would be dominated by the negative
class. So instead of fitting weights, we keep a small bank of hand-labelled
*evidence patterns* and reason by similarity to them -- with an explicit
abstention when the case does not resemble anything in the bank.

The representation is deliberately coarse. An exemplar is a pattern of
(genotype signal, phenotype signal, completeness) across drug classes -- never
a specific determinant. Two cases are "similar" when their evidence has the
same shape, not when they share a gene. This keeps the bank clinically
meaningful while containing nothing that describes how resistance works.

Missingness is first-class: every signal carries a companion "known" channel,
so "screened and negative" is a different point in feature space from "never
screened". Collapsing those two is the single most dangerous thing this module
could do, because it is what turns an untested isolate into a reassuring
report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..knowledge.regimens import CORE_REGIMEN_CLASSES
from ..models import (
    DrugClass,
    GenomicEvidence,
    Interpretation,
    SusceptibilityResult,
)

#: Feature layout, documented so the vector stays interpretable in the UI.
#: Per core class: determinant-present, determinant-known,
#:                 phenotype-nonsusceptible, phenotype-known
#: Then globals.
GLOBAL_FEATURES = (
    "breadth_of_determinants",
    "genomic_screen_performed",
    "genomic_quality_sufficient",
    "phenotype_coverage",
    "confirmed_by_repeat",
)

FEATURE_NAMES: tuple[str, ...] = tuple(
    [
        f"{cls.value}:{suffix}"
        for cls in CORE_REGIMEN_CLASSES
        for suffix in ("geno", "geno_known", "pheno", "pheno_known")
    ]
    + list(GLOBAL_FEATURES)
)

N_FEATURES = len(FEATURE_NAMES)

#: Weights emphasise agreement between genotype and phenotype over either
#: alone, and weight the "known" channels highly so that completeness drives
#: similarity as much as signal does.
FEATURE_WEIGHTS: tuple[float, ...] = tuple(
    [
        w
        for _ in CORE_REGIMEN_CLASSES
        for w in (1.3, 1.0, 1.5, 1.0)
    ]
    + [0.8, 0.9, 0.7, 0.9, 0.6]
)


@dataclass(frozen=True)
class Exemplar:
    """A labelled evidence pattern."""

    key: str
    label: int  # 1 = resistance anomaly present, 0 = absent
    vector: tuple[float, ...]
    description: str
    basis: str

    def __post_init__(self) -> None:
        if len(self.vector) != N_FEATURES:
            raise ValueError(
                f"exemplar {self.key}: expected {N_FEATURES} features, "
                f"got {len(self.vector)}"
            )


def _pattern(
    *,
    amino: tuple[float, float, float, float],
    fluoro: tuple[float, float, float, float],
    tetra: tuple[float, float, float, float],
    breadth: float,
    screened: float,
    quality: float,
    pheno_cov: float,
    repeat: float,
) -> tuple[float, ...]:
    """Build a feature vector in the documented layout."""
    return (*amino, *fluoro, *tetra, breadth, screened, quality, pheno_cov, repeat)


#: The bank. Negatives are plentiful and reflect the real public distribution;
#: positives are few, which is the whole point -- they are carried as named
#: patterns rather than learned.
EXEMPLAR_BANK: tuple[Exemplar, ...] = (
    Exemplar(
        key="wildtype_fully_characterised",
        label=0,
        vector=_pattern(
            amino=(0.0, 1.0, 0.0, 1.0),
            fluoro=(0.0, 1.0, 0.0, 1.0),
            tetra=(0.0, 1.0, 0.0, 1.0),
            breadth=0.0,
            screened=1.0,
            quality=1.0,
            pheno_cov=1.0,
            repeat=0.0,
        ),
        description="Complete genomic screen and full phenotypic panel, all "
        "core classes susceptible.",
        basis="Modal pattern among publicly characterised Y. pestis isolates.",
    ),
    Exemplar(
        key="wildtype_genome_only",
        label=0,
        vector=_pattern(
            amino=(0.0, 1.0, 0.0, 0.0),
            fluoro=(0.0, 1.0, 0.0, 0.0),
            tetra=(0.0, 1.0, 0.0, 0.0),
            breadth=0.0,
            screened=1.0,
            quality=1.0,
            pheno_cov=0.0,
            repeat=0.0,
        ),
        description="Good-quality genome, no determinants called, no "
        "phenotypic testing yet.",
        basis="Common sequencing-first workflow in reference laboratories.",
    ),
    Exemplar(
        key="wildtype_phenotype_only",
        label=0,
        vector=_pattern(
            amino=(0.0, 0.0, 0.0, 1.0),
            fluoro=(0.0, 0.0, 0.0, 1.0),
            tetra=(0.0, 0.0, 0.0, 1.0),
            breadth=0.0,
            screened=0.0,
            quality=0.0,
            pheno_cov=1.0,
            repeat=0.0,
        ),
        description="Full susceptibility panel, all core classes susceptible, "
        "no sequencing performed.",
        basis="Typical of settings without sequencing capacity.",
    ),
    Exemplar(
        key="wildtype_partial_phenotype",
        label=0,
        vector=_pattern(
            amino=(0.0, 1.0, 0.0, 1.0),
            fluoro=(0.0, 1.0, 0.0, 0.0),
            tetra=(0.0, 1.0, 0.0, 1.0),
            breadth=0.0,
            screened=1.0,
            quality=1.0,
            pheno_cov=0.66,
            repeat=0.0,
        ),
        description="Genome clean, partial phenotypic panel, tested classes "
        "susceptible.",
        basis="Panel composition varies between reference laboratories.",
    ),
    Exemplar(
        key="low_quality_genome_no_phenotype",
        label=0,
        vector=_pattern(
            amino=(0.0, 0.5, 0.0, 0.0),
            fluoro=(0.0, 0.5, 0.0, 0.0),
            tetra=(0.0, 0.5, 0.0, 0.0),
            breadth=0.0,
            screened=1.0,
            quality=0.0,
            pheno_cov=0.0,
            repeat=0.0,
        ),
        description="Low-depth or incomplete assembly; a negative screen "
        "carries little weight.",
        basis="Direct-from-specimen metagenomic runs with poor recovery.",
    ),
    Exemplar(
        key="noncore_determinant_only",
        label=0,
        vector=_pattern(
            amino=(0.0, 1.0, 0.0, 1.0),
            fluoro=(0.0, 1.0, 0.0, 1.0),
            tetra=(0.0, 1.0, 0.0, 1.0),
            breadth=0.25,
            screened=1.0,
            quality=1.0,
            pheno_cov=1.0,
            repeat=0.0,
        ),
        description="Determinant called in a class with no role in plague "
        "therapy; core classes unaffected and phenotypically susceptible.",
        basis="Intrinsic or incidental elements reported by screening tools; "
        "clinically unremarkable for plague.",
    ),
    Exemplar(
        key="untested_isolate",
        label=0,
        vector=_pattern(
            amino=(0.0, 0.0, 0.0, 0.0),
            fluoro=(0.0, 0.0, 0.0, 0.0),
            tetra=(0.0, 0.0, 0.0, 0.0),
            breadth=0.0,
            screened=0.0,
            quality=0.0,
            pheno_cov=0.0,
            repeat=0.0,
        ),
        description="No resistance evidence of any kind.",
        basis="Anchor for the abstention boundary — resembles nothing "
        "informative, and must never score as reassuring.",
    ),
    Exemplar(
        key="single_core_phenotype_nonsusceptible_unconfirmed",
        label=1,
        vector=_pattern(
            amino=(0.0, 1.0, 1.0, 1.0),
            fluoro=(0.0, 1.0, 0.0, 1.0),
            tetra=(0.0, 1.0, 0.0, 1.0),
            breadth=0.0,
            screened=1.0,
            quality=1.0,
            pheno_cov=1.0,
            repeat=0.0,
        ),
        description="One core class non-susceptible on a single unrepeated "
        "test, with no corresponding genotypic signal.",
        basis="Pattern most often produced by a testing artefact, but it is "
        "the pattern a genuine novel event would also present as. Labelled "
        "positive so it escalates; the Discordance Agent then demands repeat "
        "testing rather than accepting it.",
    ),
    Exemplar(
        key="single_core_concordant",
        label=1,
        vector=_pattern(
            amino=(1.0, 1.0, 1.0, 1.0),
            fluoro=(0.0, 1.0, 0.0, 1.0),
            tetra=(0.0, 1.0, 0.0, 1.0),
            breadth=0.33,
            screened=1.0,
            quality=1.0,
            pheno_cov=1.0,
            repeat=1.0,
        ),
        description="One core class with a determinant called and a confirmed "
        "non-susceptible phenotype — genotype and phenotype agree.",
        basis="Concordant single-class resistance; the strongest form of "
        "evidence available short of multi-class involvement.",
    ),
    Exemplar(
        key="multiclass_concordant",
        label=1,
        vector=_pattern(
            amino=(1.0, 1.0, 1.0, 1.0),
            fluoro=(0.0, 1.0, 0.0, 1.0),
            tetra=(1.0, 1.0, 1.0, 1.0),
            breadth=0.8,
            screened=1.0,
            quality=1.0,
            pheno_cov=1.0,
            repeat=1.0,
        ),
        description="Determinants called across multiple classes including "
        "core therapeutic classes, with concordant confirmed phenotype.",
        basis="Corresponds to the rare documented reports of transferable "
        "multiresistance in Y. pestis. Pattern-level only — no mechanism is "
        "represented here.",
    ),
    Exemplar(
        key="genotype_positive_phenotype_susceptible",
        label=1,
        vector=_pattern(
            amino=(1.0, 1.0, 0.0, 1.0),
            fluoro=(0.0, 1.0, 0.0, 1.0),
            tetra=(0.0, 1.0, 0.0, 1.0),
            breadth=0.33,
            screened=1.0,
            quality=1.0,
            pheno_cov=1.0,
            repeat=0.0,
        ),
        description="Determinant called in a core class but the isolate tests "
        "susceptible.",
        basis="Determinants are not always expressed. Labelled positive as an "
        "*anomaly requiring review*, not as established resistance — the "
        "distinction the Discordance Agent draws.",
    ),
    Exemplar(
        key="phenotype_nonsusceptible_no_screen",
        label=1,
        vector=_pattern(
            amino=(0.0, 0.0, 1.0, 1.0),
            fluoro=(0.0, 0.0, 0.0, 1.0),
            tetra=(0.0, 0.0, 0.0, 1.0),
            breadth=0.0,
            screened=0.0,
            quality=0.0,
            pheno_cov=1.0,
            repeat=1.0,
        ),
        description="Confirmed non-susceptible phenotype in a core class with "
        "no genomic screen performed.",
        basis="Phenotype is the clinically governing result; absence of "
        "sequencing does not downgrade it.",
    ),
)


def extract_features(
    genomic: GenomicEvidence | None,
    susceptibility: list[SusceptibilityResult],
) -> tuple[float, ...]:
    """Project a case's resistance evidence into the exemplar feature space."""
    screened = bool(genomic and genomic.amr_screen_performed)
    quality = bool(genomic and genomic.quality_sufficient) if screened else False
    determinant_classes = (
        genomic.classes_with_determinants() if genomic else set()
    )

    by_class: dict[DrugClass, list[SusceptibilityResult]] = {}
    for result in susceptibility:
        by_class.setdefault(result.drug_class, []).append(result)

    vector: list[float] = []
    for cls in CORE_REGIMEN_CLASSES:
        # Genotype channel. A screen on a poor assembly is "partially known".
        if screened:
            geno = 1.0 if cls in determinant_classes else 0.0
            geno_known = 1.0 if quality else 0.5
        else:
            geno, geno_known = 0.0, 0.0

        # Phenotype channel.
        results = by_class.get(cls, [])
        if results:
            pheno = 1.0 if any(r.is_nonsusceptible for r in results) else 0.0
            pheno_known = 1.0
        else:
            pheno, pheno_known = 0.0, 0.0

        vector.extend((geno, geno_known, pheno, pheno_known))

    # Globals.
    breadth = min(len(determinant_classes) / 4.0, 1.0) if screened else 0.0
    pheno_cov = sum(
        1.0 for cls in CORE_REGIMEN_CLASSES if by_class.get(cls)
    ) / len(CORE_REGIMEN_CLASSES)
    repeat = 1.0 if any(r.confirmed_by_repeat for r in susceptibility) else 0.0

    vector.extend(
        (
            breadth,
            1.0 if screened else 0.0,
            1.0 if quality else 0.0,
            pheno_cov,
            repeat,
        )
    )
    return tuple(vector)


def weighted_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Weighted Euclidean distance in feature space."""
    total = 0.0
    for x, y, w in zip(a, b, FEATURE_WEIGHTS):
        d = x - y
        total += w * d * d
    return math.sqrt(total)


#: Kernel bandwidth. Set so that a case differing from an exemplar in one
#: weighted feature retains meaningful similarity, while a case differing
#: across several does not.
KERNEL_BANDWIDTH: float = 1.1


def similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    d = weighted_distance(a, b)
    return math.exp(-((d / KERNEL_BANDWIDTH) ** 2))


@dataclass
class Neighbour:
    exemplar: Exemplar
    similarity: float
    distance: float


@dataclass
class FewShotResult:
    """Output of the few-shot vote, including everything needed to audit it."""

    score: float
    """Similarity-weighted fraction of neighbour mass labelled positive."""

    neighbours: list[Neighbour]
    max_similarity: float
    support_mass: float
    """Total similarity mass. Low mass means the bank has little to say."""

    vector: tuple[float, ...]

    @property
    def nearest(self) -> Neighbour | None:
        return self.neighbours[0] if self.neighbours else None

    def explain(self, top: int = 3) -> list[str]:
        lines = []
        for n in self.neighbours[:top]:
            verdict = "anomaly" if n.exemplar.label == 1 else "no anomaly"
            lines.append(
                f"{n.similarity:.2f} similar to '{n.exemplar.key}' "
                f"({verdict}): {n.exemplar.description}"
            )
        return lines


def few_shot_vote(
    vector: tuple[float, ...],
    bank: tuple[Exemplar, ...] = EXEMPLAR_BANK,
    k: int = 5,
) -> FewShotResult:
    """Similarity-weighted k-nearest-exemplar vote."""
    neighbours = sorted(
        (
            Neighbour(
                exemplar=ex,
                similarity=similarity(vector, ex.vector),
                distance=weighted_distance(vector, ex.vector),
            )
            for ex in bank
        ),
        key=lambda n: -n.similarity,
    )
    top = neighbours[:k]
    mass = sum(n.similarity for n in top)
    if mass <= 1e-9:
        score = 0.0
    else:
        score = sum(n.similarity for n in top if n.exemplar.label == 1) / mass

    return FewShotResult(
        score=score,
        neighbours=neighbours,
        max_similarity=neighbours[0].similarity if neighbours else 0.0,
        support_mass=mass,
        vector=vector,
    )
