"""Out-of-distribution detection and principled abstention.

A system that reports a confident answer on a case unlike anything it has seen
is worse than one that reports nothing, because the confident answer gets
acted on. This module decides when PlagueShield should say "I cannot make a
reliable prediction here" and why.

Three independent reasons to abstain, any one of which is sufficient:

  1. **No support.** The case does not resemble any exemplar closely enough.
     Calibrated by leave-one-out conformal thresholding over the bank itself,
     so the threshold reflects how tightly the bank actually covers its own
     space rather than a number picked by hand.

  2. **Insufficient evidence.** There is too little resistance evidence for
     any method to be informative -- notably, no genomic screen and no
     phenotypic panel. This is distinct from (1): such a case may sit very
     close to the `untested_isolate` exemplar, i.e. be perfectly "in
     distribution" while carrying no information.

  3. **Unreliable inputs.** Evidence exists but cannot be trusted at face
     value: a negative screen on an assembly too poor to support it, or a
     single unrepeated result carrying the whole finding.

Reason (2) is the one most often got wrong in practice. Similarity to a
well-populated "nothing was tested" region of feature space is not knowledge,
and the abstention logic treats it as the opposite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..models import GenomicEvidence, SusceptibilityResult
from .exemplars import (
    EXEMPLAR_BANK,
    Exemplar,
    FewShotResult,
    similarity,
)


class AbstainReason(str, Enum):
    NO_SUPPORT = "no_support"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNRELIABLE_INPUTS = "unreliable_inputs"


def _loo_nonconformity(bank: tuple[Exemplar, ...]) -> list[float]:
    """Leave-one-out nonconformity scores for the bank itself."""
    scores: list[float] = []
    for i, ex in enumerate(bank):
        others = [o for j, o in enumerate(bank) if j != i]
        best = max((similarity(ex.vector, o.vector) for o in others), default=0.0)
        scores.append(1.0 - best)
    return scores


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    # Conformal-style index with finite-sample correction.
    idx = min(
        len(ordered) - 1,
        max(0, int((len(ordered) + 1) * q) - 1),
    )
    return ordered[idx]


#: Nonconformity threshold above which a case is treated as unsupported.
#: Derived from the bank, not hand-tuned.
SUPPORT_THRESHOLD: float = _quantile(_loo_nonconformity(EXEMPLAR_BANK), 0.90)

#: Minimum total similarity mass across the k neighbours for the vote to mean
#: anything at all.
MIN_SUPPORT_MASS: float = 0.35


@dataclass
class OodAssessment:
    """Whether the system should answer, and the evidence for that decision."""

    abstain: bool
    reasons: list[AbstainReason] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    nonconformity: float = 0.0
    support_threshold: float = SUPPORT_THRESHOLD
    support_mass: float = 0.0
    completeness: float = 0.0
    reliability: float = 1.0

    @property
    def novelty(self) -> float:
        """0 = sits squarely inside the bank, 1 = nothing like it."""
        return min(max(self.nonconformity, 0.0), 1.0)

    def summary(self) -> str:
        if not self.abstain:
            return "Within supported evidence distribution."
        return "; ".join(self.messages) or "Outside supported distribution."


def evidence_completeness(
    genomic: GenomicEvidence | None,
    susceptibility: list[SusceptibilityResult],
) -> float:
    """Fraction of the resistance evidence we would want that we actually have."""
    from ..knowledge.regimens import CORE_REGIMEN_CLASSES

    have = 0.0
    want = 0.0

    # Genomic screen on an adequate assembly.
    want += 1.0
    if genomic and genomic.amr_screen_performed:
        have += 1.0 if genomic.quality_sufficient else 0.4

    # Phenotypic coverage of each core class.
    tested = {r.drug_class for r in susceptibility}
    for cls in CORE_REGIMEN_CLASSES:
        want += 1.0
        if cls in tested:
            have += 1.0

    return have / want if want else 0.0


def input_reliability(
    genomic: GenomicEvidence | None,
    susceptibility: list[SusceptibilityResult],
) -> tuple[float, list[str]]:
    """How much the available evidence can be taken at face value."""
    score = 1.0
    notes: list[str] = []

    if genomic and genomic.amr_screen_performed and not genomic.quality_sufficient:
        score -= 0.45
        notes.append(
            "AMR screen was run on an assembly below quality thresholds; a "
            "negative screen cannot be relied upon"
        )

    if genomic and genomic.amr_screen_performed and not genomic.amr_tool_run:
        score -= 0.1
        notes.append("AMR screening tool and database version not recorded")

    nonsusceptible = [r for r in susceptibility if r.is_nonsusceptible]
    if nonsusceptible and not any(r.confirmed_by_repeat for r in nonsusceptible):
        score -= 0.3
        notes.append(
            "Non-susceptible result has not been confirmed by repeat testing"
        )

    no_breakpoint = [
        r for r in susceptibility if r.breakpoint_source in (None, "")
    ]
    if no_breakpoint:
        score -= 0.1
        notes.append(
            f"{len(no_breakpoint)} susceptibility result(s) lack a recorded "
            "breakpoint standard"
        )

    return max(score, 0.0), notes


def assess(
    few_shot: FewShotResult,
    genomic: GenomicEvidence | None,
    susceptibility: list[SusceptibilityResult],
) -> OodAssessment:
    """Decide whether the resistance call is safe to make."""
    nonconformity = 1.0 - few_shot.max_similarity
    completeness = evidence_completeness(genomic, susceptibility)
    reliability, reliability_notes = input_reliability(genomic, susceptibility)

    out = OodAssessment(
        abstain=False,
        nonconformity=nonconformity,
        support_mass=few_shot.support_mass,
        completeness=completeness,
        reliability=reliability,
    )

    # (1) No support in the exemplar bank.
    if nonconformity > SUPPORT_THRESHOLD or few_shot.support_mass < MIN_SUPPORT_MASS:
        out.abstain = True
        out.reasons.append(AbstainReason.NO_SUPPORT)
        out.messages.append(
            f"Evidence pattern is unlike any reference pattern "
            f"(novelty {nonconformity:.2f} > threshold {SUPPORT_THRESHOLD:.2f})"
        )

    # (2) Nothing informative was measured. Checked independently of (1),
    #     because "nothing tested" is itself a well-populated region.
    screened = bool(genomic and genomic.amr_screen_performed)
    has_phenotype = bool(susceptibility)
    if not screened and not has_phenotype:
        out.abstain = True
        out.reasons.append(AbstainReason.INSUFFICIENT_EVIDENCE)
        out.messages.append(
            "No genomic AMR screen and no phenotypic susceptibility results — "
            "there is no resistance evidence to reason from"
        )
    elif completeness < 0.34:
        out.abstain = True
        out.reasons.append(AbstainReason.INSUFFICIENT_EVIDENCE)
        out.messages.append(
            f"Resistance evidence is {completeness:.0%} complete, below the "
            "minimum for a reliable call"
        )

    # (3) Inputs present but untrustworthy.
    if reliability < 0.55:
        out.abstain = True
        out.reasons.append(AbstainReason.UNRELIABLE_INPUTS)
        out.messages.extend(reliability_notes)
    elif reliability_notes:
        out.messages.extend(reliability_notes)

    return out
