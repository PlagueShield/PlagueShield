"""Base-rate prior for antimicrobial resistance in Yersinia pestis.

The defining statistical problem for this system: resistance in Y. pestis is
genuinely rare, so a naive classifier trained on available data learns to say
"susceptible" always and is right almost every time while being useless. The
answer is not to pretend the positives are more common than they are, but to
carry the rarity explicitly as a prior and make the evidence do the work of
moving off it -- and to say so when it cannot.

Two things follow from a low prior that operators must understand, so the
agent states them rather than hiding them:

  1. A low prior means a *positive* resistance signal needs strong evidence
     before it is believed. One unconfirmed result should not flip the call.
  2. A low prior does NOT mean a negative screen is reassuring in proportion.
     Absence of evidence is weak evidence of absence when the screen itself is
     incomplete, and the system reports "insufficient evidence" rather than
     "susceptible" in that case.

Prior calibration
-----------------
Published surveillance of Y. pestis isolates has found resistance to be rare;
transferable multiresistance has been documented, but as isolated reports
rather than as an established circulating population. We encode this as a Beta
prior with a small pseudo-count of positives against a much larger negative
count, which yields a low central estimate with honestly wide uncertainty --
not a false zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Number of Y. pestis isolates with public, standardised AMR genotype
#: characterisation. Anchored to the count of public Y. pestis assemblies in
#: NCBI, refreshed by scripts/fetch_public_data.py (3,670 as of 2026-10-05).
#:
#: We deliberately use a figure *below* the raw assembly count. Not every
#: public assembly has been screened with a current AMR catalogue, and many
#: are clonally related isolates from the same outbreak rather than
#: independent observations. Treating 3,670 correlated genomes as 3,670
#: independent trials would make the prior far more confident than the
#: evidence supports, which is the same compounding error the Diagnostic
#: Agent damps. A conservative effective sample size keeps the credible
#: interval honestly wide.
CHARACTERISED_ISOLATES: int = 1800

#: Isolates in the public record described as carrying transferable
#: multiresistance. Existence-level count only -- see knowledge/regimens.py for
#: why mechanism detail is deliberately excluded from this codebase.
DOCUMENTED_RESISTANT: int = 2

#: Jeffreys prior offsets, so the posterior never collapses to exactly zero.
JEFFREYS_A: float = 0.5
JEFFREYS_B: float = 0.5


@dataclass(frozen=True)
class BetaPrior:
    """A Beta(alpha, beta) belief over a probability."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        n = a + b
        return (a * b) / (n * n * (n + 1.0))

    @property
    def stdev(self) -> float:
        return math.sqrt(self.variance)

    def credible_interval(self, mass: float = 0.95) -> tuple[float, float]:
        """Normal-approximation credible interval, clipped to [0, 1].

        Adequate for reporting; the mean is what drives decisions.
        """
        z = 1.959963985 if abs(mass - 0.95) < 1e-9 else 1.644853627
        lo = max(0.0, self.mean - z * self.stdev)
        hi = min(1.0, self.mean + z * self.stdev)
        return lo, hi

    def update(self, successes: float, failures: float) -> "BetaPrior":
        return BetaPrior(self.alpha + successes, self.beta + failures)

    @property
    def effective_n(self) -> float:
        return self.alpha + self.beta


#: Prior probability that an arbitrary, otherwise uncharacterised Y. pestis
#: isolate carries clinically meaningful resistance in a core regimen class.
RESISTANCE_PRIOR = BetaPrior(
    alpha=JEFFREYS_A + DOCUMENTED_RESISTANT,
    beta=JEFFREYS_B + (CHARACTERISED_ISOLATES - DOCUMENTED_RESISTANT),
)


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class EvidenceUpdate:
    """One piece of evidence moving the resistance posterior, with its reason."""

    source: str
    log_likelihood_ratio: float
    description: str

    @property
    def odds_factor(self) -> float:
        return math.exp(self.log_likelihood_ratio)


class ResistancePosterior:
    """Accumulates evidence in log-odds space on top of the base-rate prior."""

    def __init__(self, prior: BetaPrior = RESISTANCE_PRIOR) -> None:
        self.prior = prior
        self._log_odds = logit(prior.mean)
        self.updates: list[EvidenceUpdate] = []

    def apply(self, update: EvidenceUpdate) -> "ResistancePosterior":
        self._log_odds += update.log_likelihood_ratio
        self.updates.append(update)
        return self

    @property
    def log_odds(self) -> float:
        return self._log_odds

    @property
    def probability(self) -> float:
        return inv_logit(self._log_odds)

    @property
    def total_llr(self) -> float:
        return sum(u.log_likelihood_ratio for u in self.updates)

    @property
    def evidence_strength(self) -> float:
        """Absolute magnitude of evidence moved, regardless of direction.

        Used by the Uncertainty Agent: a posterior that sits near the prior
        because nothing moved it is very different from one that sits there
        because strong evidence pushed both ways.
        """
        return sum(abs(u.log_likelihood_ratio) for u in self.updates)

    def explain(self) -> list[str]:
        lines = [
            f"Base rate prior: {self.prior.mean:.4f} "
            f"({DOCUMENTED_RESISTANT} documented among ~{CHARACTERISED_ISOLATES} "
            f"characterised isolates)"
        ]
        for u in self.updates:
            direction = "↑" if u.log_likelihood_ratio > 0 else "↓"
            lines.append(
                f"{direction} {u.source}: ×{u.odds_factor:.2f} odds — {u.description}"
            )
        lines.append(f"Posterior probability of resistance: {self.probability:.3f}")
        return lines
