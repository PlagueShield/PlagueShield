"""Behavioural tests.

These assert the properties that make the system safe to deploy, not its exact
numbers. A test that pins the posterior to three decimal places breaks on every
calibration change and tells you nothing; a test that asserts "never reports
certainty" catches a real regression.
"""

from __future__ import annotations

import math

import pytest

from plagueshield.agents.diagnostic import (
    PROBABILITY_CEILING,
    DiagnosticAgent,
    damped_sum,
)
from plagueshield.agents.next_test import entropy, expected_information_gain
from plagueshield.data import load_all_cases, load_case
from plagueshield.fewshot.exemplars import extract_features, few_shot_vote
from plagueshield.fewshot.ood import assess as ood_assess
from plagueshield.fewshot.prior import RESISTANCE_PRIOR
from plagueshield.models import (
    AnomalyStatus,
    Confidence,
    GenomicEvidence,
    Severity,
    SusceptibilityOutlook,
)
from plagueshield.orchestrator import Pipeline


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return Pipeline()


@pytest.fixture(scope="module")
def assessments(pipeline: Pipeline) -> dict:
    return {c.case_id: pipeline.run(c) for c in load_all_cases()}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_never_reports_certainty(assessments):
    """No finite evidence should produce a probability of 1.0."""
    for case_id, a in assessments.items():
        assert a.plague_probability is not None
        assert a.plague_probability <= PROBABILITY_CEILING, case_id
        assert a.plague_probability > 0.0, case_id


def test_damping_discounts_correlated_evidence():
    """Three correlated terms must count for less than their raw sum."""
    terms = [math.log(5.0)] * 3
    damped = damped_sum(terms, damping=0.5, cap=99.0)
    assert damped < sum(terms)
    assert damped > max(terms)  # but the strongest still counts in full


def test_damping_does_not_suppress_contradicting_evidence():
    """A negative term must survive alongside a stack of positives."""
    positives = [math.log(4.0)] * 4
    with_negative = damped_sum(positives + [math.log(0.2)], 0.5, 99.0)
    without = damped_sum(positives, 0.5, 99.0)
    assert with_negative < without


def test_single_field_rapid_test_is_not_near_certain():
    """One field RDT must not yield a near-certain call (regression guard)."""
    a = Pipeline().run(load_case("PS-2020-CD-021"))
    assert a.plague_probability < 0.90
    assert a.confidence == Confidence.LOW
    assert any(f.code == "SINGLE_RESULT_DEPENDENCY" for f in a.flags)


def test_probability_and_classification_are_independent(assessments):
    """A high probability must not silently promote a case to 'confirmed'."""
    a = assessments["PS-2017-MG-014"]
    assert a.plague_probability > 0.8
    # No confirmatory-grade positive exists for this case.
    assert a.case_classification.value == "presumptive"


# ---------------------------------------------------------------------------
# Resistance: the rare-positive problem
# ---------------------------------------------------------------------------


def test_resistance_prior_is_low_but_not_zero():
    assert 0.0 < RESISTANCE_PRIOR.mean < 0.01
    lo, hi = RESISTANCE_PRIOR.credible_interval()
    assert lo >= 0.0 and hi > RESISTANCE_PRIOR.mean


def test_absent_evidence_is_not_reported_as_susceptible(assessments):
    """The core safety property: no screen + no panel must never read as clean."""
    a = assessments["PS-2017-MG-014"]  # no genomic screen, no AST
    assert a.standard_treatment_susceptibility == (
        SusceptibilityOutlook.INSUFFICIENT_EVIDENCE
    )
    assert a.resistance_anomaly == AnomalyStatus.INSUFFICIENT_EVIDENCE
    assert a.verdicts["resistance"].abstained


def test_untested_isolate_abstains_rather_than_reassures():
    """Similarity to the 'nothing tested' exemplar must not become confidence."""
    vector = extract_features(None, [])
    vote = few_shot_vote(vector)
    ood = ood_assess(vote, None, [])
    assert ood.abstain
    assert any(r.value == "insufficient_evidence" for r in ood.reasons)


def test_poor_quality_genome_does_not_license_a_clean_screen():
    """A negative screen on a bad assembly must be treated as unreliable."""
    bad = GenomicEvidence(
        read_depth_x=4.0,
        completeness_pct=55.0,
        contamination_pct=12.0,
        amr_screen_performed=True,
        amr_tool_run="AMRFinderPlus 3.12.8",
    )
    assert not bad.quality_sufficient
    vote = few_shot_vote(extract_features(bad, []))
    ood = ood_assess(vote, bad, [])
    assert ood.abstain
    assert ood.reliability < 1.0


def test_confirmed_resistance_is_not_downgraded(assessments):
    """Phenotype governs: a non-susceptible core result must escalate."""
    a = assessments["PS-SYN-DISC-01"]
    assert a.resistance_anomaly in (AnomalyStatus.SUSPECTED, AnomalyStatus.CONFIRMED)
    assert a.standard_treatment_susceptibility in (
        SusceptibilityOutlook.POSSIBLY_COMPROMISED,
        SusceptibilityOutlook.COMPROMISED,
    )


# ---------------------------------------------------------------------------
# Discordance
# ---------------------------------------------------------------------------


def test_genotype_phenotype_conflict_is_critical_and_escalates(assessments):
    """The scenario from the brief: genome says susceptible, lab says resistant."""
    a = assessments["PS-SYN-DISC-01"]
    codes = {f.code for f in a.flags}
    assert "GENOTYPE_SUSCEPTIBLE_PHENOTYPE_RESISTANT" in codes
    conflict = next(
        f for f in a.flags if f.code == "GENOTYPE_SUSCEPTIBLE_PHENOTYPE_RESISTANT"
    )
    assert conflict.severity == Severity.CRITICAL
    assert a.human_review_required
    assert any("disagree" in line for line in a.escalation)


def test_species_conflict_is_flagged(assessments):
    a = assessments["PS-SYN-SPEC-01"]
    assert any(f.code == "SPECIES_CALL_CONFLICT" for f in a.flags)
    assert a.human_review_required


def test_unconfirmed_resistance_drives_repeat_testing(assessments):
    """An unrepeated non-susceptible result should rank repeat testing first."""
    a = assessments["PS-SYN-DISC-01"]
    assert a.ranked_tests
    assert a.ranked_tests[0].key == "ast_repeat_confirm"
    assert "Repeat susceptibility" in (a.most_valuable_next_result or "")


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def test_sparse_case_is_declared_unreliable(assessments):
    a = assessments["PS-SYN-OOD-01"]
    u = a.verdicts["uncertainty"]
    assert u.score < 0.6
    assert a.confidence == Confidence.LOW
    assert any(f.code == "SPARSE_EVIDENCE" for f in a.flags)


def test_uncertainty_can_only_lower_confidence(assessments):
    """The veto is one-directional."""
    order = ["Low", "Moderate", "High"]
    for case_id, a in assessments.items():
        ceiling = a.verdicts["uncertainty"].confidence
        assert order.index(a.confidence.value) <= order.index(ceiling.value), case_id


# ---------------------------------------------------------------------------
# Next-test information gain
# ---------------------------------------------------------------------------


def test_eig_is_zero_when_already_certain():
    assert expected_information_gain(0.9999, 0.95, 0.99) < 1e-3
    assert expected_information_gain(0.0001, 0.95, 0.99) < 1e-3


def test_eig_peaks_at_maximum_uncertainty():
    mid = expected_information_gain(0.5, 0.9, 0.9)
    assert mid > expected_information_gain(0.9, 0.9, 0.9)
    assert mid > expected_information_gain(0.1, 0.9, 0.9)


def test_eig_never_exceeds_prior_entropy():
    for p in (0.05, 0.2, 0.5, 0.8, 0.95):
        assert expected_information_gain(p, 0.95, 0.95) <= entropy(p) + 1e-9


def test_uninformative_test_yields_no_information():
    """A coin-flip test (sens = 1 - spec) carries zero information."""
    assert expected_information_gain(0.4, 0.5, 0.5) < 1e-9


def test_isolate_bottleneck_is_surfaced(assessments):
    """Without an isolate, the system should say so rather than silently rank down."""
    a = assessments["PS-2017-MG-014"]  # culture negative
    assert any(f.code == "ISOLATE_IS_BOTTLENECK" for f in a.flags)


# ---------------------------------------------------------------------------
# Pipeline robustness and provenance
# ---------------------------------------------------------------------------


def test_every_bundled_case_assesses_without_error(assessments):
    assert len(assessments) == 8
    for case_id, a in assessments.items():
        assert a.report_markdown, case_id
        assert a.report_ascii, case_id
        assert a.escalation, case_id
        assert a.citations, case_id


def test_ascii_report_stays_inside_its_box(assessments):
    for case_id, a in assessments.items():
        for line in a.report_ascii.splitlines():
            assert len(line) <= 70, f"{case_id}: {line!r}"


def test_failing_agent_degrades_rather_than_crashes(monkeypatch):
    """A broken agent must yield a flagged failure verdict, not an exception."""
    pipe = Pipeline()

    def boom(self, case, context):
        raise RuntimeError("simulated agent failure")

    monkeypatch.setattr(type(pipe.resistance), "run", boom)
    a = pipe.run(load_case("PS-2021-US-003"))

    assert a.verdicts["resistance"].abstained
    assert any(f.code == "AGENT_FAILURE" for f in a.flags)
    assert a.report_markdown  # the rest of the pipeline still produced a report


def test_every_case_carries_provenance():
    for case in load_all_cases():
        assert case.provenance.description
        assert case.provenance.deidentified
        # No case should carry a name-like or record-locator field.
        dumped = case.model_dump()
        assert "name" not in dumped
        assert "mrn" not in dumped


def test_system_never_issues_dosing_instructions(assessments):
    """It names drug classes and escalates; it must not direct therapy.

    Matching is on *prescriptive constructions* rather than on words like
    "prescribe", which legitimately appear in the disclaimer ("does not
    prescribe"). A naive substring check flags its own safety notice.
    """
    import re

    banned = [
        r"\d+\s*mg\b",            # a dose
        r"\bmg\s*/\s*kg\b",       # weight-based dosing
        r"\bq\d+h\b",             # frequency shorthand
        r"\b(?:once|twice|three times)\s+daily\b",
        r"\badminister\b",
        r"\bstart\s+(?:the\s+)?patient\s+on\b",
        r"\bswitch\s+to\b",
        r"\btreat\s+with\b",
        r"\b(?:should|must)\s+(?:be\s+)?(?:given|prescribed)\b",
    ]
    for case_id, a in assessments.items():
        text = (a.report_markdown + "\n" + a.report_ascii).lower()
        for pattern in banned:
            match = re.search(pattern, text)
            assert match is None, (
                f"{case_id} contains prescriptive text {match.group(0)!r} "
                f"matching {pattern!r}"
            )


def test_disclaimer_is_present_on_every_report(assessments):
    for case_id, a in assessments.items():
        assert "does not prescribe" in a.report_markdown.lower(), case_id
        assert "decision support only" in a.report_ascii.lower(), case_id
