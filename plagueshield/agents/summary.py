"""Clinical Summary Agent — produces a short, auditable report.

Audience: infectious-disease physicians and clinical microbiologists. They do
not need to be taught what plague is; they need to know what this system
concluded, how confident it is, what it could not determine, and what to do
next. So the report leads with the answer, states its own limits, and keeps
the full reasoning chain available underneath rather than in place of it.

Explicit non-goal: this agent does not prescribe. It names drug *classes*
whose standing is in question and routes to the humans who prescribe. The
distinction is load-bearing — a decision-support tool that issues treatment
instructions acquires a duty of care it cannot discharge.
"""

from __future__ import annotations

from ..models import (
    AnomalyStatus,
    CaseRecord,
    Confidence,
    Flag,
    Likelihood,
    Severity,
    SusceptibilityOutlook,
    Verdict,
)
from .base import Agent

RULE = "═" * 66
THIN = "─" * 66


class ClinicalSummaryAgent(Agent):
    name = "summary"
    description = "Produces a short, auditable report for specialist review"
    depends_on = ("diagnostic", "resistance", "evidence", "discordance",
                  "uncertainty", "next_test")

    def run(self, case: CaseRecord, context: dict[str, Verdict]) -> Verdict:
        diagnostic = context.get("diagnostic")
        resistance = context.get("resistance")
        discordance = context.get("discordance")
        uncertainty = context.get("uncertainty")
        next_test = context.get("next_test")
        evidence = context.get("evidence")

        likelihood = Likelihood(
            diagnostic.data.get("likelihood", "Indeterminate")
            if diagnostic
            else "Indeterminate"
        )
        outlook = SusceptibilityOutlook(
            resistance.data.get("outlook", "Insufficient evidence")
            if resistance
            else "Insufficient evidence"
        )
        anomaly = AnomalyStatus(
            resistance.data.get("anomaly", "Insufficient evidence")
            if resistance
            else "Insufficient evidence"
        )
        confidence = (
            uncertainty.confidence if uncertainty else Confidence.LOW
        )
        recommendation = (
            next_test.data.get("recommendation") if next_test else None
        )

        escalation, review_required = self._escalation(case, context)

        all_flags: list[Flag] = []
        for verdict in context.values():
            all_flags.extend(verdict.flags)

        markdown = self._markdown(
            case, context, likelihood, outlook, anomaly, confidence,
            recommendation, escalation, all_flags,
        )
        ascii_report = self._ascii(
            case, likelihood, outlook, anomaly, confidence, recommendation,
            escalation, review_required,
        )

        return self.verdict(
            headline="Clinical summary prepared",
            confidence=confidence,
            rationale=[
                f"Plague likelihood: {likelihood.value}",
                f"Standard-treatment susceptibility: {outlook.value}",
                f"Resistance anomaly: {anomaly.value}",
                f"Confidence: {confidence.value}",
                f"Most valuable next result: {recommendation or 'None identified'}",
                f"Human review required: {'yes' if review_required else 'no'}",
            ],
            citations=evidence.citations if evidence else [],
            data={
                "report_markdown": markdown,
                "report_ascii": ascii_report,
                "escalation": escalation,
                "human_review_required": review_required,
            },
        )

    # -- escalation ------------------------------------------------------

    def _escalation(
        self, case: CaseRecord, context: dict[str, Verdict]
    ) -> tuple[list[str], bool]:
        lines: list[str] = []
        review = False

        diagnostic = context.get("diagnostic")
        resistance = context.get("resistance")
        discordance = context.get("discordance")
        uncertainty = context.get("uncertainty")

        # The condition named in the brief: genomic and phenotypic disagree.
        if discordance:
            conflicts = discordance.data.get("conflicts", [])
            geno_pheno = [
                c
                for c in conflicts
                if c["code"]
                in (
                    "GENOTYPE_SUSCEPTIBLE_PHENOTYPE_RESISTANT",
                    "GENOTYPE_RESISTANT_PHENOTYPE_SUSCEPTIBLE",
                )
            ]
            if geno_pheno:
                review = True
                lines.append(
                    "HUMAN REVIEW REQUIRED — genomic and phenotypic results "
                    "disagree. Phenotype governs interim management; the "
                    "conflict must be adjudicated by a clinical microbiologist."
                )
            if discordance.data.get("n_critical", 0):
                review = True
                lines.append(
                    f"{discordance.data['n_critical']} critical evidence "
                    "conflict(s) require resolution before this assessment is "
                    "relied upon."
                )

        if resistance and resistance.data.get("anomaly") in (
            AnomalyStatus.SUSPECTED.value,
            AnomalyStatus.CONFIRMED.value,
        ):
            review = True
            lines.append(
                "Resistance anomaly in a core therapeutic class — notify the "
                "reference laboratory and the responsible public health "
                "authority. Resistance in Y. pestis is rare and reportable."
            )

        if uncertainty and uncertainty.abstained:
            review = True
            lines.append(
                "System reports it CANNOT make a reliable prediction for this "
                "case. Specialist assessment required; do not treat the "
                "numbers below as decision-grade."
            )

        form = diagnostic.data.get("clinical_form") if diagnostic else None
        probability = (diagnostic.score if diagnostic else 0.0) or 0.0
        if form == "pneumonic" and probability >= 0.3:
            review = True
            lines.append(
                "Pneumonic presentation with material probability of plague — "
                "person-to-person transmissible. Institute respiratory "
                "isolation, trace contacts, and notify public health "
                "immediately under IHR (2005); do not wait for confirmation."
            )

        if case.exposure.laboratory_exposure:
            review = True
            lines.append(
                "Laboratory exposure reported — notify the institutional "
                "biosafety officer and select-agent programme."
            )

        if case.exposure.cluster_size and case.exposure.cluster_size > 1:
            lines.append(
                f"Epidemiologically linked cluster of {case.exposure.cluster_size} "
                "cases — treat as an outbreak investigation."
            )

        if not lines:
            lines.append(
                "No escalation trigger met. Routine specialist review still "
                "applies before any clinical action."
            )

        return lines, review

    # -- renderers -------------------------------------------------------

    def _markdown(
        self,
        case: CaseRecord,
        context: dict[str, Verdict],
        likelihood: Likelihood,
        outlook: SusceptibilityOutlook,
        anomaly: AnomalyStatus,
        confidence: Confidence,
        recommendation: str | None,
        escalation: list[str],
        flags: list[Flag],
    ) -> str:
        diagnostic = context.get("diagnostic")
        resistance = context.get("resistance")
        uncertainty = context.get("uncertainty")
        next_test = context.get("next_test")

        out: list[str] = []
        out.append(f"# PlagueShield assessment — {case.label}")
        out.append("")
        out.append(f"**Case ID** `{case.case_id}`  ")
        out.append(
            f"**Record origin** {case.provenance.origin.value} — "
            f"{case.provenance.description}  "
        )
        if case.country:
            loc = case.country + (f" / {case.admin1}" if case.admin1 else "")
            out.append(f"**Location** {loc}  ")
        out.append("")
        out.append("## Summary")
        out.append("")
        out.append(f"| | |")
        out.append(f"|---|---|")
        out.append(f"| Plague likelihood | **{likelihood.value}** |")
        out.append(
            f"| Standard-treatment susceptibility | **{outlook.value}** |"
        )
        out.append(f"| Resistance anomaly | **{anomaly.value}** |")
        out.append(f"| Confidence | **{confidence.value}** |")
        out.append(
            f"| Most valuable next result | **{recommendation or 'None identified'}** |"
        )
        out.append("")

        out.append("## Escalation")
        out.append("")
        for line in escalation:
            out.append(f"- {line}")
        out.append("")

        if diagnostic:
            out.append("## Is this plague?")
            out.append("")
            out.append(
                f"Posterior probability **{(diagnostic.score or 0):.1%}**, "
                f"surveillance classification "
                f"**{diagnostic.data.get('classification', 'unknown')}**, "
                f"clinical form **{diagnostic.data.get('clinical_form', 'unknown')}**."
            )
            out.append("")
            for line in diagnostic.rationale:
                out.append(f"- {line}")
            out.append("")

        if resistance:
            out.append("## Will standard treatment work?")
            out.append("")
            if resistance.abstained:
                out.append(
                    f"> **The system abstains on this question.** "
                    f"{resistance.abstain_reason}"
                )
                out.append("")
            per_class = resistance.data.get("per_class", {})
            if per_class:
                out.append("| Drug class | Role | Status | Determinant | Phenotype |")
                out.append("|---|---|---|---|---|")
                for name, info in sorted(
                    per_class.items(), key=lambda kv: not kv[1]["core"]
                ):
                    out.append(
                        f"| {name}{' ⬥' if info['core'] else ''} "
                        f"| {info['role'].replace('_', ' ')} "
                        f"| {info['status']} "
                        f"| {'yes' if info['determinant_called'] else 'no'} "
                        f"| {'tested' if info['phenotype_tested'] else '—'} |"
                    )
                out.append("")
                out.append("⬥ = core therapeutic class for plague")
                out.append("")
            for line in resistance.rationale:
                out.append(f"- {line}")
            out.append("")

        if uncertainty:
            out.append("## What the system does not know")
            out.append("")
            d = uncertainty.data
            out.append(
                f"Reliability **{d.get('reliability', 0):.0%}** · "
                f"coverage {d.get('coverage', 0):.0%} · "
                f"novelty {d.get('novelty', 0):.2f} · "
                f"fragility {d.get('fragility', 0):.2f}"
            )
            out.append("")
            for line in uncertainty.rationale:
                out.append(f"- {line}")
            out.append("")

        if next_test:
            out.append("## What to measure next")
            out.append("")
            out.append("| # | Test | EIG (bits) | TAT | Utility |")
            out.append("|---|---|---|---|---|")
            for cand in next_test.data.get("ranked", [])[:6]:
                out.append(
                    f"| {cand['rank']} | {cand['name']} "
                    f"| {(cand.get('expected_information_gain_bits') or 0):.3f} "
                    f"| {cand['turnaround_hours']:.0f}h "
                    f"| {(cand.get('utility') or 0):.3f} |"
                )
            out.append("")

        significant = [
            f for f in flags if f.severity in (Severity.CRITICAL, Severity.HIGH)
        ]
        if significant:
            out.append("## Flags requiring attention")
            out.append("")
            for f in significant:
                out.append(f"- **[{f.severity.value.upper()}] {f.message}**")
                if f.detail:
                    out.append(f"  - {f.detail}")
            out.append("")

        out.append("## Provenance and limits")
        out.append("")
        out.append(
            "- Decision support only. This system does not prescribe, withhold "
            "or modify treatment."
        )
        out.append(
            "- Resistance reasoning operates at drug-class level on determinant "
            "calls made by external tools; it does not model resistance "
            "mechanisms."
        )
        out.append(
            "- All outputs require review by a qualified infectious-disease "
            "clinician or clinical microbiologist."
        )
        if case.provenance.citation:
            out.append(f"- Record source: {case.provenance.citation.short()}")
        out.append("")

        return "\n".join(out)

    def _ascii(
        self,
        case: CaseRecord,
        likelihood: Likelihood,
        outlook: SusceptibilityOutlook,
        anomaly: AnomalyStatus,
        confidence: Confidence,
        recommendation: str | None,
        escalation: list[str],
        review_required: bool,
    ) -> str:
        # The rule is 66 chars; keep every row inside it so the box stays
        # square in a terminal and in the dashboard's <pre>.
        label_w, value_w = 34, 64 - 34

        def row(label: str, value: str) -> list[str]:
            segments = self._wrap(value, value_w) or [""]
            out = [f"  {label:<{label_w}}{segments[0]}"]
            for seg in segments[1:]:
                out.append(f"  {'':<{label_w}}{seg}")
            return out

        lines = [RULE, "  PLAGUESHIELD :: ASSESSMENT REPORT", RULE]
        lines += row("CASE", case.case_id)
        lines += row("", case.label)
        lines += row("ORIGIN", case.provenance.origin.value.upper())
        lines.append(THIN)
        lines += row("PLAGUE LIKELIHOOD", likelihood.value.upper())
        lines += row("STANDARD-TREATMENT SUSCEPTIBILITY", outlook.value.upper())
        lines += row("RESISTANCE ANOMALY", anomaly.value.upper())
        lines += row("CONFIDENCE", confidence.value.upper())
        lines += row("MOST VALUABLE NEXT RESULT", (recommendation or "NONE").upper())
        lines.append(THIN)
        lines.append("  ESCALATION")
        for line in escalation:
            wrapped = self._wrap(line, 62)
            for i, seg in enumerate(wrapped):
                lines.append(f"    {'>' if i == 0 else ' '} {seg}")
        lines.append(THIN)
        lines.append(
            "  HUMAN REVIEW: " + ("REQUIRED" if review_required else "ROUTINE")
        )
        lines.append(RULE)
        lines.append("  Decision support only. Does not prescribe treatment.")
        lines.append(RULE)
        return "\n".join(lines)

    @staticmethod
    def _wrap(text: str, width: int) -> list[str]:
        words = text.split()
        out: list[str] = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 > width:
                out.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            out.append(current)
        return out or [""]
