"""Immutable research articles assembled from recorded backend experiments."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from plagueshield.data import load_case
from plagueshield.knowledge.case_sources import case_reference_info
from plagueshield.models import CaseRecord
from server.worker import AGENT_NAMES, CORE_AGENT_NAMES

ARTICLE_VERSION = '1.1'


def number(value, digits=4):
    return round(value, digits) if isinstance(value, (int, float)) else 'Not recorded'


def build_article(record: dict) -> dict:
    verdicts = record['verdicts']
    recorded_case = verdicts['diagnostic'].get('data', {}).get('execution', {}).get('case_input')
    case = CaseRecord.model_validate(recorded_case) if recorded_case else load_case(record['case_id'])
    references = case_reference_info(case)
    canonical = json.dumps(record, sort_keys=True, separators=(',', ':'))
    checksum = hashlib.sha256(canonical.encode()).hexdigest()
    diagnostic = verdicts['diagnostic']
    computational = verdicts['code_analysis']['data']
    next_test = verdicts['next_test']['data']
    analysis = verdicts['analysis']
    probability = record.get('plague_probability')
    abstract = (
        f"This iteration evaluates {case.label} using {len([name for name in AGENT_NAMES if name in verdicts])} coordinated research agents. "
        f"The diagnostic model reported {number(probability)} probability and "
        f"{record.get('confidence', 'unreported')} assessment confidence. "
        f"The Python audit executed {len(computational.get('ablations', []))} evidence-removal experiments; "
        f"the largest absolute model change was {number(computational.get('max_change_percentage_points'))} percentage points. "
        "These are computational findings for a research vignette, not observed clinical outcomes."
    )
    executions = [
        {'agent': name, 'status': 'failed' if any(flag.get('code') == 'AGENT_FAILURE' for flag in verdicts[name].get('flags', [])) else 'abstained' if verdicts[name].get('abstained') else 'completed',
         'duration_ms': verdicts[name].get('data', {}).get('execution', {}).get('duration_ms'),
         'confidence': verdicts[name].get('confidence')}
        for name in AGENT_NAMES if name in verdicts
    ]
    sources = []
    seen = set()
    for citation in [*references['references'], *record.get('citations', [])]:
        identifier = citation.get('identifier') or ''
        url = citation.get('url') or (f"https://pubmed.ncbi.nlm.nih.gov/{identifier[5:]}/" if identifier.startswith('PMID:') else None)
        key = (citation.get('title'), url)
        if key not in seen:
            sources.append({**citation, 'url': url})
            seen.add(key)
    setup = {
        'case_id': case.case_id, 'origin': case.provenance.origin.value,
        'country': case.country, 'assessed_at': record.get('assessed_at'),
        'pipeline_version': record.get('pipeline_version'),
        'analysis_model': analysis.get('data', {}).get('model'),
        'analysis_prompt_version': analysis.get('data', {}).get('prompt_revision', {}).get('version'),
        'live_evidence_requested': verdicts['evidence'].get('data', {}).get('live_refresh'),
        'recorded_assay_results': len(case.diagnostics),
        'recorded_genomic_evidence': case.genomic is not None,
    }
    sections = [
        {'id': 'abstract', 'title': 'Abstract', 'paragraphs': [abstract]},
        {'id': 'question', 'title': 'Research Question & Provenance', 'paragraphs': [
            'How strongly does the recorded evidence support the diagnostic hypothesis, where are the uncertainty and discordance, and how sensitive are model outputs to evidence removal?',
            references['source_note'],
            'This article is automatically assembled from one completed iteration. It is not peer reviewed and does not establish clinical validity or a new biological discovery.',
        ]},
        {'id': 'setup', 'title': 'Experimental Setup', 'paragraphs': [
            f"Run {record.get('assessed_at')} used pipeline {record.get('pipeline_version')}. "
            f"The record contains {len(case.diagnostics)} diagnostic assay results. "
            'Independent diagnostic and resistance agents run first, followed by evidence retrieval and discordance checks, uncertainty assessment, next-test ranking, Python experiments, LLM interpretation, and report assembly. New iterations end with a GPT-5.5 methodology meta-review that proposes next-iteration research focus; older snapshots may omit this stage.',
            'Most agents are deterministic Python computations. GPT-5.5 supplies a research interpretation of their outputs; it does not replace their structured decisions. Failed or abstaining agents remain visible as missing contributions.',
        ], 'data': setup, 'table': {'columns': ['Agent', 'State', 'Duration (ms)', 'Confidence'],
                                 'rows': [[row['agent'], row['status'], number(row['duration_ms'], 2), row['confidence']] for row in executions]}},
        {'id': 'methods', 'title': 'Methods & Assumptions', 'paragraphs': [
            'Diagnostic scoring combines pre-test odds with recorded epidemiological, clinical, and laboratory evidence. Correlated results are damped and posterior probabilities are capped. Surveillance classification is assigned independently of the probability estimate.',
            'The code-analysis agent re-executes the diagnostic model with each recorded assay family removed and with all recorded assays removed. Clinical and genomic evidence remain fixed. It reports the signed probability change relative to the original result.',
            'The posterior-odds sweep scales the already-computed diagnostic odds by 0.25, 0.5, 1, 2, and 4. These are illustrative stress assumptions, not a prior refit, measured uncertainty, or confidence intervals.',
            'Next-test rankings use model-based expected information gain, turnaround, actionability, and feasibility assumptions. They are research rankings, not instructions to order tests or change treatment.',
        ]},
        {'id': 'findings', 'title': 'Agent Findings', 'agents': [
            {'name': name, 'headline': verdicts[name].get('headline'),
             'rationale': verdicts[name].get('rationale', []),
             'abstained': verdicts[name].get('abstained', False),
             'abstain_reason': verdicts[name].get('abstain_reason'),
             'data': {key: value for key, value in verdicts[name].get('data', {}).items()
                      if key not in {'execution', 'llm_request', 'traceback', 'report_markdown', 'report_ascii', 'synthesis'}}}
            for name in AGENT_NAMES if name != 'analysis' and name in verdicts
        ]},
        {'id': 'experiments', 'title': 'Evidence-Removal Experiments', 'paragraphs': [
            f"Baseline diagnostic probability: {number(diagnostic.get('score'))}. "
            f"Binary entropy: {number(computational.get('entropy_bits'))} bits. "
            f"Maximum absolute probability change: {number(computational.get('max_change_percentage_points'))} percentage points.",
            'A small single-family change can reflect redundant evidence or probability caps; it is not proof that the removed evidence is unimportant.',
        ], 'table': {'columns': ['Removed evidence', 'Results removed', 'Probability', 'Change (pp)', 'Classification'],
                    'rows': [[row.get('removed_family'), row.get('removed_results'), number(row.get('probability')),
                              number(row.get('change_percentage_points')), row.get('classification')]
                             for row in computational.get('ablations', [])]}},
        {'id': 'sensitivity', 'title': 'Posterior-Odds Sensitivity',
         'table': {'columns': ['Odds multiplier', 'Probability'],
                   'rows': [[row.get('odds_multiplier'), number(row.get('probability'))] for row in computational.get('posterior_odds_sweep', [])]}},
        {'id': 'rankings', 'title': 'Information-Gain Results', 'paragraphs': [next_test.get('recommendation') or 'No next-test ranking was recorded.'],
         'table': {'columns': ['Rank', 'Measurement', 'EIG (bits)', 'Turnaround (h)', 'Utility'],
                   'rows': [[row.get('rank'), row.get('name'), number(row.get('expected_information_gain_bits')),
                             number(row.get('turnaround_hours')), number(row.get('utility'))] for row in next_test.get('ranked', [])]}},
        {'id': 'discussion', 'title': 'Discussion / GPT-5.5 Interpretation', 'paragraphs': [
            analysis.get('data', {}).get('synthesis') or analysis.get('abstain_reason') or 'No LLM interpretation was recorded.',
            'This interpretation is model-generated. Numerical claims should be checked against the recorded experiment tables and source artifacts.',
        ]},
        {'id': 'limits', 'title': 'Limitations & Review', 'paragraphs': [
            'The case is a synthetic or reconstructed vignette, not a prospectively collected patient cohort. Model probabilities and sensitivity experiments have not been validated as clinical performance estimates.',
            'The experiments do not establish causality. Shared data, heuristic likelihood ratios, omitted variables, evidence correlation, and model caps limit inference. The odds sweep does not quantify real-world uncertainty.',
            'Official sources provide guidance or aggregate event context; they do not independently verify every detail of a reconstructed case. No treatment decisions should be made from this article.',
            *[f"{flag.get('severity')} / {flag.get('code')}: {flag.get('message')}. {flag.get('detail') or ''}" for flag in record.get('flags', [])],
        ]},
        {'id': 'references', 'title': 'References', 'references': sources},
        {'id': 'reproducibility', 'title': 'Reproducibility Appendix', 'paragraphs': [
            'The attached JSON snapshot contains the case input, every recorded upstream verdict, execution timing, full structured outputs, and the exact LLM request where available. Missing historical traces are not reconstructed.',
            f"Source snapshot SHA-256: {checksum}. Article template version: {ARTICLE_VERSION}. This article preserves one iteration; subsequent iterations receive separate permanent URLs.",
        ]},
    ]
    if 'meta_review' in verdicts:
        reviewer = verdicts['meta_review']
        review_data = reviewer.get('data', {})
        proposal = review_data.get('review', {})
        sections.insert(-3, {
            'id': 'meta-review', 'title': 'Methodology Meta-Review & Next Iteration',
            'paragraphs': [reviewer.get('abstain_reason') or proposal.get('summary', 'No review recorded.'),
                           *proposal.get('methodology_findings', []), *proposal.get('result_findings', []),
                           'Proposed prompt revision: ' + proposal.get('revision_reason', 'None.'),
                           'Proposed evaluation (not executed): ' + proposal.get('evaluation_plan', 'None.'),
                           'Revisions affect subsequent iterations only. Safety instructions and deterministic methods remain unchanged; no performance improvement has been demonstrated.'],
            'data': {'base_prompt_version': review_data.get('base_prompt_version'),
                     'next_focus': proposal.get('next_focus', []),
                     'proposed_instructions': review_data.get('proposed_instructions'),
                     'code_change_proposals': proposal.get('code_change_proposals', [])},
        })
    return {'id': checksum[:24], 'title': f"Evidence sensitivity and uncertainty: {case.label}",
            'case_id': case.case_id, 'origin': case.provenance.origin.value,
            'run_at': record.get('assessed_at'), 'published_at': datetime.now(timezone.utc).isoformat(),
            'abstract': abstract, 'pipeline_version': record.get('pipeline_version'),
            'template_version': ARTICLE_VERSION, 'source_sha256': checksum,
            'sections': sections, 'artifact': record}


class ResultsStore:
    def __init__(self, path: Path):
        self.path = path
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS articles (id TEXT PRIMARY KEY, published_at TEXT, run_at TEXT, case_id TEXT, origin TEXT, title TEXT, abstract TEXT, payload TEXT)')

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    def publish(self, record: dict) -> str | None:
        if record.get('_publisher_source') != 'research_worker' or not all(name in record.get('verdicts', {}) for name in CORE_AGENT_NAMES):
            return None
        article = build_article(record)
        with self._db() as db:
            db.execute('INSERT OR IGNORE INTO articles VALUES (?,?,?,?,?,?,?,?)',
                       (article['id'], article['published_at'], article['run_at'], article['case_id'], article['origin'], article['title'], article['abstract'], json.dumps(article)))
        return article['id']

    def list(self, *, limit=24, offset=0, query='', origin='') -> dict:
        clause = "WHERE instr(lower(title || ' ' || case_id), lower(?)) > 0 AND (? = '' OR origin = ?)"
        parameters = (query, origin, origin)
        with self._db() as db:
            count = db.execute('SELECT count(*) FROM articles ' + clause, parameters).fetchone()[0]
            rows = db.execute('SELECT id, published_at, run_at, case_id, origin, title, abstract FROM articles ' + clause + ' ORDER BY run_at DESC, id DESC LIMIT ? OFFSET ?', (*parameters, limit, offset)).fetchall()
        keys = ('id', 'published_at', 'run_at', 'case_id', 'origin', 'title', 'abstract')
        return {'count': count, 'articles': [dict(zip(keys, row)) for row in rows], 'next_offset': offset + limit if offset + limit < count else None}

    def get(self, article_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute('SELECT payload FROM articles WHERE id=?', (article_id,)).fetchone()
        return json.loads(row[0]) if row else None
