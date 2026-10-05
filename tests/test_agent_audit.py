from fastapi.testclient import TestClient

from plagueshield.data import load_case
from plagueshield.orchestrator import Pipeline
from server import app as server


def test_code_analysis_is_reproducible_and_does_not_mutate_case(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    case = load_case('PS-2019-MN-001')
    original = case.model_dump()
    result = Pipeline().run(case)
    audit = result.verdicts['code_analysis']
    assert not audit.abstained
    assert audit.data['ablations']
    assert audit.data['ablations'][-1]['removed_family'] == 'all_recorded_assays'
    assert audit.data['ablations'][-1]['removed_results'] == len(case.diagnostics)
    assert len(audit.data['posterior_odds_sweep']) == 5
    assert case.model_dump() == original
    assert audit.data['clinical_decisions_changed'] is False
    assert audit.data['execution']['duration_ms'] >= 0
    assert 'diagnostic' in audit.data['execution']['upstream']
    request = result.verdicts['analysis'].data['llm_request']
    assert '120 words' not in request['instructions']
    assert 'code_analysis' in request['input']
    assert 'Authorization' not in request


def test_agent_runs_include_full_results_and_sources(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assessment = Pipeline().run(load_case('PS-2019-MN-001')).model_dump(mode='json')
    monkeypatch.setattr(server, '_read_all', lambda: [assessment])
    client = TestClient(server.app)
    response = client.get('/api/agents/code_analysis/runs')
    assert response.status_code == 200
    payload = response.json()
    assert 'class CodeAnalysisAgent' in payload['implementation']
    assert payload['runs'][0]['verdict']['data']['ablations']
    assert payload['runs'][0]['references']
    assert payload['runs'][0]['source_note']
    assert client.get('/api/agents/not_an_agent/runs').status_code == 404


def test_activity_uses_real_headline_not_placeholder():
    verdict = {'headline': '3 guidance sources retrieved', 'data': {}, 'rationale': []}
    assert server._working_on('evidence', verdict, {}) == verdict['headline']
