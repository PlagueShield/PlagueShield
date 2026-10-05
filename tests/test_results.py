import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from plagueshield.data import load_case
from plagueshield.orchestrator import Pipeline
from server import app as server
from server.results import ResultsStore


@pytest.fixture
def record(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assessment = Pipeline().run(load_case('PS-2019-MN-001')).model_dump(mode='json')
    assessment['_publisher_source'] = 'research_worker'
    return assessment


def test_article_has_real_experiments_and_reproducibility(tmp_path, record):
    store = ResultsStore(tmp_path / 'articles.db')
    article_id = store.publish(record)
    article = store.get(article_id)
    sections = {section['id']: section for section in article['sections']}
    assert {'abstract', 'setup', 'methods', 'findings', 'experiments', 'discussion', 'limits', 'references', 'reproducibility'} <= set(sections)
    actual = record['verdicts']['code_analysis']['data']['ablations']
    assert len(sections['experiments']['table']['rows']) == len(actual)
    assert sections['experiments']['table']['rows'][-1][2] == round(actual[-1]['probability'], 4)
    assert sections['references']['references']
    assert 'not peer reviewed' in ' '.join(sections['question']['paragraphs'])
    assert article['artifact'] == record
    checksum = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert article['source_sha256'] == checksum
    assert record['verdicts']['analysis']['abstain_reason'] in sections['discussion']['paragraphs']


def test_republishing_is_idempotent_and_iterations_are_distinct(tmp_path, record):
    store = ResultsStore(tmp_path / 'articles.db')
    first_id = store.publish(record)
    original = store.get(first_id)
    assert store.publish(record) == first_id
    assert store.get(first_id) == original
    record['assessed_at'] = '2026-10-05T23:59:00Z'
    assert store.publish(record) != first_id
    assert store.list()['count'] == 2


def test_search_provenance_and_pagination(tmp_path, record):
    store = ResultsStore(tmp_path / 'articles.db')
    store.publish(record)
    record['assessed_at'] = '2026-10-05T23:59:00Z'
    store.publish(record)
    assert store.list(limit=1)['next_offset'] == 1
    assert len(store.list(limit=1, offset=1)['articles']) == 1
    assert store.list(query='PS-2019-MN')['count'] == 2
    assert store.list(query='unknown')['count'] == 0
    assert store.list(origin='synthetic')['count'] == 0
    assert store.list(origin='published_report')['count'] == 2
    assert store.list(query="' OR 1=1 --")['count'] == 0


def test_publicly_submitted_and_incomplete_records_are_not_papers(tmp_path, record):
    store = ResultsStore(tmp_path / 'articles.db')
    record.pop('_publisher_source')
    assert store.publish(record) is None
    record['_publisher_source'] = 'research_worker'
    record['verdicts'].pop('code_analysis')
    assert store.publish(record) is None
    assert store.list()['count'] == 0


def test_results_routes_and_download(tmp_path, record, monkeypatch):
    store = ResultsStore(tmp_path / 'articles.db')
    article_id = store.publish(record)
    monkeypatch.setattr(server, '_results', store)
    client = TestClient(server.app)
    assert client.get('/api/results').json()['count'] == 1
    response = client.get(f'/api/results/{article_id}')
    assert response.status_code == 200
    assert 'artifact' not in response.json()
    download = client.get(f'/api/results/{article_id}/artifact')
    assert download.json() == record
    assert 'attachment' in download.headers['content-disposition']
    assert client.get('/api/results/missing').status_code == 404
    assert client.get('/api/results/missing/artifact').status_code == 404
    assert client.get('/api/results?limit=0').status_code == 422
    assert client.get('/results').status_code == 200
    assert client.get(f'/results/{article_id}').status_code == 200


def test_background_publication_creates_article(tmp_path, record, monkeypatch):
    store = ResultsStore(tmp_path / 'articles.db')
    monkeypatch.setattr(server, '_results', store)
    monkeypatch.setattr(server, '_append', lambda record: None)
    from plagueshield.models import CaseAssessment
    server._publish_background(CaseAssessment.model_validate(record))
    assert store.list()['count'] == 1
