import json

import httpx
import pytest
from authlib.integrations.httpx_client import OAuth1Client

from plagueshield.data import load_case
from plagueshield.orchestrator import Pipeline
from server.x_publisher import XPublisher, compose_thread


@pytest.fixture
def record(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    result = Pipeline().run(load_case('PS-2019-MN-001')).model_dump(mode='json')
    result['_publisher_source'] = 'research_worker'
    for name in ('analysis', 'meta_review'):
        result['verdicts'][name].update(abstained=False, abstain_reason=None, flags=[])
    result['verdicts']['analysis']['data']['synthesis'] = 'Evidence gaps prevent validation of the model estimates.'
    result['verdicts']['meta_review']['data']['review'] = {'summary': 'Independent validation is needed.', 'methodology_findings': ['Probability caps can mask evidence sensitivity.']}
    return result


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv('PLAGUESHIELD_X_ENABLED', 'true')
    monkeypatch.setenv('PLAGUESHIELD_X_DRY_RUN', 'false')
    monkeypatch.setenv('PLAGUESHIELD_X_ACCOUNT', 'plagueshield')
    monkeypatch.setenv('PLAGUESHIELD_PUBLIC_URL', 'https://plagueshield.example')
    for key in ['X_API_KEY', 'X_API_KEY_SECRET', 'X_ACCESS_TOKEN', 'X_ACCESS_TOKEN_SECRET']:
        monkeypatch.setenv(key, 'test-secret')


def publisher(tmp_path, record, handler):
    return XPublisher(lambda: [record], tmp_path / 'outbox.db', client_factory=lambda: OAuth1Client(
        'test-key', client_secret='test-secret', token='test-token', token_secret='test-token-secret',
        force_include_body=True,
        transport=httpx.MockTransport(handler),
    ))


def test_thread_covers_every_agent_with_conservative_lengths(record):
    posts = compose_thread(record, 'https://plagueshield.example')
    assert len(posts) == 5
    assert all(len(post) <= 280 and post.isascii() for post in posts)
    assert 'Reconstructed' in posts[0]
    assert 'not clinical advice' in posts[0]
    assert '/cases' in posts[0]
    for agent in record['verdicts']:
        assert agent.replace('_', '-') + ':' in '\n'.join(posts)


def test_oauth_thread_posting_and_restart_dedup(tmp_path, record, settings):
    requests = []

    def handler(request):
        assert request.headers['authorization'].startswith('OAuth ')
        if request.method == 'GET':
            return httpx.Response(200, json={'data': {'username': 'plagueshield'}})
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) > 1:
            assert payload['reply']['in_reply_to_tweet_id'] == str(len(requests) - 1)
        return httpx.Response(201, json={'data': {'id': str(len(requests))}})

    agent = publisher(tmp_path, record, handler)
    agent.tick()
    assert len(requests) == 5, agent.snapshot()['error']
    assert agent.snapshot()['status'] == 'published'
    restarted = publisher(tmp_path, record, handler)
    restarted.tick()
    assert len(requests) == 5
    assert restarted.snapshot()['status'] == 'no_new_findings'
    assert 'test-secret' not in json.dumps(restarted.snapshot())


def test_dry_run_never_sends(tmp_path, record, settings, monkeypatch):
    monkeypatch.setenv('PLAGUESHIELD_X_DRY_RUN', 'true')
    agent = publisher(tmp_path, record, lambda request: pytest.fail('Unexpected network request'))
    agent.tick()
    assert agent.snapshot()['status'] == 'previewed'
    assert not agent.snapshot()['history'][0]['post_urls']


def test_wrong_account_blocks_before_posting(tmp_path, record, settings):
    def handler(request):
        assert request.method == 'GET'
        return httpx.Response(200, json={'data': {'username': 'wrong_account'}})
    agent = publisher(tmp_path, record, handler)
    agent.tick()
    assert agent.snapshot()['status'] == 'blocked'


def test_timeout_is_not_replayed(tmp_path, record, settings):
    calls = []
    def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'data': {'username': 'plagueshield'}})
        calls.append(request)
        raise httpx.ReadTimeout('Unconfirmed write', request=request)
    agent = publisher(tmp_path, record, handler)
    agent.tick()
    assert agent.snapshot()['status'] == 'uncertain'
    publisher(tmp_path, record, handler).tick()
    assert len(calls) == 1


def test_partial_thread_rate_limit_keeps_ids_and_waits(tmp_path, record, settings):
    calls = []
    def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'data': {'username': 'plagueshield'}})
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(201, json={'data': {'id': '123'}})
        return httpx.Response(429, headers={'retry-after': '600'})
    agent = publisher(tmp_path, record, handler)
    agent.tick()
    assert len(agent.snapshot()['history'][0]['post_urls']) == 1, agent.snapshot()['error']
    agent.tick()
    assert len(calls) == 2
    assert agent.snapshot()['status'] == 'rate_limited'


def test_disabled_and_missing_credentials_never_send(tmp_path, record, settings, monkeypatch):
    monkeypatch.setenv('PLAGUESHIELD_X_ENABLED', 'false')
    agent = publisher(tmp_path, record, lambda request: pytest.fail('Unexpected request'))
    agent.tick()
    assert agent.snapshot()['status'] == 'disabled'
    monkeypatch.setenv('PLAGUESHIELD_X_ENABLED', 'true')
    monkeypatch.delenv('X_ACCESS_TOKEN', raising=False)
    agent = publisher(tmp_path, record, lambda request: pytest.fail('Unexpected request'))
    agent.tick()
    assert agent.snapshot()['status'] == 'configuration_required'


def test_publicly_submitted_records_are_not_published(tmp_path, record, settings):
    record.pop('_publisher_source')
    agent = publisher(tmp_path, record, lambda request: pytest.fail('Unexpected request'))
    agent.tick()
    assert agent.snapshot()['status'] == 'waiting_for_research'


@pytest.mark.parametrize('name', ['analysis', 'meta_review'])
def test_failed_llm_runs_never_publish(tmp_path, record, settings, name):
    record['verdicts'][name].update(abstained=True, abstain_reason='OpenAI API HTTP 429')
    agent = publisher(tmp_path, record, lambda request: pytest.fail('Unexpected request'))
    agent.tick()
    assert agent.snapshot()['status'] == 'waiting_for_research'
    assert agent.snapshot()['history'] == []


def test_summaries_show_findings_and_qualify_scores(record):
    text = '\n'.join(compose_thread(record, 'https://plagueshield.example'))
    assert 'not accuracy' in text
    assert 'Clinical summary prepared' not in text
    assert 'Probability caps' in text


def test_latest_failure_does_not_publish_stale_success(tmp_path, record, settings):
    import copy
    failed = copy.deepcopy(record)
    failed['assessed_at'] = '9999-01-01'
    failed['verdicts']['analysis']['abstained'] = True
    agent = XPublisher(lambda: [record, failed], tmp_path / 'outbox.db', client_factory=lambda: pytest.fail('Unexpected request'))
    agent.tick()
    assert agent.snapshot()['history'] == []
