import json

import httpx
import pytest

from plagueshield import llm


def test_responses_request_and_text_extraction(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client_class = httpx.Client

    def respond(request):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-5.5"
        assert payload["store"] is False
        assert json.loads(payload["input"])["case"]["case_id"] == "example"
        return httpx.Response(200, json={
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "Checking evidence gaps."},
                ]},
            ],
        })

    monkeypatch.setattr(llm.httpx, "Client", lambda **kwargs: client_class(
        transport=httpx.MockTransport(respond), **kwargs,
    ))
    assert llm.analyze_with_openai(
        case_payload={"case_id": "example"}, verdicts={}, model="gpt-5.5",
    ) == "Checking evidence gaps."


def test_missing_openai_key_does_not_use_anthropic_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    assert not llm.is_configured()
    with pytest.raises(llm.LLMUnavailable, match="OPENAI_API_KEY"):
        llm.analyze_with_openai(case_payload={}, verdicts={})


@pytest.mark.parametrize("status,body", [
    (401, {}),
    (200, {"status": "incomplete", "output": []}),
    (200, {"status": "completed", "output": []}),
])
def test_failed_or_empty_responses_are_unavailable(monkeypatch, status, body):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client_class = httpx.Client
    monkeypatch.setattr(llm.httpx, "Client", lambda **kwargs: client_class(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body)),
        **kwargs,
    ))
    with pytest.raises(llm.LLMUnavailable):
        llm.analyze_with_openai(case_payload={}, verdicts={})


def test_quota_exhaustion_is_distinguished_without_retry_or_raw_error(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    client_class = httpx.Client
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(429, json={'error': {'code': 'credit_balance_exhausted', 'type': 'insufficient_quota', 'message': 'private-provider-message'}})
    monkeypatch.setattr(llm.httpx, 'Client', lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs))
    with pytest.raises(llm.LLMUnavailable, match='credits or spending quota exhausted') as error:
        llm.analyze_with_openai(case_payload={}, verdicts={})
    assert len(calls) == 1
    assert 'private-provider-message' not in str(error.value)
