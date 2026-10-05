from fastapi import Response

from server import app as server


def test_market_cap_cache_refreshes_and_is_shared(monkeypatch):
    clock = [0.0]
    calls = []
    monkeypatch.setattr(server, '_token_status_cache', {})
    monkeypatch.setattr(server, '_token_status_expires', 0.0)
    monkeypatch.setattr(server.time, 'monotonic', lambda: clock[0])

    def fetch():
        calls.append(True)
        return {'market_cap': len(calls) * 1000, 'market_cap_source': 'dexscreener'}

    monkeypatch.setattr(server, '_fetch_token_status', fetch)
    response = Response()
    assert server.token_status(response)['market_cap'] == 1000
    assert response.headers['cache-control'] == 'no-store'
    clock[0] = 3
    assert server.token_status(Response())['market_cap'] == 1000
    assert len(calls) == 1
    clock[0] = 5
    assert server.token_status(Response())['market_cap'] == 2000
    assert len(calls) == 2


def test_temporary_provider_failure_preserves_last_value(monkeypatch):
    monkeypatch.setattr(server, '_token_status_cache', {'address': 'verified-token', 'market_cap': 12345, 'market_cap_source': 'dexscreener', 'refreshed_at': 'previous'})
    monkeypatch.setattr(server, '_token_status_expires', 0.0)
    monkeypatch.setattr(server, '_fetch_token_status', lambda: {'address': 'verified-token', 'market_cap': None, 'market_cap_source': None})
    status = server.token_status(Response())
    assert status['market_cap'] == 12345
    assert status['refreshed_at'] == 'previous'
    assert status['stale']


def test_unconfigured_token_has_no_market_cap_and_never_contacts_provider(monkeypatch):
    monkeypatch.setattr(server, 'TOKEN_ADDRESS', '')
    monkeypatch.setattr(server, 'TOKEN_MARKET_CAP', '999999')
    import httpx
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: (_ for _ in ()).throw(AssertionError('Token provider called without an address')))
    status = server._fetch_token_status()
    assert status['address'] is None
    assert status['market_cap'] is None
    assert status['market_cap_source'] is None
    assert not status['configured']


def test_removing_or_changing_address_does_not_resurface_old_market_cap(monkeypatch):
    for address in (None, 'replacement-token'):
        monkeypatch.setattr(server, '_token_status_cache', {'address': 'incorrect-token', 'market_cap': 12345, 'market_cap_source': 'dexscreener'})
        monkeypatch.setattr(server, '_token_status_expires', 0.0)
        monkeypatch.setattr(server, '_fetch_token_status', lambda: {'address': address, 'market_cap': None, 'market_cap_source': None})
        assert server.token_status(Response())['market_cap'] is None
