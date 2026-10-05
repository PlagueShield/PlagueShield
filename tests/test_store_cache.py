import json

from server import app as server


def test_history_cache_reuses_parse_and_refreshes_on_append(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'STORE_PATH', tmp_path / 'assessments.ndjson')
    monkeypatch.setattr(server, '_records_signature', None)
    monkeypatch.setattr(server, '_records_cache', [])
    server._append({'case_id': 'first'})
    assert server._read_all() == [{'case_id': 'first'}]
    original_loads = json.loads
    calls = []
    def tracked_loads(value):
        calls.append(value)
        return original_loads(value)
    monkeypatch.setattr(server.json, 'loads', tracked_loads)
    returned = server._read_all()
    returned.clear()
    assert server._read_all() == [{'case_id': 'first'}]
    assert not calls
    server._append({'case_id': 'second'})
    assert len(server._read_all()) == 2
    assert len(calls) == 2
