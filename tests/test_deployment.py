from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.container_start import initialize_data


@pytest.mark.parametrize("path", ["/api/run/PS-2019-MN-001", "/api/run-all", "/api/assessments"])
def test_public_deployment_rejects_writes_before_execution(monkeypatch, path):
    from server import app as api
    monkeypatch.setenv("PLAGUESHIELD_PUBLIC_READ_ONLY", "true")
    monkeypatch.setattr(api._pipeline, "run", lambda *args: pytest.fail("Public visitor triggered LLM spend"))
    response = TestClient(api.app).post(path, json={})
    assert response.status_code == 403
    assert "read-only" in response.json()["detail"]


def test_initial_volume_seed_never_overwrites_existing_results(tmp_path: Path):
    seed, data = tmp_path / "seed", tmp_path / "data"
    seed.mkdir()
    (seed / "assessments.ndjson").write_text("initial\n")
    initialize_data(data, seed)
    assert (data / "assessments.ndjson").read_text() == "initial\n"
    (data / "assessments.ndjson").write_text("new research\n")
    (seed / "assessments.ndjson").write_text("older snapshot\n")
    initialize_data(data, seed)
    assert (data / "assessments.ndjson").read_text() == "new research\n"
    assert not list(data.glob("*.initializing"))
