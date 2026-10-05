from collections import defaultdict
from threading import Event

from plagueshield.data import load_case
from plagueshield.orchestrator import Pipeline
from server.worker import AGENT_NAMES, ResearchWorker


def test_progress_reports_every_agent_and_dependency_order(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    events = []
    result = Pipeline().run(load_case('PS-2019-MN-001'), on_progress=lambda name, status, verdict: events.append((name, status)))
    per_agent = defaultdict(list)
    for name, status in events:
        per_agent[name].append(status)
    assert set(per_agent) == set(AGENT_NAMES)
    for statuses in per_agent.values():
        assert statuses[0] == 'running'
        assert statuses[1] in {'completed', 'abstained', 'failed'}
    assert per_agent['analysis'] == ['running', 'abstained']
    assert events.index(('analysis', 'running')) > events.index(('next_test', 'completed'))
    assert events.index(('summary', 'running')) > events.index(('analysis', 'abstained'))
    assert result.report_ascii


def test_agent_failures_emit_terminal_progress(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    pipe = Pipeline()

    def fail(*args):
        raise RuntimeError('agent failed')

    monkeypatch.setattr(pipe.resistance, 'run', fail)
    events = []
    result = pipe.run(load_case('PS-2019-MN-001'), on_progress=lambda name, status, verdict: events.append((name, status)))
    assert ('resistance', 'failed') in events
    assert ('summary', 'completed') in events
    assert result.report_ascii


def test_progress_observer_failure_cannot_break_assessment(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    def fail(*args):
        raise RuntimeError('observer disconnected')

    assert Pipeline().run(load_case('PS-2019-MN-001'), on_progress=fail).report_ascii


def test_worker_exposes_running_steps_and_publishes(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    entered = Event()
    release = Event()
    published = Event()
    records = []
    pipe = Pipeline()
    real_run = pipe.analysis.run

    def blocking_run(case, context):
        entered.set()
        assert release.wait(3)
        return real_run(case, context)

    monkeypatch.setattr(pipe.analysis, 'run', blocking_run)

    def publish(assessment):
        records.append(assessment)
        published.set()

    worker = ResearchWorker(pipe, publish, interval=120)
    worker.start()
    try:
        assert entered.wait(3)
        snapshot = worker.snapshot()
        assert snapshot['alive']
        assert snapshot['status'] == 'running'
        steps = {s['agent']: s for s in snapshot['steps']}
        assert steps['analysis']['status'] == 'running'
        assert steps['summary']['status'] == 'queued'
        assert steps['next_test']['status'] == 'completed'
        snapshot['steps'][0]['status'] = 'tampered'
        assert worker.snapshot()['steps'][0]['status'] == 'completed'
        release.set()
        assert published.wait(3)
    finally:
        release.set()
        worker.stop()
    assert len(records) == 1
    final = worker.snapshot()
    assert final['status'] == 'stopped'
    assert not final['alive']
    assert final['completed_runs'] == 1
    assert final['next_run_at'] is None
    assert len(final['events']) == 2 * len(AGENT_NAMES)
