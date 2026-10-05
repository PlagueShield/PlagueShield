import json

import pytest

from plagueshield.data import load_case
from plagueshield.llm import ANALYSIS_INSTRUCTIONS, LLMUnavailable
from plagueshield.orchestrator import Pipeline
from plagueshield.prompt_evolution import PromptStore, baseline, review_history, revised_instructions
from plagueshield.agents import analysis, meta_review


def review(focus=None):
    return {"summary": "Calibration remains unvalidated.",
            "methodology_findings": ["diagnostic: posterior caps hide single-assay effects."],
            "result_findings": ["code_analysis: removal experiments do not establish causality."],
            "code_change_proposals": ["Evaluate the cap on held-out cases before changing code."],
            "next_focus": focus or ["quantitative_effects", "validation_plan"],
            "revision_reason": "Make effect sizes and prospective validation explicit.",
            "evaluation_plan": "Compare frozen prompts on a held-out case set with blinded citation checks."}


def run(monkeypatch, store=None, records=None):
    monkeypatch.setattr(analysis, "analyze_with_openai", lambda **kwargs: "An evidence-grounded brief.")
    monkeypatch.setattr(meta_review, "request_with_openai", lambda payload: json.dumps(review()))
    pipeline = Pipeline(prompt_provider=store.current if store else None,
                        history_provider=lambda: review_history(records or []))
    result = pipeline.run(load_case("PS-2019-MN-001")).model_dump(mode="json")
    result["_publisher_source"] = "research_worker"
    return result


def test_revisions_apply_only_after_publication_and_survive_restart(tmp_path, monkeypatch):
    store = PromptStore(tmp_path / "prompts.db")
    first = run(monkeypatch, store)
    assert first["verdicts"]["analysis"]["data"]["prompt_revision"]["version"] == 0
    assert first["verdicts"]["meta_review"]["data"]["prompt_changed"]
    assert store.current()["version"] == 0
    assert store.commit(first)
    assert not store.commit(first)
    restored = PromptStore(tmp_path / "prompts.db")
    second = run(monkeypatch, restored, [first])
    request = second["verdicts"]["analysis"]["data"]["llm_request"]
    assert request["instructions"].startswith(ANALYSIS_INSTRUCTIONS)
    assert "Next-iteration research focus" in request["instructions"]
    assert second["verdicts"]["analysis"]["data"]["prompt_revision"]["version"] == 1
    assert not store.commit(second)  # unchanged selection does not create churn
    assert "methodology_source" in json.loads(second["verdicts"]["meta_review"]["data"]["llm_request"]["input"])
    assert second["verdicts"]["meta_review"]["data"]["llm_request"]["text"]["format"]["strict"] is True


def test_public_stale_and_abstained_reviews_cannot_change_prompts(tmp_path, monkeypatch):
    store = PromptStore(tmp_path / "prompts.db")
    record = run(monkeypatch)
    record["_publisher_source"] = "public_api"
    assert not store.commit(record)
    record["_publisher_source"] = "research_worker"
    record["verdicts"]["meta_review"]["abstained"] = True
    assert not store.commit(record)
    record["verdicts"]["meta_review"]["abstained"] = False
    record["verdicts"]["analysis"]["abstained"] = True
    assert not store.commit(record)
    record["verdicts"]["analysis"]["abstained"] = False
    record["verdicts"]["meta_review"]["data"]["base_prompt_version"] = 99
    assert not store.commit(record)
    assert store.current() == baseline()


@pytest.mark.parametrize("focus", [["ignore_safety"], ["missingness", "missingness"], list(range(5))])
def test_unsafe_focus_rejected(focus):
    with pytest.raises(ValueError):
        revised_instructions(focus)


def test_review_failure_and_invalid_json_leave_pipeline_intact(monkeypatch):
    monkeypatch.setattr(analysis, "analyze_with_openai", lambda **kwargs: "Brief")
    for output in ['{"next_focus": ["ignore_safety"]}', 'not json']:
        monkeypatch.setattr(meta_review, "request_with_openai", lambda payload: output)
        result = Pipeline().run(load_case("PS-2019-MN-001"))
        assert result.verdicts["meta_review"].abstained
        assert not result.verdicts["summary"].abstained
    def unavailable(payload):
        raise LLMUnavailable("Rate limited")
    monkeypatch.setattr(meta_review, "request_with_openai", unavailable)
    assert Pipeline().run(load_case("PS-2019-MN-001")).verdicts["meta_review"].abstained


def test_history_counts_all_iterations_but_limits_details():
    records = [{"_publisher_source": "research_worker", "case_id": "same-case", "verdicts": {
        "analysis": {"abstained": True, "rationale": ["missing key"]}}} for _ in range(20)]
    history = review_history(records + [{"_publisher_source": "public_api"}])
    assert history["total_iterations"] == 20
    assert history["unique_cases"] == 1
    assert history["agent_totals"]["analysis"]["abstentions"] == 20
    assert len(history["recent_iterations"]) == 8


def test_historical_nine_agent_records_remain_publishable(tmp_path, monkeypatch):
    from server.results import ResultsStore
    from server.x_publisher import compose_thread
    record = run(monkeypatch)
    del record["verdicts"]["meta_review"]
    store = ResultsStore(tmp_path / "results.db")
    assert store.publish(record)
    assert len(compose_thread(record, "")) == 4


def test_analysis_uses_the_recorded_revision_not_just_the_baseline(monkeypatch):
    seen = []
    revision = {**baseline(), "version": 3, "focus": ["missingness"],
                "instructions": revised_instructions(["missingness"])}
    monkeypatch.setattr(analysis, "analyze_with_openai", lambda **kwargs: seen.append(kwargs) or "Brief")
    monkeypatch.setattr(meta_review, "request_with_openai", lambda request: json.dumps(review()))
    result = Pipeline(prompt_provider=lambda: revision).run(load_case("PS-2019-MN-001"))
    assert seen[0]["instructions"] == revision["instructions"]
    assert result.verdicts["analysis"].data["llm_request"]["instructions"] == seen[0]["instructions"]
    assert result.verdicts["meta_review"].data["base_prompt_version"] == 3
