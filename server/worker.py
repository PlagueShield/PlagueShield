"""Background research runs and thread-safe, public progress snapshots."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from plagueshield.data import available_cases, load_case
from plagueshield.models import CaseAssessment, Verdict
from plagueshield.orchestrator import Pipeline

logger = logging.getLogger(__name__)
CORE_AGENT_NAMES = ("diagnostic", "resistance", "evidence", "discordance", "uncertainty", "next_test", "code_analysis", "analysis", "summary")
AGENT_NAMES = (*CORE_AGENT_NAMES, "meta_review")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchWorker:
    def __init__(self, pipeline: Pipeline, publish: Callable[[CaseAssessment], None], *, interval: float = 120.0) -> None:
        self.pipeline = pipeline
        self.publish = publish
        self.interval = max(1.0, interval)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=120)
        self._started: dict[str, float] = {}
        self._state: dict[str, Any] = {
            "status": "stopped", "revision": 0, "run_id": None,
            "case_id": None, "case_label": None, "started_at": None,
            "finished_at": None, "next_run_at": None, "completed_runs": 0,
            "interval_seconds": self.interval, "updated_at": now(),
            "steps": [{"agent": name, "status": "queued"} for name in AGENT_NAMES],
        }

    def _changed(self) -> None:
        self._state["revision"] += 1
        self._state["updated_at"] = now()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = deepcopy(self._state)
            result["events"] = deepcopy(list(reversed(self._events)))
        result["alive"] = bool(self._thread and self._thread.is_alive())
        return result

    def revision(self) -> int:
        with self._lock:
            return self._state['revision']

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        with self._lock:
            self._state["status"] = "starting"
            self._changed()
        self._thread = threading.Thread(target=self._loop, name="plagueshield-research", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=180)
        with self._lock:
            self._state["status"] = "stopped"
            self._state["next_run_at"] = None
            self._changed()

    def _progress(self, name: str, status: str, verdict: Verdict | None) -> None:
        with self._lock:
            step = next(s for s in self._state["steps"] if s["agent"] == name)
            step["status"] = status
            if status == "running":
                step["started_at"] = now()
                self._started[name] = time.monotonic()
            if verdict is not None:
                step.update({
                    "finished_at": now(),
                    "duration_ms": round((time.monotonic() - self._started[name]) * 1000),
                    "headline": verdict.headline,
                    "confidence": verdict.confidence.value,
                    "rationale": verdict.rationale[:5],
                    "reason": verdict.abstain_reason,
                    "model": verdict.data.get("model"),
                })
            self._events.append({
                **deepcopy(step), "case_id": self._state["case_id"],
                "run_id": self._state["run_id"], "produced_at": now(),
                "working_on": verdict.headline if verdict else f"Analyzing {self._state['case_id']}",
            })
            self._changed()

    def _loop(self) -> None:
        index = 0
        while not self._stop.is_set():
            try:
                cases = available_cases()
                if not cases:
                    raise RuntimeError("No case records available")
                case = load_case(cases[index % len(cases)]["case_id"])
                index += 1
                with self._lock:
                    self._state.update({
                        "status": "running", "run_id": str(uuid4()),
                        "case_id": case.case_id, "case_label": case.label,
                        "started_at": now(), "finished_at": None,
                        "next_run_at": None, "last_error": None,
                        "steps": [{"agent": name, "status": "queued"} for name in AGENT_NAMES],
                    })
                    self._started.clear()
                    self._changed()
                assessment = self.pipeline.run(case, on_progress=self._progress)
                self.publish(assessment)
                with self._lock:
                    self._state["completed_runs"] += 1
                    self._state["status"] = "waiting"
                    self._state["finished_at"] = now()
                    self._changed()
            except Exception:
                logger.exception("Background research run failed")
                with self._lock:
                    self._state["status"] = "error"
                    self._state["last_error"] = "Research run failed; retrying at the next scheduled run."
                    self._changed()
            with self._lock:
                self._state["next_run_at"] = (datetime.now(timezone.utc) + timedelta(seconds=self.interval)).isoformat()
                self._changed()
            self._stop.wait(self.interval)
