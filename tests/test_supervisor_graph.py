"""Smoke tests for the Supervisor StateGraph skeleton.

Verifies that:
1. The graph compiles without error.
2. A dummy run traverses all nodes and reaches 'completed' status.
3. Conditional edges route correctly (duplicate, budget, quarantine).
4. SqliteSaver checkpointing round-trips state.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from core.supervisor.graph import build_graph, compile_graph
from core.supervisor.state import ExperimentState


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run_graph(initial_state: dict | None = None, db_path: str = ":memory:") -> dict:
    """Compile and invoke the graph, returning final state."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    compiled = build_graph().compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "test"}}
    result = compiled.invoke(initial_state or {}, config=config)
    return result


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestGraphCompilation:
    def test_build_graph_returns_state_graph(self):
        graph = build_graph()
        assert graph is not None

    def test_compile_graph_with_file(self, tmp_path: Path):
        db = tmp_path / "test.sqlite"
        compiled, saver = compile_graph(db_path=db)
        assert compiled is not None
        assert db.exists()


class TestHappyPath:
    def test_full_run_completes(self, monkeypatch):
        from core.supervisor import nodes
        from core.measurement.metrics import MetricResult

        monkeypatch.setattr(nodes, "run_eval", lambda s: {
            "metric_result": MetricResult(
                p_at_k={1: 0.0}, r_at_k={1: 0.0}, eval_mode="SPAN_OVERLAP"
            ),
            "failure_counts": {},
            "trace_id": "stub-trace-id",
        })
        result = _run_graph()
        assert result["status"] == "completed"
        assert result["notified"] is True

    def test_all_state_fields_populated(self, monkeypatch):
        from core.supervisor import nodes
        from core.measurement.metrics import MetricResult

        monkeypatch.setattr(nodes, "run_eval", lambda s: {
            "metric_result": MetricResult(
                p_at_k={1: 0.0}, r_at_k={1: 0.0}, eval_mode="SPAN_OVERLAP"
            ),
            "failure_counts": {},
            "trace_id": "stub-trace-id",
        })
        result = _run_graph()
        assert result["run_number"] == 1
        assert result["config"]["chunk_size"] == 512
        assert result["hypothesis"] == "stub hypothesis"
        assert result["config_hash"] == "stub-hash"
        assert result["duplicate"] is False
        assert result["spend_ok"] is True
        assert result["trace_id"] == "stub-trace-id"
        assert result["sanity"]["passed"] is True
        assert result["decision_entry"] == "stub entry"


class TestConditionalEdges:
    def test_duplicate_config_skips_to_notify(self, monkeypatch):
        """When check_hash returns duplicate=True, skip eval entirely."""
        from core.supervisor import nodes

        monkeypatch.setattr(nodes, "check_hash", lambda s: {
            "config_hash": "dup-hash", "duplicate": True,
            "status": "aborted", "error": "duplicate config",
        })
        # notify should still fire (sets notified=True)
        monkeypatch.setattr(nodes, "notify", lambda s: {
            "notified": True, "status": "aborted",
        })
        result = _run_graph()
        assert result["duplicate"] is True
        assert result["notified"] is True
        # run_eval should NOT have run — no trace_id
        assert "trace_id" not in result

    def test_budget_exhausted_skips_to_notify(self, monkeypatch):
        """When check_spend returns spend_ok=False, skip eval."""
        from core.supervisor import nodes

        monkeypatch.setattr(nodes, "check_spend", lambda s: {
            "spend_ok": False, "status": "aborted", "error": "budget exhausted",
        })
        monkeypatch.setattr(nodes, "notify", lambda s: {
            "notified": True, "status": "aborted",
        })
        result = _run_graph()
        assert result["spend_ok"] is False
        assert result["notified"] is True
        assert "trace_id" not in result

    def test_quarantine_skips_log_results(self, monkeypatch):
        """When sanity_check quarantines, skip log_results and write_entry."""
        from core.supervisor import nodes
        from core.measurement.metrics import MetricResult

        monkeypatch.setattr(nodes, "run_eval", lambda s: {
            "metric_result": MetricResult(
                p_at_k={1: 0.0}, r_at_k={1: 0.0}, eval_mode="SPAN_OVERLAP"
            ),
            "failure_counts": {},
            "trace_id": "stub-trace-id",
        })
        monkeypatch.setattr(nodes, "sanity_check", lambda s: {
            "sanity": {"passed": False, "violations": ["R@k not monotonic"], "quarantined": True},
        })
        monkeypatch.setattr(nodes, "notify", lambda s: {
            "notified": True, "status": "quarantined",
        })
        result = _run_graph()
        assert result["sanity"]["quarantined"] is True
        assert result["notified"] is True
        assert result["status"] == "quarantined"
        # write_entry should NOT have run — no decision_entry
        assert "decision_entry" not in result


class TestCheckpointing:
    def test_state_persisted_to_sqlite(self, monkeypatch, tmp_path: Path):
        from core.supervisor import nodes
        from core.measurement.metrics import MetricResult

        monkeypatch.setattr(nodes, "run_eval", lambda s: {
            "metric_result": MetricResult(
                p_at_k={1: 0.0}, r_at_k={1: 0.0}, eval_mode="SPAN_OVERLAP"
            ),
            "failure_counts": {},
            "trace_id": "stub-trace-id",
        })
        db = str(tmp_path / "cp.sqlite")
        result = _run_graph(db_path=db)
        assert result["status"] == "completed"

        # Re-open the saver and verify the checkpoint exists
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(db, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        config = {"configurable": {"thread_id": "test"}}
        checkpoint = saver.get_tuple(config)
        assert checkpoint is not None
