"""Supervisor graph: LangGraph StateGraph with SqliteSaver checkpointing.

Nodes are wired in the Phase 3 order specified in CLAUDE.md:
  read_ledger -> propose_config -> check_hash -> check_spend
  -> run_eval -> sanity_check -> log_results -> write_entry -> notify

Conditional edges after check_hash and check_spend short-circuit
to notify on duplicate config or budget exhaustion.  Conditional
edge after sanity_check routes to log_results (pass) or directly
to notify (quarantine).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from core.supervisor.state import ExperimentState
from core.supervisor import nodes


def _after_check_hash(state: ExperimentState) -> str:
    """Skip to notify if config is a duplicate."""
    if state.get("duplicate"):
        return "notify"
    return "check_spend"


def _after_check_spend(state: ExperimentState) -> str:
    """Skip to notify if budget is exhausted."""
    if not state.get("spend_ok"):
        return "notify"
    return "run_eval"


def _after_sanity_check(state: ExperimentState) -> str:
    """Route quarantined runs past log_results to notify."""
    sanity = state.get("sanity", {})
    if sanity.get("quarantined"):
        return "notify"
    return "log_results"


def build_graph() -> StateGraph:
    """Construct the un-compiled StateGraph (useful for testing)."""
    graph = StateGraph(ExperimentState)

    # --- nodes ---
    graph.add_node("read_ledger", nodes.read_ledger)
    graph.add_node("propose_config", nodes.propose_config)
    graph.add_node("check_hash", nodes.check_hash)
    graph.add_node("check_spend", nodes.check_spend)
    graph.add_node("run_eval", nodes.run_eval)
    graph.add_node("sanity_check", nodes.sanity_check)
    graph.add_node("log_results", nodes.log_results)
    graph.add_node("write_entry", nodes.write_entry)
    graph.add_node("notify", nodes.notify)

    # --- edges ---
    graph.set_entry_point("read_ledger")
    graph.add_edge("read_ledger", "propose_config")
    graph.add_edge("propose_config", "check_hash")

    graph.add_conditional_edges("check_hash", _after_check_hash)
    graph.add_conditional_edges("check_spend", _after_check_spend)

    graph.add_edge("run_eval", "sanity_check")

    graph.add_conditional_edges("sanity_check", _after_sanity_check)

    graph.add_edge("log_results", "write_entry")
    graph.add_edge("write_entry", "notify")
    graph.add_edge("notify", END)

    return graph


def compile_graph(db_path: str | Path = "checkpoints.sqlite") -> tuple:
    """Compile the graph with SqliteSaver checkpointing.

    Returns (compiled_graph, saver) so the caller can manage the
    saver's connection lifecycle.  The caller should close
    ``saver.conn`` when done.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    compiled = build_graph().compile(checkpointer=saver)
    return compiled, saver
