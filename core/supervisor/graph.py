"""Supervisor graph: LangGraph StateGraph with SqliteSaver checkpointing.

v2 phase wiring (phases 3–10):
  query_understanding → retrieval → fusion → reranking
  → context_construction → synthesis → verification → agentic_loop
  → (conditional: END if loop_complete, else back to retrieval)

Each node receives PipelineContext via bind_context wrapper.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from core.supervisor.context import PipelineContext
from core.supervisor.state import ExperimentState
from core.supervisor import nodes


def bind_context(node_fn: Callable, context: PipelineContext) -> Callable:
    """Wrap a node function so PipelineContext is injected as second arg."""

    def wrapped(state: ExperimentState) -> dict:
        return node_fn(state, context)

    wrapped.__name__ = node_fn.__name__
    return wrapped


def _after_agentic_loop(state: ExperimentState) -> str:
    """Route agentic loop: END if complete, else back to retrieval."""
    if state.get("loop_complete", True):
        return END
    return "retrieval"


def build_graph(context: PipelineContext) -> StateGraph:
    """Construct the un-compiled StateGraph (useful for testing)."""
    graph = StateGraph(ExperimentState)

    # --- nodes (phases 3–10) ---
    graph.add_node("query_understanding",  bind_context(nodes.query_understanding, context))
    graph.add_node("retrieval",            bind_context(nodes.retrieval, context))
    graph.add_node("fusion",               bind_context(nodes.fusion, context))
    graph.add_node("reranking",            bind_context(nodes.reranking, context))
    graph.add_node("context_construction", bind_context(nodes.context_construction, context))
    graph.add_node("synthesis",            bind_context(nodes.synthesis, context))
    graph.add_node("verification",         bind_context(nodes.verification, context))
    graph.add_node("agentic_loop",         bind_context(nodes.agentic_loop, context))

    # --- edges ---
    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding",  "retrieval")
    graph.add_edge("retrieval",            "fusion")
    graph.add_edge("fusion",               "reranking")
    graph.add_edge("reranking",            "context_construction")
    graph.add_edge("context_construction", "synthesis")
    graph.add_edge("synthesis",            "verification")
    graph.add_edge("verification",         "agentic_loop")

    # Phase 10 conditional: loop back or finish
    graph.add_conditional_edges(
        "agentic_loop",
        _after_agentic_loop,
        {"retrieval": "retrieval", END: END},
    )

    return graph


def compile_graph(
    db_path: str | Path,
    context: PipelineContext,
) -> tuple:
    """Compile the graph with SqliteSaver checkpointing.

    Returns (compiled_graph, saver) so the caller can manage the
    saver's connection lifecycle.  The caller should close
    ``saver.conn`` when done.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    compiled = build_graph(context).compile(checkpointer=saver)
    return compiled, saver
