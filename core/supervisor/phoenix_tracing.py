"""Phoenix local observability setup for Meridian.

Runs alongside Langfuse — Phoenix is for experiment comparison
and embedding visualization, Langfuse is for production tracing.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def setup_phoenix(project_name: str = "meridian") -> bool:
    """Register Phoenix tracer. Returns True if setup succeeded.

    Reads PHOENIX_ENABLED=1 from env to activate.
    If not set or Phoenix not installed, returns False silently.
    Phoenix runs at localhost:6006 by default.
    """
    if os.environ.get("PHOENIX_ENABLED") != "1":
        return False
    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import (
            LangChainInstrumentor,
        )
        from openinference.instrumentation.openai import (
            OpenAIInstrumentor,
        )

        register(
            project_name=project_name,
            endpoint="http://localhost:6006/v1/traces",
        )

        # Auto-instrument LangGraph (via LangChain instrumentor)
        # and DeepSeek (via OpenAI-compatible instrumentor)
        LangChainInstrumentor().instrument()
        OpenAIInstrumentor().instrument()

        logger.info(
            "Phoenix tracing enabled — UI at http://localhost:6006"
        )
        return True
    except ImportError:
        logger.warning("Phoenix not installed — tracing disabled")
        return False
    except Exception as e:
        logger.warning("Phoenix setup failed: %s", e)
        return False


def log_reranker_span(
    query: str,
    candidates_in: list[dict],
    candidates_out: list[dict],
    model: str = "rerank-2.5",
) -> None:
    """Log a custom reranker span to Phoenix.

    candidates_in: list of {chunk_id, score, text} dicts (top 50)
    candidates_out: list of {chunk_id, score, text} dicts (top 8)

    This is the critical span for understanding why the reranker
    promotes wrong-document chunks above right-document ones.
    """
    if os.environ.get("PHOENIX_ENABLED") != "1":
        return
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("reranker") as span:
            span.set_attribute("reranker.model", model)
            span.set_attribute("reranker.query", query)
            span.set_attribute(
                "reranker.candidates_in_count", len(candidates_in)
            )
            span.set_attribute(
                "reranker.candidates_out_count", len(candidates_out)
            )

            # Log top 5 candidates IN with their doc_ids and scores
            for i, c in enumerate(candidates_in[:5]):
                span.set_attribute(
                    f"reranker.in.{i}.chunk_id",
                    c.get("chunk_id", ""),
                )
                span.set_attribute(
                    f"reranker.in.{i}.doc_id",
                    c.get("chunk_id", "").split("#")[0],
                )
                span.set_attribute(
                    f"reranker.in.{i}.rrf_score",
                    float(c.get("score", 0)),
                )

            # Log top 5 candidates OUT with reranker scores
            for i, c in enumerate(candidates_out[:5]):
                span.set_attribute(
                    f"reranker.out.{i}.chunk_id",
                    c.get("chunk_id", ""),
                )
                span.set_attribute(
                    f"reranker.out.{i}.doc_id",
                    c.get("chunk_id", "").split("#")[0],
                )
                span.set_attribute(
                    f"reranker.out.{i}.rerank_score",
                    float(c.get("score", 0)),
                )
    except Exception as e:
        logger.warning("Phoenix reranker span failed: %s", e)
