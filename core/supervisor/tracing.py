"""Thin Langfuse tracing helpers.

All functions fail silently — tracing must never break the pipeline.
Nodes call these instead of importing Langfuse directly.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def start_trace(context: Any, name: str, input: dict) -> Any:
    """Start a Langfuse trace. Returns trace object or None."""
    if not context.langfuse_client:
        return None
    try:
        return context.langfuse_client.trace(
            name=name,
            input=input,
        )
    except Exception as e:
        logger.warning("Langfuse trace start failed: %s", e)
        return None


def start_span(trace: Any, name: str, input: dict) -> Any:
    """Start a span on an existing trace. Returns span or None."""
    if trace is None:
        return None
    try:
        return trace.span(name=name, input=input)
    except Exception as e:
        logger.warning("Langfuse span start failed: %s", e)
        return None


def end_span(span: Any, output: dict) -> None:
    """End a span with output data."""
    if span is None:
        return
    try:
        span.end(output=output)
    except Exception as e:
        logger.warning("Langfuse span end failed: %s", e)


def end_trace(trace: Any, output: dict) -> None:
    """End trace with final output."""
    if trace is None:
        return
    try:
        trace.update(output=output)
    except Exception as e:
        logger.warning("Langfuse trace end failed: %s", e)


def flush(context: Any) -> None:
    """Flush pending Langfuse events. Call before process exit."""
    if not context.langfuse_client:
        return
    try:
        context.langfuse_client.flush()
    except Exception as e:
        logger.warning("Langfuse flush failed: %s", e)


def log_llm_call(
    trace: Any,
    name: str,
    model: str,
    input: dict,
    output: dict,
    usage: dict | None = None,
) -> None:
    """Log an LLM generation call as a Langfuse generation."""
    if trace is None:
        return
    try:
        trace.generation(
            name=name,
            model=model,
            input=input,
            output=output,
            usage=usage,
        )
    except Exception as e:
        logger.warning("Langfuse generation log failed: %s", e)
