"""Thin Langfuse tracing helpers (v4 SDK).

All functions fail silently — tracing must never break the pipeline.
Nodes call these instead of importing Langfuse directly.

Langfuse v4 API:
  lf.start_observation(name, input) → LangfuseSpan (creates trace implicitly)
  span.start_observation(name, input) → child LangfuseSpan
  span.update(output=...) → set output
  span.end() → close the span
  lf.flush() → send pending events
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def start_trace(context: Any, name: str, input: dict) -> Any:
    """Start a Langfuse trace (top-level observation). Returns span or None."""
    if not context.langfuse_client:
        return None
    try:
        return context.langfuse_client.start_observation(
            name=name,
            input=input,
        )
    except Exception as e:
        logger.warning("Langfuse trace start failed: %s", e)
        return None


def start_span(trace: Any, name: str, input: dict) -> Any:
    """Start a child span on an existing trace/span. Returns span or None."""
    if trace is None:
        return None
    try:
        return trace.start_observation(name=name, input=input)
    except Exception as e:
        logger.warning("Langfuse span start failed: %s", e)
        return None


def end_span(span: Any, output: dict) -> None:
    """End a span with output data."""
    if span is None:
        return
    try:
        span.update(output=output)
        span.end()
    except Exception as e:
        logger.warning("Langfuse span end failed: %s", e)


def end_trace(trace: Any, output: dict) -> None:
    """End trace with final output."""
    if trace is None:
        return
    try:
        trace.update(output=output)
        trace.end()
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
