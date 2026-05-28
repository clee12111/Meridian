"""Pydantic models for Phase 8 structured synthesis output.

Used by instructor to force the LLM to return verifiable
claim-citation pairs alongside the answer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Claim(BaseModel):
    claim: str = Field(
        description="One atomic factual assertion from the answer. "
                    "One fact per claim — do not bundle multiple facts."
    )
    cited_chunk_id: str = Field(
        description="The chunk_id of the single chunk that most "
                    "directly supports this claim."
    )
    cited_text: str = Field(
        description="A verbatim excerpt from the cited chunk that "
                    "supports the claim. Must be copied exactly — "
                    "do not paraphrase."
    )


class StructuredAnswer(BaseModel):
    answer: str = Field(
        description="Full natural language answer to the query."
    )
    claims: list[Claim] = Field(
        min_length=1,
        description="One Claim per atomic fact in the answer. "
                    "Every factual assertion must be cited. "
                    "Maximum 10 claims."
    )
