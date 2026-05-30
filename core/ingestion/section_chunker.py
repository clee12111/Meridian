"""Section-aware chunker for legal documents.

Splits on structural boundaries (numbered sections, ALL-CAPS headings,
article markers) then merges/splits to hit the ~512 char target.
Falls back to fixed-stride chunking when no structure is detected.

Output: list of SectionChunk with character-based half-open [start, end)
offsets — source_text[start:end] == chunk.content for every chunk.
This invariant is load-bearing: Tier B measurement does char-set overlap
of retrieved-chunk spans vs GT spans.

Boundary detection is corpus-general (cascading regex, not per-corpus
branches) so the same chunker works on ContractNLI, CUAD, MAUD, PrivacyQA,
or any future corpus.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# ── Size budget (chars, matching baseline 512/128) ──────────────────────────

CHUNK_CAP = 512       # hard cap per chunk
CHUNK_FLOOR = 200     # merge fragments below this into neighbor
SPLIT_OVERLAP = 64    # small overlap only when force-splitting oversize sections

# ── Boundary detection ──────────────────────────────────────────────────────
#
# Cascading patterns, tested in order. A line that matches ANY pattern is a
# section boundary. Patterns are intentionally general — they fire on CUAD,
# ContractNLI, MAUD, and PrivacyQA without corpus-specific branches.
#
# Order matters: more specific patterns first, catch-all last.

_BOUNDARY_PATTERNS: list[re.Pattern] = [
    # Article-level: "Article I", "ARTICLE IV", "Article 1"
    re.compile(r'^(?:ARTICLE|Article)\s+[IVXLCDM\d]+', re.MULTILINE),

    # Numbered sections: "1.", "2.1", "10.3.2", "1.1." (with or without trailing period)
    # Must start at line beginning (possibly after whitespace).
    # Require the number to be followed by a capital letter, space+capital, or tab
    # to avoid matching decimal numbers in prose like "3.5 million".
    re.compile(r'^\s*(\d{1,3}\.)+\s*[A-Z\("]', re.MULTILINE),

    # ALL-CAPS titled lines (>=4 caps chars, at most 80 chars total, mostly uppercase).
    # Filters out lines that are just short abbreviations or inline shouts.
    re.compile(r'^[A-Z][A-Z\s,;:\-/&]{3,78}[A-Z.)]\s*$', re.MULTILINE),

    # Lettered subsections at line start: "(a)", "(b)", "(i)", "(iv)"
    re.compile(r'^\s*\([a-z]{1,4}\)\s', re.MULTILINE),

    # "Section X" / "SECTION X" headers
    re.compile(r'^(?:SECTION|Section)\s+\d+', re.MULTILINE),

    # "EXHIBIT", "SCHEDULE", "ANNEX", "APPENDIX" headers
    re.compile(r'^(?:EXHIBIT|SCHEDULE|ANNEX|APPENDIX)\s+[A-Z\d]', re.MULTILINE),
]

# Sub-split boundaries: sentence ends or paragraph breaks, used when a single
# section exceeds CHUNK_CAP.
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z("])')
_PARA_BREAK = re.compile(r'\n\s*\n')


@dataclass
class SectionChunk:
    """One chunk ready for parquet emission."""
    doc_id: str
    content: str
    start: int          # char offset into source doc (inclusive)
    end: int            # char offset into source doc (exclusive)
    chunk_index: int    # sequential within this doc


@dataclass
class HierarchicalResult:
    """Parent + child spans from hierarchical chunking.

    parents: list of (start, end) for each parent section.
    children: list of (start, end, parent_idx) where parent_idx indexes
              into parents. For leaf sections (no split needed), the leaf
              is both a parent and a child pointing to itself.
    """
    parents: list[tuple[int, int]]
    children: list[tuple[int, int, int]]  # (start, end, parent_idx)


def _find_boundaries(text: str) -> list[int]:
    """Return sorted, deduplicated char offsets where section boundaries occur."""
    offsets: set[int] = set()
    for pattern in _BOUNDARY_PATTERNS:
        for m in pattern.finditer(text):
            offsets.add(m.start())
    return sorted(offsets)


def _sub_split(text: str, start_offset: int, cap: int, overlap: int) -> list[tuple[int, int]]:
    """Split an oversize section into chunks of <= cap chars.

    Tries paragraph breaks first, then sentence boundaries, then hard stride.
    Returns list of (start, end) half-open offsets into the original document.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        if text_len - pos <= cap:
            spans.append((start_offset + pos, start_offset + text_len))
            break

        window = text[pos : pos + cap]

        # Try paragraph break
        cut = None
        for m in _PARA_BREAK.finditer(window):
            candidate = m.end()
            if candidate >= CHUNK_FLOOR // 2:
                cut = candidate

        # Try sentence break
        if cut is None:
            for m in _SENTENCE_END.finditer(window):
                candidate = m.start()
                if candidate >= CHUNK_FLOOR // 2:
                    cut = candidate

        # Hard stride
        if cut is None:
            cut = cap

        spans.append((start_offset + pos, start_offset + pos + cut))
        pos += cut - overlap if cut > overlap else cut

    return spans


def chunk_document(text: str, doc_id: str) -> list[SectionChunk]:
    """Chunk one document using section-aware boundaries with fixed-stride fallback.

    Returns chunks with char-based [start, end) offsets such that
    text[start:end] == chunk.content for every chunk.
    """
    if not text.strip():
        return []

    boundaries = _find_boundaries(text)

    # If no structural boundaries found, fall back to fixed-stride chunking
    if len(boundaries) < 2:
        return _fixed_stride_chunks(text, doc_id)

    # Ensure we have boundaries at start and end
    if boundaries[0] != 0:
        boundaries.insert(0, 0)

    # Build raw sections from boundaries
    raw_sections: list[tuple[int, int]] = []
    for i in range(len(boundaries)):
        s = boundaries[i]
        e = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        # Skip empty sections
        if s < e and text[s:e].strip():
            raw_sections.append((s, e))

    if not raw_sections:
        return _fixed_stride_chunks(text, doc_id)

    # Merge/split to hit size budget
    final_spans = _merge_and_split(text, raw_sections)

    return [
        SectionChunk(
            doc_id=doc_id,
            content=text[s:e],
            start=s,
            end=e,
            chunk_index=i,
        )
        for i, (s, e) in enumerate(final_spans)
    ]


def _merge_and_split(text: str, sections: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge sub-floor sections and split oversize ones.

    No overlap at section boundaries; SPLIT_OVERLAP only within force-split sections.
    """
    spans: list[tuple[int, int]] = []

    pending_start: int | None = None
    pending_end: int | None = None

    def flush() -> None:
        nonlocal pending_start, pending_end
        if pending_start is None:
            return
        length = pending_end - pending_start
        if length > CHUNK_CAP:
            spans.extend(_sub_split(text[pending_start:pending_end],
                                    pending_start, CHUNK_CAP, SPLIT_OVERLAP))
        else:
            spans.append((pending_start, pending_end))
        pending_start = None
        pending_end = None

    for s, e in sections:
        sec_len = e - s

        if pending_start is None:
            pending_start = s
            pending_end = e
            continue

        pending_len = pending_end - pending_start
        combined = e - pending_start  # contiguous from pending_start to e

        # If pending is below floor, always merge
        if pending_len < CHUNK_FLOOR:
            pending_end = e
            continue

        # If combined fits in cap, merge
        if combined <= CHUNK_CAP:
            pending_end = e
            continue

        # Combined doesn't fit — flush pending, start new
        flush()
        pending_start = s
        pending_end = e

    flush()

    # Post-pass: merge any trailing sub-floor chunk into its predecessor.
    # This handles the case where the last section in a document is tiny
    # (e.g., "EXHIBIT 10.1") and the forward-merge couldn't absorb it
    # because the predecessor was already flushed.
    if len(spans) >= 2:
        last_s, last_e = spans[-1]
        if last_e - last_s < CHUNK_FLOOR:
            prev_s, prev_e = spans[-2]
            # Merge by extending the predecessor to cover the trailing chunk
            spans[-2] = (prev_s, last_e)
            spans.pop()

    return spans


def _fixed_stride_chunks(text: str, doc_id: str) -> list[SectionChunk]:
    """Baseline fixed-stride chunking: 512 char cap, 128 char overlap."""
    stride = CHUNK_CAP - 128  # 384
    chunks: list[SectionChunk] = []
    pos = 0
    idx = 0
    while pos < len(text):
        end = min(pos + CHUNK_CAP, len(text))
        content = text[pos:end]
        if content.strip():
            chunks.append(SectionChunk(
                doc_id=doc_id,
                content=content,
                start=pos,
                end=end,
                chunk_index=idx,
            ))
            idx += 1
        if end == len(text):
            break
        pos += stride
    return chunks


def _merge_and_split_hierarchical(
    text: str, sections: list[tuple[int, int]],
) -> HierarchicalResult:
    """Merge sub-floor sections and split oversize ones, emitting BOTH levels.

    For sections <= CHUNK_CAP: emitted as leaf (parent and child are identical).
    For sections > CHUNK_CAP: parent span preserved, children are sub-splits.
    """
    parents: list[tuple[int, int]] = []
    children: list[tuple[int, int, int]] = []  # (start, end, parent_idx)

    pending_start: int | None = None
    pending_end: int | None = None

    def flush() -> None:
        nonlocal pending_start, pending_end
        if pending_start is None:
            return
        length = pending_end - pending_start
        parent_idx = len(parents)
        parents.append((pending_start, pending_end))

        if length > CHUNK_CAP:
            # Oversize: emit parent + sub-split children
            child_spans = _sub_split(
                text[pending_start:pending_end],
                pending_start, CHUNK_CAP, SPLIT_OVERLAP,
            )
            for cs, ce in child_spans:
                children.append((cs, ce, parent_idx))
        else:
            # Leaf: parent == child
            children.append((pending_start, pending_end, parent_idx))

        pending_start = None
        pending_end = None

    for s, e in sections:
        if pending_start is None:
            pending_start = s
            pending_end = e
            continue

        pending_len = pending_end - pending_start
        combined = e - pending_start

        if pending_len < CHUNK_FLOOR:
            pending_end = e
            continue

        if combined <= CHUNK_CAP:
            pending_end = e
            continue

        flush()
        pending_start = s
        pending_end = e

    flush()

    # Post-pass: merge trailing sub-floor parent into predecessor
    if len(parents) >= 2:
        last_s, last_e = parents[-1]
        if last_e - last_s < CHUNK_FLOOR:
            prev_s, _ = parents[-2]
            # Merge: extend predecessor parent, reassign children
            merged_parent_idx = len(parents) - 2
            removed_parent_idx = len(parents) - 1
            parents[merged_parent_idx] = (prev_s, last_e)
            parents.pop()
            # Reassign children of removed parent to merged parent
            children = [
                (cs, ce, merged_parent_idx if pi == removed_parent_idx else pi)
                for cs, ce, pi in children
            ]
            # Re-merge children of the extended parent if needed
            # (the predecessor's children + the removed parent's children
            # now all point to merged_parent_idx)

    return HierarchicalResult(parents=parents, children=children)


def chunk_document_hierarchical(text: str, doc_id: str) -> HierarchicalResult:
    """Chunk one document with parent-child hierarchy.

    Returns HierarchicalResult with parent sections and child sub-splits.
    For leaf sections (no split needed), parent == child.
    Falls back to fixed-stride with synthetic parents on unstructured docs.
    """
    if not text.strip():
        return HierarchicalResult(parents=[], children=[])

    boundaries = _find_boundaries(text)

    if len(boundaries) < 2:
        # Fallback: fixed-stride children, each is its own parent
        parents = []
        children = []
        stride = CHUNK_CAP - 128
        pos = 0
        while pos < len(text):
            end = min(pos + CHUNK_CAP, len(text))
            if text[pos:end].strip():
                parent_idx = len(parents)
                parents.append((pos, end))
                children.append((pos, end, parent_idx))
            if end == len(text):
                break
            pos += stride
        return HierarchicalResult(parents=parents, children=children)

    if boundaries[0] != 0:
        boundaries.insert(0, 0)

    raw_sections: list[tuple[int, int]] = []
    for i in range(len(boundaries)):
        s = boundaries[i]
        e = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        if s < e and text[s:e].strip():
            raw_sections.append((s, e))

    if not raw_sections:
        # Same fallback
        parents = []
        children = []
        stride = CHUNK_CAP - 128
        pos = 0
        while pos < len(text):
            end = min(pos + CHUNK_CAP, len(text))
            if text[pos:end].strip():
                parent_idx = len(parents)
                parents.append((pos, end))
                children.append((pos, end, parent_idx))
            if end == len(text):
                break
            pos += stride
        return HierarchicalResult(parents=parents, children=children)

    return _merge_and_split_hierarchical(text, raw_sections)


def compute_checksum(content: str) -> str:
    """SHA-256 hex digest of chunk content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
