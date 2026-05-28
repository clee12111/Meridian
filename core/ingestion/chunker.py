"""
Markdown section-based chunker for the FastAPI documentation corpus.

Token counting: tiktoken cl100k_base (GPT-4 / text-embedding-ada-002 encoding).
  - cl100k_base is the established production-standard reference encoding for RAG
    systems; widely used for chunk-size decisions even when the serving model differs.
  - o200k_base (GPT-4o / o1) is marginally different on code; cl100k_base chosen
    because it's the documented default in LangChain, LlamaIndex, and most RAG guides.
  - The encoding is loaded once at import time and shared via module-level singleton.

Strategy (locked from corpus inspection 2026-05-21):
  - Split on ## and ### header boundaries; each section = header + content.
  - Merge consecutive small sections within a file (floor: ~200 tokens).
  - Target chunk size: ~512 tokens.
  - Never split a fenced code block: oversized chunks allowed, flagged in metadata.
  - Zero overlap to start (revisit as a measured experiment later).
"""

from __future__ import annotations

import re
import sys
import io
from dataclasses import dataclass
from pathlib import Path

import tiktoken

# ─── Token counting ──────────────────────────────────────────────────────────

# Encoding loaded once; cl100k_base chosen as the field-standard reference.
TOKEN_METHOD = "tiktoken/cl100k_base"
_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Exact token count via tiktoken cl100k_base."""
    return len(_ENC.encode(text))


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Section:
    """One parsed markdown section: a header + its content until the next peer header."""
    source_file: Path
    # Full hierarchy leading to this section, e.g. "Tutorial > First Steps > Step 1"
    header_path: str
    # The raw markdown text of this section (header line + body)
    text: str
    token_count: int


@dataclass
class Chunk:
    """One chunk as it will be stored in the vector DB."""
    source_file: Path
    header_path: str        # of the first section in this chunk
    text: str
    token_count: int
    # Metadata flags
    oversized: bool = False
    oversized_reason: str = ""
    token_method: str = TOKEN_METHOD


# ─── Section parser ──────────────────────────────────────────────────────────

_HEADER_LINE_RE = re.compile(r'^(#{1,3})\s+(.+)$')


def _find_real_headers(text: str) -> list[tuple[int, int, str]]:
    """
    Return (char_offset, level, title) for every H1/H2/H3 that is NOT inside
    a fenced code block.

    Bug fixed 2026-05-21: the regex-only approach matched `# Python comments`
    and `# shell comments` inside ``` blocks as section headers, splitting code
    blocks across section (and chunk) boundaries and producing odd fence counts.

    Fix: walk the document line by line, tracking open/close fence state.
    Only emit a header match while the fence depth is 0.
    Fence type (``` vs ~~~) is tracked separately so a backtick fence cannot be
    closed by a tilde fence and vice versa.
    """
    results: list[tuple[int, int, str]] = []
    pos = 0
    in_fence = False
    fence_char: str = ""          # "`" or "~"
    fence_min_len: int = 0        # minimum run of fence_char that closes this fence

    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        stripped = line.lstrip()

        if not in_fence:
            # Check for fence opening: line whose stripped form starts with ``` or ~~~
            if stripped.startswith("```") or stripped.startswith("~~~"):
                ch = stripped[0]
                run = len(stripped) - len(stripped.lstrip(ch))
                in_fence = True
                fence_char = ch
                fence_min_len = run
            else:
                # Only match headers outside fences
                m = _HEADER_LINE_RE.match(line)
                if m:
                    results.append((pos, len(m.group(1)), m.group(2).strip()))
        else:
            # Inside a fence: look for a closing delimiter of the same type and
            # length >= the opening run.
            if stripped.startswith(fence_char * fence_min_len):
                remainder = stripped.lstrip(fence_char)
                # Closing fence: stripped content is only fence chars (possibly trailing space)
                if not remainder.strip():
                    in_fence = False
                    fence_char = ""
                    fence_min_len = 0

        pos += len(raw_line)

    return results


def parse_sections(source_file: Path, text: str) -> list[Section]:
    """
    Split a markdown document into sections at H1/H2/H3 boundaries.

    Rules:
    - Each section = its header line + all content until the next header
      of the same or higher level (i.e. same-or-fewer #'s).
    - The header_path carries the full ancestor chain, e.g.
        "Tutorial > Request Body > Use the model"
    - If a document has no headers at all, the whole file is one section
      with header_path = "(no headers)".
    - Headers inside fenced code blocks are ignored (they are code comments,
      not structural markdown headers).
    """
    if not text.strip():
        return []

    spans = _find_real_headers(text)   # [(offset, level, title), ...]

    if not spans:
        return [Section(
            source_file=source_file,
            header_path="(no headers)",
            text=text,
            token_count=_count_tokens(text),
        )]

    sections: list[Section] = []
    ancestor_titles: list[str] = []
    ancestor_levels: list[int] = []

    for i, (start, level, title) in enumerate(spans):
        end = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        body = text[start:end]

        # Maintain ancestor stack
        while ancestor_levels and ancestor_levels[-1] >= level:
            ancestor_levels.pop()
            ancestor_titles.pop()
        ancestor_titles.append(title)
        ancestor_levels.append(level)

        sections.append(Section(
            source_file=source_file,
            header_path=" > ".join(ancestor_titles),
            text=body,
            token_count=_count_tokens(body),
        ))

    return sections


# ─── Chunk assembly ──────────────────────────────────────────────────────────

CHUNK_TARGET = 512    # target max tokens per chunk
CHUNK_FLOOR  = 200    # merge sections below this floor
def _fence_count(text: str) -> int:
    """
    Count fence-delimiter lines in text.

    A fence delimiter is a line whose stripped content starts with ``` or ~~~
    (three or more backticks/tildes), optionally followed by a language tag.
    This is line-level detection — it does NOT count ``` appearing inside prose
    or inline code spans, avoiding false positives from text like "use the ``
    ` operator" or inline `code` ticks.
    """
    count = 0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            count += 1
    return count


def _contains_code_block(text: str) -> bool:
    return _fence_count(text) >= 2


def _oversize_reason(text: str, token_count: int, context: str = "single section") -> str:
    """Produce a specific oversized reason: distinguish code-block from dense prose."""
    if _contains_code_block(text):
        return f"{context} exceeds target ({token_count} tok); large code block"
    return f"{context} exceeds target ({token_count} tok); dense prose (no code block)"


def assemble_chunks(sections: list[Section]) -> list[Chunk]:
    """
    Assemble sections into chunks.

    Rules (locked 2026-05-21):
    1. Walk sections in document order within each file. Never merge across files.
    2. Accumulate into a pending chunk; emit when the next section would push it
       past CHUNK_TARGET — UNLESS pending is below CHUNK_FLOOR (force-merge).
    3. If a single section alone exceeds CHUNK_TARGET, emit it as one oversized
       chunk (B1: never split a fenced code block).
    4. End-of-file remnants (pending below floor at file end) are flushed as-is;
       they are reported but not treated as bugs.
    """
    if not sections:
        return []

    chunks: list[Chunk] = []

    pending_text: str = ""
    pending_tokens: int = 0
    pending_header_path: str = ""
    pending_source: Path = sections[0].source_file

    def flush(oversized: bool = False, reason: str = "") -> None:
        nonlocal pending_text, pending_tokens, pending_header_path
        if not pending_text.strip():
            return
        chunks.append(Chunk(
            source_file=pending_source,
            header_path=pending_header_path,
            text=pending_text,
            token_count=pending_tokens,
            oversized=oversized,
            oversized_reason=reason,
        ))
        pending_text = ""
        pending_tokens = 0
        pending_header_path = ""

    for sec in sections:
        # ── File boundary ──────────────────────────────────────────────────
        if sec.source_file != pending_source:
            flush()                          # end-of-file remnant if under floor
            pending_source = sec.source_file

        # ── Start a fresh pending chunk ────────────────────────────────────
        if not pending_text:
            pending_text = sec.text
            pending_tokens = sec.token_count
            pending_header_path = sec.header_path
            # Single section already oversized — emit immediately (B1)
            if sec.token_count > CHUNK_TARGET:
                flush(oversized=True, reason=_oversize_reason(sec.text, sec.token_count))
            continue

        combined_tokens = pending_tokens + sec.token_count

        if combined_tokens <= CHUNK_TARGET:
            # ── Normal merge: fits within target ──────────────────────────
            pending_text += "\n\n" + sec.text
            pending_tokens = combined_tokens

        elif pending_tokens < CHUNK_FLOOR:
            # ── Force-merge: pending is below floor, must absorb ──────────
            # Merge regardless of target; emit oversized if we crossed it.
            pending_text += "\n\n" + sec.text
            pending_tokens = combined_tokens
            if combined_tokens > CHUNK_TARGET:
                flush(
                    oversized=True,
                    reason=_oversize_reason(
                        pending_text, combined_tokens,
                        context="floor-forced merge",
                    ),
                )

        else:
            # ── Emit pending, start new chunk with this section ───────────
            flush()
            pending_text = sec.text
            pending_tokens = sec.token_count
            pending_header_path = sec.header_path
            # New pending is itself oversized (B1)
            if sec.token_count > CHUNK_TARGET:
                flush(oversized=True, reason=_oversize_reason(sec.text, sec.token_count))

    flush()  # end-of-file remnant for last file
    return chunks


# ─── Corpus-level runner ─────────────────────────────────────────────────────

def chunk_corpus(corpus_dir: Path) -> list[Chunk]:
    """Parse and chunk every .md file under corpus_dir."""
    md_files = sorted(corpus_dir.rglob("*.md"))
    all_chunks: list[Chunk] = []
    for f in md_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        sections = parse_sections(f, text)
        chunks = assemble_chunks(sections)
        all_chunks.extend(chunks)
    return all_chunks


# ─── Reporting helpers ────────────────────────────────────────────────────────

_SECTION_BUCKETS = ["<200", "200-500", "500-1000", "1000-2000", ">2000"]
_CHUNK_BUCKETS   = ["<200", "200-512", "512-1000", "1000-2000", ">2000"]


def _section_bucket(t: int) -> str:
    if t < 200:   return "<200"
    if t < 500:   return "200-500"
    if t < 1000:  return "500-1000"
    if t < 2000:  return "1000-2000"
    return ">2000"


def _chunk_bucket(t: int) -> str:
    if t < 200:   return "<200"
    if t < 512:   return "200-512"
    if t < 1000:  return "512-1000"
    if t < 2000:  return "1000-2000"
    return ">2000"


def report_sections(sections: list[Section]) -> None:
    from collections import Counter
    buckets: Counter[str] = Counter()
    for s in sections:
        buckets[_section_bucket(s.token_count)] += 1
    print(f"Total sections: {len(sections)}")
    print("Token-size distribution:")
    for b in _SECTION_BUCKETS:
        print(f"  {b:>10} tokens: {buckets[b]:>4}")


def report_chunks(chunks: list[Chunk]) -> None:
    from collections import Counter
    import statistics

    buckets: Counter[str] = Counter()
    under_floor: list[Chunk] = []
    oversized: list[Chunk] = []

    for c in chunks:
        buckets[_chunk_bucket(c.token_count)] += 1
        if c.token_count < CHUNK_FLOOR:
            under_floor.append(c)
        if c.oversized:
            oversized.append(c)

    counts = [c.token_count for c in chunks]
    total = len(chunks)

    print(f"\nTotal chunks: {total}")
    print(f"Mean size:    {statistics.mean(counts):.0f} tokens")
    print(f"Median size:  {statistics.median(counts):.0f} tokens")
    print("\nChunk size distribution:")
    for b in _CHUNK_BUCKETS:
        n = buckets[b]
        bar = "#" * (n // 2)
        print(f"  {b:>10} tokens: {n:>4}  {bar}")

    # ── Under-floor classification ────────────────────────────────────────
    # Build a set of (file, is_last_chunk_for_file) to classify remnants.
    # A chunk is a whole-file chunk if it's the only chunk from that file.
    # A chunk is an end-of-file remnant if it's the last chunk from its file
    # and there are other chunks from the same file.
    # Any other under-floor chunk is a merging bug.
    from collections import defaultdict
    file_chunks: dict[Path, list[Chunk]] = defaultdict(list)
    for c in chunks:
        file_chunks[c.source_file].append(c)

    print(f"\nChunks under floor (<{CHUNK_FLOOR} tokens): {len(under_floor)}")
    if under_floor:
        for c in sorted(under_floor, key=lambda x: x.token_count):
            siblings = file_chunks[c.source_file]
            if len(siblings) == 1:
                tag = "whole-file (file smaller than floor)"
            elif siblings[-1] is c:
                tag = "end-of-file remnant"
            else:
                tag = "MERGING BUG — not last chunk in file"
            rel = str(c.source_file.relative_to(c.source_file.parent.parent.parent.parent)
                      if c.source_file.parts else c.source_file)
            # Use just filename + parent for brevity
            short = f"{c.source_file.parent.name}/{c.source_file.name}"
            print(f"  {c.token_count:>4} tok  {short:<35}  [{tag}]")

    # ── Oversized list ────────────────────────────────────────────────────
    print(f"\nOversized chunks (>{CHUNK_TARGET} tokens): {len(oversized)}")
    for c in sorted(oversized, key=lambda x: -x.token_count):
        short = f"{c.source_file.parent.name}/{c.source_file.name}"
        print(f"  {c.token_count:>5} tok  {short:<35}  {c.header_path[:45]}")
        print(f"           reason: {c.oversized_reason}")


def report_fence_integrity(chunks: list[Chunk]) -> None:
    """
    Exhaustively verify no fenced code block was split across chunk boundaries.

    Method: count fence-delimiter lines (line-level, not substring) in every chunk.
    A chunk with correctly contained code blocks has an EVEN fence count.
    An ODD count means a block was torn — opening fence in this chunk, closing in
    the next (or vice versa).

    Also reports the forced-merge sub-population separately, since that's the
    highest-risk path for accidental fence splitting.
    """
    odd: list[tuple[Chunk, int]] = []
    forced_merge_odd: list[tuple[Chunk, int]] = []

    for c in chunks:
        fc = _fence_count(c.text)
        if fc % 2 != 0:
            odd.append((c, fc))
            if "floor-forced merge" in c.oversized_reason:
                forced_merge_odd.append((c, fc))

    forced_merge_chunks = [c for c in chunks if "floor-forced merge" in c.oversized_reason]

    print(f"\n{'='*70}")
    print("FENCE INTEGRITY — exhaustive check across all 592 chunks")
    print(f"{'='*70}")
    print(f"\nTotal chunks checked:            {len(chunks)}")
    print(f"Floor-forced-merge chunks:       {len(forced_merge_chunks)}")
    print()
    print(f"Chunks with ODD fence count (torn code): {len(odd)}")

    if odd:
        print("\n  *** TORN CODE BLOCKS — BUG — STOP ***")
        for c, fc in odd:
            short = f"{c.source_file.parent.name}/{c.source_file.name}"
            print(f"  fences={fc}  {short}  |  {c.header_path[:60]}")
    else:
        print("  PASS — all chunks have even fence counts")
        print(f"  Floor-forced-merge population: {len(forced_merge_chunks)} chunks, "
              f"{len(forced_merge_odd)} odd (must be 0) — "
              + ("PASS" if not forced_merge_odd else "FAIL"))


def show_sample_chunks(chunks: list[Chunk]) -> None:
    """
    Show 3 full sample chunks for eyeball coherence check:
      a. Normal prose+code chunk in the 200-512 band
      b. Floor-forced-merge chunk (multiple small sections combined)
      c. The stream-data 2831-token oversized chunk
    """
    def show(label: str, chunk: Chunk) -> None:
        fc = _fence_count(chunk.text)
        short = f"{chunk.source_file.parent.name}/{chunk.source_file.name}"
        print(f"\n{'='*70}")
        print(f"SAMPLE: {label}")
        print(f"{'='*70}")
        print(f"File:         {short}")
        print(f"Header path:  {chunk.header_path}")
        print(f"Tokens:       {chunk.token_count}  |  fence markers: {fc} ({'even OK' if fc % 2 == 0 else 'ODD — BUG'})")
        if chunk.oversized_reason:
            print(f"Oversized:    {chunk.oversized_reason}")
        print(f"\n{'-'*70}")
        # Print full text — no truncation for sample inspection
        print(chunk.text)
        print(f"{'-'*70}")

    # (a) Normal prose+code in 200-512 band with a code block
    normal_with_code = [
        c for c in chunks
        if not c.oversized
        and CHUNK_FLOOR <= c.token_count <= CHUNK_TARGET
        and _fence_count(c.text) >= 2
    ]
    # Pick a mid-corpus example (not the very first file)
    pick_a = normal_with_code[len(normal_with_code) // 2] if normal_with_code else None

    # (b) Floor-forced-merge chunk
    forced_merge = [c for c in chunks if "floor-forced merge" in c.oversized_reason]
    # Pick a smallish one so the merge structure is visible
    pick_b = min(forced_merge, key=lambda c: c.token_count) if forced_merge else None

    # (c) stream-data 2831-token chunk (the known largest outlier)
    pick_c = max(chunks, key=lambda c: c.token_count)

    if pick_a:
        show("(a) Normal prose+code chunk — 200-512 tok band", pick_a)
    if pick_b:
        show("(b) Floor-forced-merge chunk — smallest merged example", pick_b)
    if pick_c:
        show("(c) Oversized chunk — stream-data.md 2831-tok (largest in corpus)", pick_c)


# ─── CLI entry ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    CORPUS = Path(__file__).parent.parent.parent.parent / "data" / "corpus"

    stage = sys.argv[1] if len(sys.argv) > 1 else "1"

    md_files = sorted(CORPUS.rglob("*.md"))
    print(f"Token method: {TOKEN_METHOD}")
    print(f"Corpus: {CORPUS}  ({len(md_files)} files)")

    # ── Stage 1: section parsing only ────────────────────────────────────────
    if stage == "1":
        print("\n" + "=" * 70)
        print("STAGE 1 — Section parsing")
        print("=" * 70)
        all_sections: list[Section] = []
        for f in md_files:
            text = f.read_text(encoding="utf-8", errors="replace")
            all_sections.extend(parse_sections(f, text))
        report_sections(all_sections)

    # ── Stage 2: chunk assembly ───────────────────────────────────────────────
    elif stage == "2":
        print("\n" + "=" * 70)
        print("STAGE 2 — Chunk assembly")
        print("=" * 70)
        chunks = chunk_corpus(CORPUS)
        report_chunks(chunks)

    # ── Stage 3: fence integrity + samples ───────────────────────────────────
    elif stage == "3":
        print("\n" + "=" * 70)
        print("STAGE 3 — Fence integrity + samples")
        print("=" * 70)
        chunks = chunk_corpus(CORPUS)
        report_fence_integrity(chunks)
        show_sample_chunks(chunks)
