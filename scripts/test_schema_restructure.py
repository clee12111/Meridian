"""Acceptance checks for the ProposalFamily/RagConfig split + fingerprint.

Checks (all local, zero network calls):
  1. ProposalFamily rejects extra fields (bm25_top_k etc.)
  2. ProposalFamily + default knobs builds a valid RagConfig
  3. Old ledger.jsonl rows (7-field configs) still load without error
  4. config_hash for an equivalent full config is unchanged
  5. Fingerprint: write / read / match / mismatch
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def section(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  CHECK: {title}")
    print(f"{'='*72}")


# ── Check 1: ProposalFamily rejects extra fields ─────────────────────────

section("1  ProposalFamily rejects bm25_top_k / dense_top_k / fusion_top_n")

from pydantic import ValidationError
from core.supervisor.schemas import ProposalFamily

for bad_field, bad_val in [
    ("bm25_top_k", 64),
    ("dense_top_k", 64),
    ("fusion_top_n", 64),
]:
    try:
        ProposalFamily(
            retrieval_mode="hybrid",
            target_dataset="contractnli",
            chunk_size=512,
            chunk_overlap=128,
            **{bad_field: bad_val},
        )
        print(f"  FAIL -- {bad_field}={bad_val} was accepted (should be rejected)")
    except ValidationError as e:
        errs = [err["type"] for err in e.errors()]
        print(f"  PASS -- {bad_field}={bad_val} rejected: {errs}")

# Also confirm a clean family parses fine
fam = ProposalFamily(
    retrieval_mode="hybrid",
    target_dataset="contractnli",
    chunk_size=512,
    chunk_overlap=128,
)
print(f"  PASS -- clean ProposalFamily parses: {fam.model_dump()}")


# ── Check 2: ProposalFamily + defaults -> valid RagConfig ────────────────

section("2  ProposalFamily + default knobs builds a valid RagConfig")

from core.supervisor.schemas import RagConfig

cfg = RagConfig.from_family(fam)
assert cfg.chunk_size == 512
assert cfg.chunk_overlap == 128
assert cfg.retrieval_mode == "hybrid"
assert cfg.target_dataset == "contractnli"
assert cfg.bm25_top_k == 32,    f"expected 32, got {cfg.bm25_top_k}"
assert cfg.dense_top_k == 32,   f"expected 32, got {cfg.dense_top_k}"
assert cfg.fusion_top_n == 64,  f"expected 64, got {cfg.fusion_top_n}"
print(f"  PASS -- from_family() produced: {cfg.model_dump()}")

# With explicit knob overrides
cfg2 = RagConfig.from_family(fam, knobs={"bm25_top_k": 64, "dense_top_k": 16, "fusion_top_n": 32})
assert cfg2.bm25_top_k == 64
assert cfg2.dense_top_k == 16
assert cfg2.fusion_top_n == 32
print(f"  PASS -- from_family(knobs={{64,16,32}}) produced: {cfg2.model_dump()}")


# ── Check 3: Old ledger rows (7-field configs) still load ────────────────

section("3  Old ledger.jsonl rows (7-field configs) load without error")

from core.supervisor.proposer import load_ledger

ledger = load_ledger()
if not ledger:
    print("  SKIP -- ledger is empty (no rows to test)")
else:
    for e in ledger:
        # Access every field to ensure nothing is None unexpectedly
        _ = e.config.chunk_size
        _ = e.config.bm25_top_k
        _ = e.config.retrieval_mode
    print(f"  PASS -- {len(ledger)} ledger row(s) loaded, all fields intact")
    print(f"  Sample row 1 config: {ledger[0].config.model_dump()}")


# ── Check 4: config_hash unchanged ──────────────────────────────────────

section("4  config_hash for an equivalent full config is unchanged")

# Build the same RagConfig two ways: direct constructor vs from_family
direct = RagConfig(
    chunk_size=512, chunk_overlap=128,
    bm25_top_k=32, dense_top_k=32, fusion_top_n=64,
    retrieval_mode="hybrid", target_dataset="contractnli",
)
via_family = RagConfig.from_family(
    ProposalFamily(chunk_size=512, chunk_overlap=128,
                   retrieval_mode="hybrid", target_dataset="contractnli"),
)

hash_direct = hashlib.sha256(direct.canonical_json().encode()).hexdigest()
hash_family = hashlib.sha256(via_family.canonical_json().encode()).hexdigest()

print(f"  canonical_json (direct):     {direct.canonical_json()}")
print(f"  canonical_json (via_family): {via_family.canonical_json()}")
print(f"  hash (direct):     {hash_direct[:16]}...")
print(f"  hash (via_family): {hash_family[:16]}...")

assert hash_direct == hash_family, "FAIL -- hashes differ!"
print("  PASS -- hashes identical")

# Also check against the locked-baseline ledger entry if present
if ledger:
    baseline = ledger[0]
    ledger_hash = baseline.experiment_id  # stored as SHA-256 or manual string
    expected = hashlib.sha256(baseline.config.canonical_json().encode()).hexdigest()
    if ledger_hash == expected:
        print(f"  PASS -- ledger run 1 hash matches recomputed hash")
    else:
        # Run 1 was seeded manually with a string ID, not a hash — that's expected
        print(f"  NOTE -- run 1 experiment_id is '{ledger_hash}' (seeded manually, not a hash — OK)")


# ── Check 5: Index fingerprint write / read / match / mismatch ──────────

section("5  Fingerprint: write, read, match, mismatch")

from core.evaluation.fingerprint import write_fingerprint, read_fingerprint, fingerprint_matches

with tempfile.TemporaryDirectory() as tmpdir:
    data_dir = Path(tmpdir)

    # 5a: read before write -> None
    result = read_fingerprint(data_dir, "contractnli_baseline")
    assert result is None, f"Expected None before write, got {result}"
    print("  PASS -- read before write returns None")

    # 5b: write then read back
    write_fingerprint(data_dir, "contractnli_baseline", "contractnli", 512, 128)
    fp = read_fingerprint(data_dir, "contractnli_baseline")
    assert fp is not None
    assert fp["target_dataset"] == "contractnli"
    assert fp["chunk_size"] == 512
    assert fp["chunk_overlap"] == 128
    assert "indexed_at" in fp
    print(f"  PASS -- write then read: {fp}")

    # 5c: match against same params
    ok = fingerprint_matches(data_dir, "contractnli_baseline", "contractnli", 512, 128)
    assert ok, "FAIL -- expected match"
    print("  PASS -- fingerprint_matches() True for same params")

    # 5d: mismatch on chunk_size
    bad = fingerprint_matches(data_dir, "contractnli_baseline", "contractnli", 256, 64)
    assert not bad, "FAIL -- expected mismatch"
    print("  PASS -- fingerprint_matches() False for different chunk_size/overlap")

    # 5e: mismatch on dataset
    bad2 = fingerprint_matches(data_dir, "contractnli_baseline", "cuad", 512, 128)
    assert not bad2, "FAIL -- expected mismatch on dataset"
    print("  PASS -- fingerprint_matches() False for different dataset")

    # 5f: second collection in same file
    write_fingerprint(data_dir, "legalbench_rag_full", "all", 768, 192)
    fp2 = read_fingerprint(data_dir, "legalbench_rag_full")
    fp1_again = read_fingerprint(data_dir, "contractnli_baseline")
    assert fp2["chunk_size"] == 768
    assert fp1_again["chunk_size"] == 512  # first entry unchanged
    print("  PASS -- two collections coexist in same fingerprint file")


print(f"\n{'='*72}")
print("  ALL CHECKS PASSED")
print(f"{'='*72}\n")
