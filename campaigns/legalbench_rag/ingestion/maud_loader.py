"""MAUD ingestion from raw MAUD dataset.

Downloads the raw MAUD data from GitHub, generates document titles
using gpt-4o-mini (one-time, cached), and builds queries with
character spans using unidecode + sourcemap matching.

Ported from legalbenchrag/generate/generate_maud.py.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import cast
from urllib.request import urlopen

import pandas as pd
from unidecode import unidecode

from campaigns.legalbench_rag.ingestion.common import (
    IngestionError,
    build_corpus_df,
    sample_mini,
    verify_span_integrity,
    write_benchmark_json,
)

MAUD_URL = "https://github.com/TheAtticusProject/maud/archive/refs/heads/main.zip"
DATASET_NAME = "maud"

# Mapping from MAUD CSV column names to query text.
# None = skip (low annotation quality).
COLUMN_TO_QUERY: dict[str, str | None] = {
    "Type of Consideration": "What is the Type of Consideration",
    "Accuracy of Target R&W Closing Condition": "Information about the Closing Condition: Accuracy of Target's Representations and Warranties",
    "Compliance with Covenant Closing Condition": "Information about the Closing Condition: Compliance with Covenants",
    "Absence of Litigation Closing Condition": "Information about the Closing Condition: No Litigation clause",
    'Agreement includes a "Back-Door" MAE': None,
    '"No MAE" R&W Made as of a Specified Date': "What is the Target's Representation & Warranty of No Material Adverse Effect, with regards to some specified date",
    "MAE Definition": 'What is the Definition of "Material Adverse Effect"',
    "Knowledge Definition": 'What is the Definition of "Knowledge"',
    "No-Shop": "Where is the No-Shop Clause",
    "Fiduciary exception:  Board determination (no-shop)": "What about the Fiduciary exception to the No-Shop Clause",
    "Fiduciary exception to COR covenant": None,
    "Agreement provides for matching rights in connection with COR": None,
    "Superior Offer Definition": 'What is the Definition of "Superior Proposal"',
    "Intervening Event Definition": 'What is the Definition of "Interveining Event"',
    "FTR Triggers": "Information about the Fiduciary Termination Right Triggers for termination",
    "Limitations on FTR Exercise": None,
    "Agreement provides for matching rights in connection with FTR": None,
    "Tail Period & Acquisition Proposal Details": "Is there a Tail provision for acquisition proposals",
    "Breach of No Shop": "What happens during a Breach of No-Shop clause",
    "Breach of Meeting Covenant": "What happens during a Breach of Shareholder Meeting Covenant",
    "Ordinary course covenant": "What are the Ordinary course of business covenants",
    "Negative interim operating covenant": None,
    "General Antitrust Efforts Standard": "Where is the Closing Conditions: Regulatory Approvals clause",
    "Limitations on Antitrust Efforts": "I want information about the Limitations on Antitrust Efforts",
    "Specific Performance": "Where is the Specific Performance clause",
}

TITLE_SYSTEM_PROMPT = """You are given a long M&A contract. Create a succinct title.
You MUST mention the relevant companies involved and the contract purpose.

Think about which party is the "Parent" / acquirer and which is the "Target".
If there is a parent/acquiring body, use: "Acquisition Agreement between Parent \\"X\\" and Target \\"Y\\""
If there is no clear parent, use: "Merger Agreement between \\"X\\" and \\"Y\\""
If you cannot identify the parties, say "Arbitrary M&A Agreement"

Return JSON only (no markdown fences):
{"thoughts": ["..."], "title": "..."}"""


def _sort_and_merge_spans(
    spans: list[tuple[int, int]], max_bridge_gap: int = 0
) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    merged = [spans[0]]
    for start, end in spans[1:]:
        if start <= merged[-1][1] + max_bridge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _generate_title(filename: str, text: str, openai_client) -> str:
    """Generate a document title using gpt-4o-mini."""
    truncated = text[:10000] + text[-10000:] if len(text) > 30000 else text
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": TITLE_SYSTEM_PROMPT},
            {"role": "user", "content": f"FILENAME: {filename}.txt\nCONTENT:\n{truncated}"},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    content = resp.choices[0].message.content.strip()
    if content.endswith(",\n}"):
        content = content[:-3] + "\n}"
    parsed = json.loads(content)
    return parsed["title"]


def _load_or_generate_titles(
    data_dir: Path,
    raw_dir: Path,
    contract_names: dict[str, str],
) -> dict[str, str]:
    """Load cached titles or generate with gpt-4o-mini. Returns {filename: title}."""
    cache_path = data_dir / "maud_titles.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if set(cached.keys()) >= set(contract_names.keys()):
            print(f"  Loaded {len(cached)} cached MAUD titles")
            return cached

    from openai import OpenAI
    client = OpenAI()
    titles: dict[str, str] = {}

    for i, (filename, contract_name) in enumerate(sorted(contract_names.items())):
        contract_path = raw_dir / "data" / "contracts" / f"{contract_name}.txt"
        with open(contract_path, encoding="utf-8") as f:
            text = f.read()
        title = _generate_title(filename, text, client)
        titles[filename] = title
        if (i + 1) % 25 == 0 or (i + 1) == len(contract_names):
            print(f"  Generated {i + 1}/{len(contract_names)} MAUD titles")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(titles, f, indent=2)
    print(f"  Cached {len(titles)} titles to {cache_path}")
    return titles


def _get_contract_name_from_mae(
    df_testcases: pd.DataFrame, mae_definition: str
) -> str:
    """Deduce contract_name from MAE Definition text."""
    mae_matches = (
        df_testcases[
            (df_testcases["text_type"] == "MAE Definition")
            & (df_testcases["text"] == mae_definition)
            & (df_testcases["contract_name"] != "<RARE_ANSWERS>")
        ]["contract_name"]
        .unique()
        .tolist()
    )
    if len(mae_matches) != 1:
        return ""
    return str(mae_matches[0]).replace(".pdf", "")


def _download_and_parse(
    data_dir: Path,
) -> tuple[dict[str, str], list[dict]]:
    """Download MAUD data, generate titles, build queries with spans."""
    raw_dir = data_dir / "raw_data" / "maud" / "maud-main"

    if not raw_dir.exists():
        dl_dir = data_dir / "raw_data" / "maud"
        dl_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading MAUD from GitHub...")
        resp = urlopen(MAUD_URL)  # noqa: S310
        with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
            zf.extractall(dl_dir)

    # Unzip nested data.zip if needed
    data_zip = raw_dir / "data.zip"
    if data_zip.exists() and not (raw_dir / "data").exists():
        with zipfile.ZipFile(data_zip) as zf:
            zf.extractall(raw_dir)

    # Load test case CSVs for contract_name deduction
    df_testcases = pd.concat([
        pd.read_csv(raw_dir / "data" / f"MAUD_{split}.csv")
        for split in ("dev", "test", "train")
        if (raw_dir / "data" / f"MAUD_{split}.csv").exists()
    ])

    # Read main.csv
    df_main = pd.read_csv(raw_dir / "data" / "raw" / "main.csv")

    # Build filename -> contract_name mapping
    contract_names: dict[str, str] = {}
    row_data: list[dict] = []

    for _, row in df_main.iterrows():
        raw_filename = str(row["Filename"])
        if raw_filename == "Soliton, Inc._AbbVie Inc..pdf":
            continue
        if not raw_filename.endswith(".pdf"):
            continue
        # Sanitize: remove newlines, pipes, and other illegal filename chars
        filename = raw_filename.replace("\n", "_").replace("|", "_")[:-4]
        # Collapse multiple underscores
        filename = re.sub(r"_+", "_", filename).strip("_")

        contract_name = _get_contract_name_from_mae(
            df_testcases, str(row["MAE Definition"])
        )
        if not contract_name:
            continue

        contract_names[filename] = contract_name
        row_data.append({"filename": filename, "contract_name": contract_name, "row": row})

    # Generate or load titles
    titles = _load_or_generate_titles(data_dir, raw_dir, contract_names)

    # Process rows: build spans using unidecode + sourcemap
    queries: list[dict] = []
    used_contracts: dict[str, str] = {}
    counter = 0

    for entry in row_data:
        filename = entry["filename"]
        contract_name = entry["contract_name"]
        row = entry["row"]

        if filename not in titles:
            continue
        title = titles[filename]
        if "arbitrary" in title.lower():
            continue

        # Read contract text and build sourcemap
        contract_path = raw_dir / "data" / "contracts" / f"{contract_name}.txt"
        with open(contract_path, encoding="utf-8") as f:
            total_text_raw = f.read()

        # Build whitespace-stripped text with sourcemap to original indices
        stripped_text = ""
        sourcemap: list[int] = []
        for i, c in enumerate(total_text_raw):
            if ord(c) >= 0x80:
                for c0 in unidecode(c):
                    if not c0.isspace():
                        stripped_text += c0
                        sourcemap.append(i)
            elif not c.isspace():
                stripped_text += c
                sourcemap.append(i)

        # Process each column annotation
        for column_name, query_text in COLUMN_TO_QUERY.items():
            if query_text is None:
                continue

            column_value = str(row.get(column_name, ""))
            column_value = unidecode(column_value)
            column_value = re.sub(r"\s+", "", column_value)

            matching_parts = re.split(r"\s*<omitted>\s*", column_value)
            matching_parts[-1] = re.sub(
                r"\(Pages?\s*[\d-]+\)\s*$", "", matching_parts[-1]
            )

            # Match spans in stripped text
            spans: list[tuple[int, int]] = []
            current_idx = 0
            failed = False
            for part in matching_parts:
                idx = stripped_text.find(part, current_idx)
                if idx == -1:
                    failed = True
                    break
                current_idx = idx + len(part)
                # Reject ambiguous matches
                if stripped_text.find(part, current_idx) != -1:
                    failed = True
                    break
                # Map back to original text indices via sourcemap
                spans.append((sourcemap[idx], sourcemap[idx + len(part) - 1] + 1))

            if failed:
                continue

            spans = _sort_and_merge_spans(spans)
            if not spans:
                continue

            used_contracts[contract_name] = filename
            queries.append(
                {
                    "query_id": f"maud-{counter:04d}",
                    "query": f"Consider the {title}; {query_text}",
                    "snippets": [
                        {"file_path": f"maud/{filename}.txt", "span": list(s)}
                        for s in spans
                    ],
                }
            )
            counter += 1

    # Write corpus files
    corpus_dir = data_dir / "corpus" / DATASET_NAME
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    raw_docs: dict[str, str] = {}
    for contract_name, filename in used_contracts.items():
        src = raw_dir / "data" / "contracts" / f"{contract_name}.txt"
        with open(src, encoding="utf-8") as f:
            text = f.read()
        with open(corpus_dir / f"{filename}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        raw_docs[f"maud/{filename}.txt"] = text

    return raw_docs, queries


def ingest_maud(
    data_dir: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    max_queries: int | None = 194,
) -> tuple[pd.DataFrame, list[dict]]:
    """Full ingestion pipeline for MAUD."""
    data_dir = Path(data_dir)

    raw_docs, queries = _download_and_parse(data_dir)

    lookup = dict(raw_docs)
    verify_span_integrity(queries, lookup)
    print(f"  Span integrity verified for all {len(queries)} queries")

    if max_queries is not None:
        queries = sample_mini(queries, max_queries)

    write_benchmark_json(queries, data_dir / "benchmarks", DATASET_NAME)

    corpus_df = build_corpus_df(raw_docs, chunk_size, chunk_overlap, DATASET_NAME)
    corpus_df.to_parquet(data_dir / "corpus_maud.parquet", index=False)
    print(
        f"Ingested {len(raw_docs)} documents, "
        f"{len(corpus_df)} chunks, {len(queries)} queries"
    )

    return corpus_df, queries
