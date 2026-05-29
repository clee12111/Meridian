"""Build the document-level routing index from SAC summaries.

Embeds each of the 95 SAC summaries standalone and saves to disk.

Usage:
    python scripts/build_routing_index.py
    python scripts/build_routing_index.py --model voyage-4 --output data/routing_index_v4.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from core.retrieval.routing import build_routing_index

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build routing index")
    p.add_argument("--model", type=str, default=None,
                   help="Voyage model (default: from env or voyage-4-large)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output path (default: data/routing_index.npz)")
    args = p.parse_args()

    n = build_routing_index(output_path=args.output, model=args.model)
    print(f"Built routing index: {n} documents")
