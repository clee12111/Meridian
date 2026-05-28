"""Build the document-level routing index from SAC summaries.

Embeds each of the 95 SAC summaries standalone with voyage-4-large
(input_type="document") and saves to data/routing_index.npz.

Usage:
    python scripts/build_routing_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from core.retrieval.routing import build_routing_index

if __name__ == "__main__":
    n = build_routing_index()
    print(f"Built routing index: {n} documents")
