from __future__ import annotations

"""Minimal end-to-end demo: PDF -> LlamaParse -> Qdrant -> retrieval.

Run from inside the backend container or a venv with requirements installed.

Example (PowerShell):
    $env:LLAMA_CLOUD_API_KEY = "..."
    python demo_ingest_and_search.py data/protocol1.pdf NCT01234567 "What are the key inclusion criteria?"
"""

import sys
from pathlib import Path

from qdrant_client import QdrantClient

from app.config import settings
from app.ingestion.clinical_ingest import ingest_pdf_with_llamaparse
from app.agents import qdrant_search


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: python demo_ingest_and_search.py <protocol.pdf> <NCT_ID> [query]",
        )
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    nct_id = sys.argv[2]
    query = sys.argv[3] if len(sys.argv) > 3 else "Summarize the eligibility criteria."

    if not pdf_path.exists():
        raise SystemExit(f"PDF file not found: {pdf_path}")

    print(f"Ingesting PDF {pdf_path} for trial {nct_id} via LlamaParse...")
    ingest_pdf_with_llamaparse(pdf_path, nct_id=nct_id)
    print("Ingestion complete. Running a sample retrieval query...\n")

    # Ensure Qdrant is reachable
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    _ = client.get_collections()

    docs = qdrant_search(query, limit=5)
    print(f"Top {len(docs)} retrieved chunks for query: {query!r}\n")
    for i, d in enumerate(docs, start=1):
        payload = d.get("payload", {})
        header = payload.get("section_header", "<no header>")
        text = (payload.get("text") or "").replace("\n", " ")
        print(f"[{i}] score={d.get('score'):.3f} section={header}")
        print(text[:400] + ("..." if len(text) > 400 else ""))
        print("-" * 80)


if __name__ == "__main__":
    main()
