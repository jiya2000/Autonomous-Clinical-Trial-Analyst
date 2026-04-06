from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import os

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from llama_parse import LlamaParse

from app.config import settings
from app.agents import EMBED_DIM, embed_text


def get_trial_metadata(nct_id: str) -> Dict[str, Any]:
    """Fetch study metadata from ClinicalTrials.gov API v2.

    This follows the structure described in the spec. In production you may
    need to adjust field paths as the API evolves.
    """

    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return {}

    data = resp.json()
    protocol = data.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})

    return {
        "title": identification.get("officialTitle"),
        "status": status.get("overallStatus"),
        "phase": design.get("phases"),
    }


def chunk_markdown(markdown_text: str) -> List[Dict[str, Any]]:
    """Very simple hierarchical chunking based on Markdown headers.

    Each top-level or second-level section becomes a chunk. You can extend
    this later to handle eligibility vs schedule-of-assessments etc.
    """

    lines = markdown_text.splitlines()
    chunks: List[Dict[str, Any]] = []
    current_header: str | None = None
    current_content: List[str] = []

    def _flush():
        if current_content:
            chunks.append(
                {
                    "header": current_header or "",
                    "text": "\n".join(current_content).strip(),
                }
            )

    for line in lines:
        if line.startswith("#"):
            _flush()
            current_header = line.lstrip("# ")
            current_content = []
        else:
            current_content.append(line)

    _flush()
    return [c for c in chunks if c["text"]]


def ensure_multimodal_collection(client: QdrantClient, collection_name: str) -> None:
    """Create the Qdrant collection for multimodal clinical protocols if missing."""

    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text_vec": qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
            # Placeholder dimension for images; adjust when you plug CLIP/SigLIP.
            "image_vec": qmodels.VectorParams(size=512, distance=qmodels.Distance.COSINE),
        },
    )


def index_protocol_markdown(
    markdown_text: str,
    nct_id: str | None = None,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
) -> None:
    """Index a parsed protocol (Markdown) into Qdrant.

    This currently indexes only text chunks into the `text_vec` vector space.
    Image extraction and embeddings can be added later.
    """

    if client is None:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    if collection_name is None:
        collection_name = settings.clinical_collection

    ensure_multimodal_collection(client, collection_name)

    metadata: Dict[str, Any] = {}
    if nct_id:
        metadata = get_trial_metadata(nct_id)

    chunks = chunk_markdown(markdown_text)

    points: List[qmodels.PointStruct] = []
    for idx, ch in enumerate(chunks):
        text = ch["text"]
        payload: Dict[str, Any] = {
            "section_header": ch["header"],
            "content_type": "text",
            "text": text,
        }
        if nct_id:
            payload["nct_id"] = nct_id
        payload.update({k: v for k, v in metadata.items() if v is not None})

        vec = embed_text(text)
        points.append(
            qmodels.PointStruct(
                id=idx,
                vector={"text_vec": vec},
                payload=payload,
            )
        )

    if points:
        client.upsert(collection_name=collection_name, points=points)


def ingest_markdown_file(path: str | Path, nct_id: str | None = None) -> None:
    """Convenience wrapper to ingest a local Markdown file into Qdrant."""

    p = Path(path)
    markdown_text = p.read_text(encoding="utf-8")
    index_protocol_markdown(markdown_text=markdown_text, nct_id=nct_id)


def parse_protocol_pdf_to_markdown(pdf_path: str | Path, extra_instructions: str | None = None) -> str:
    """Parse a clinical trial PDF into Markdown using LlamaParse in multimodal mode.

    Expects LLAMA_CLOUD_API_KEY in the environment for authentication.
    The parsing instructions are tailored to extract eligibility criteria and
    schedule-of-assessments tables as described in the architecture spec.
    """

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY environment variable is not set")

    base_instructions = (
        "Extract all eligibility criteria as a Markdown list. "
        "Convert the Schedule of Assessments table into a structured JSON "
        "representation embedded in the Markdown. Preserve headings and "
        "section structure from the original protocol."
    )
    if extra_instructions:
        parsing_instruction = base_instructions + "\n" + extra_instructions
    else:
        parsing_instruction = base_instructions

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        use_vendor_multimodal_model=True,
        parsing_instruction=parsing_instruction,
    )

    pdf_path = Path(pdf_path)
    docs = parser.load_data(str(pdf_path))
    markdown_chunks = [d.text for d in docs if getattr(d, "text", None)]
    return "\n\n".join(markdown_chunks)


def ingest_pdf_with_llamaparse(pdf_path: str | Path, nct_id: str | None = None) -> None:
    """End-to-end helper: PDF -> LlamaParse -> Markdown -> Qdrant indexing."""

    markdown_text = parse_protocol_pdf_to_markdown(pdf_path)
    index_protocol_markdown(markdown_text=markdown_text, nct_id=nct_id)
