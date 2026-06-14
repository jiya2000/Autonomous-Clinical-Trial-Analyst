from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.agents import EMBED_DIM, embed_text


def _load_diabetes_trials() -> List[Dict[str, Any]]:
    """Load studies from the diabetes_trials.json file.

    The fetch script saves a JSON object of the form:
        { "studies": [ { ... }, ... ] }
    at the project root.
    """

    # Project root is three levels up from this file: backend/app/ingestion -> project
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "diabetes_trials.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}. Run app.data_get.dataget first.")

    import json

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    studies = data.get("studies", [])
    if not isinstance(studies, list):
        raise ValueError("Unexpected JSON structure: 'studies' is not a list")

    return studies


import spacy

# Load SciSpaCy model for Medical NER (fallback to None if not installed)
try:
    nlp = spacy.load("en_core_sci_sm")
except OSError:
    print("SciSpaCy model 'en_core_sci_sm' not found. Skipping Medical NER extraction.")
    nlp = None

def _trial_to_payload(study: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact text payload for Qdrant from a v2 study object."""

    protocol = study.get("protocolSection", {}) or {}

    ident = protocol.get("identificationModule", {}) or {}
    desc = protocol.get("descriptionModule", {}) or {}
    elig = protocol.get("eligibilityModule", {}) or {}
    cond = protocol.get("conditionsModule", {}) or {}

    nct_id = ident.get("nctId") or study.get("nctId")
    title = ident.get("officialTitle") or ident.get("briefTitle") or (nct_id or "Untitled study")

    brief_summary = desc.get("briefSummary") or ""
    detailed_description = desc.get("detailedDescription") or ""
    eligibility = elig.get("eligibilityCriteria") or ""
    conditions = cond.get("conditions") or []

    parts: List[str] = []
    if nct_id:
        parts.append(f"NCT ID: {nct_id}")
    if conditions:
        parts.append("Conditions: " + ", ".join(conditions))
    if brief_summary:
        parts.append("Brief summary:\n" + brief_summary)
    if detailed_description:
        parts.append("Detailed description:\n" + detailed_description)
    if eligibility:
        parts.append("Eligibility criteria:\n" + eligibility)

    text = "\n\n".join(parts).strip()

    # Extract medical entities using SciSpaCy
    medical_entities = []
    if nlp and text:
        # Process up to 2000 chars to keep ingestion fast
        doc = nlp(text[:2000])
        entities = list(set([ent.text.lower() for ent in doc.ents]))
        # Filter for reasonable length entities
        medical_entities = [e for e in entities if 3 <= len(e) <= 50]

    return {
        "nct_id": nct_id,
        "title": title,
        "content": text or title,
        "conditions": conditions,
        "extracted_entities": medical_entities,
    }


def ingest_diabetes_trials() -> None:
    """Embed and upsert diabetes trials into the clinical_protocols collection.

    This uses the same simple hash-based embeddings and payload schema
    (title + content) that the current agents expect.
    """

    studies = _load_diabetes_trials()

    if not studies:
        print("No studies found in diabetes_trials.json")
        return

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    # Ensure collection exists with the simple single-vector schema used in agents.py
    existing = [c.name for c in client.get_collections().collections]
    if settings.clinical_collection not in existing:
        client.create_collection(
            collection_name=settings.clinical_collection,
            vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
        )

    points: List[qmodels.PointStruct] = []
    # Start IDs at 1000 to avoid clashing with any existing sample docs
    for idx, study in enumerate(studies, start=1000):
        payload = _trial_to_payload(study)
        text = payload.get("content", "")
        if not text:
            continue

        vec = embed_text(text)

        points.append(
            qmodels.PointStruct(
                id=idx,
                vector=vec,
                payload=payload,
            )
        )

    if not points:
        print("No usable study texts to index.")
        return

    client.upsert(collection_name=settings.clinical_collection, points=points)
    print(f"Upserted {len(points)} diabetes trial protocols into '{settings.clinical_collection}'.")


if __name__ == "__main__":
    ingest_diabetes_trials()
