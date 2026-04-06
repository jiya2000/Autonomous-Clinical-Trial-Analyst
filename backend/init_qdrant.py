#!/usr/bin/env python3
"""Initialize Qdrant with sample clinical trial data."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib
import math

# Sample clinical trial documents
SAMPLE_DOCUMENTS = [
    {
        "id": 1,
        "title": "Phase 3 Randomized Trial of COVID-19 Vaccine",
        "content": "This Phase 3 randomized controlled trial evaluates the efficacy and safety of a novel COVID-19 vaccine. The study involves 30,000 participants aged 18 and above. Primary endpoints include virologically confirmed COVID-19 and severe COVID-19. Participants are randomized 1:1 to receive either vaccine or placebo. The trial is conducted across 150 sites in multiple countries. Safety monitoring includes adverse event tracking and laboratory assessments.",
    },
    {
        "id": 2,
        "title": "Clinical Trial Protocol for Type 2 Diabetes Treatment",
        "content": "A 52-week, multicenter, randomized, double-blind, placebo-controlled Phase 3 trial of a novel oral antidiabetic agent in patients with type 2 diabetes mellitus. Approximately 2,000 patients will be enrolled. Inclusion criteria: age 18-75 years, HbA1c 7.5%-11%. Primary objective: change in HbA1c from baseline. Secondary objectives include weight change and lipid profile changes. Visits occur at baseline, weeks 4, 12, 26, 39, and 52.",
    },
    {
        "id": 3,
        "title": "Cancer Immunotherapy Clinical Trial Protocol",
        "content": "An open-label, Phase 2 trial of a PD-1 inhibitor combined with chemotherapy for advanced non-small cell lung cancer. The study will enroll 200 patients with PD-L1 expression. Treatment consists of 6 cycles of chemotherapy plus concurrent immunotherapy, followed by immunotherapy continuation. Primary endpoint: overall response rate. Patients are assessed for response every 8 weeks using RECIST 1.1 criteria. Adverse events are graded using CTCAE v5.0.",
    },
    {
        "id": 4,
        "title": "Alzheimer's Disease Biomarker Study",
        "content": "A longitudinal observational study investigating biomarkers of Alzheimer's disease in cognitively normal older adults. The study will recruit 500 participants aged 65-85 years. Comprehensive cognitive testing, MRI imaging, and cerebrospinal fluid analysis will be performed annually for 5 years. Inclusion criteria: MMSE score ≥26, no signs of cognitive impairment. Primary outcome: rate of cognitive decline over 5 years.",
    },
    {
        "id": 5,
        "title": "Hypertension Management Trial Protocol",
        "content": "A pragmatic, cluster-randomized trial comparing intensive versus standard blood pressure control in high-risk hypertensive patients. The trial will involve 100 primary care practices and approximately 4,000 patients with hypertension and additional cardiovascular risk factors. Patients are randomized within clusters to achieve systolic BP targets of <120 mmHg (intensive) or <140 mmHg (standard). The primary outcome is a composite of cardiovascular events over 3 years of follow-up.",
    },
    {
        "id": 6,
        "title": "Depression Treatment Clinical Trial",
        "content": "An 8-week, double-blind, placebo-controlled trial of a novel antidepressant in adults with major depressive disorder. The study will enroll 400 participants (18-75 years) with moderate to severe depression. Participants receive either active drug or placebo daily. Primary outcome: change in MADRS score from baseline to week 8. Secondary outcomes include response and remission rates. Safety assessments include vital signs and laboratory tests at weeks 0, 2, 4, and 8.",
    },
    {
        "id": 7,
        "title": "Rare Disease Gene Therapy Study",
        "content": "A Phase 1/2 open-label study of an AAV-based gene therapy for a rare genetic disorder. The trial will enroll up to 15 patients with confirmed genetic mutations. Participants receive a single intravenous infusion of gene therapy. Primary objectives: safety and tolerability. Secondary objectives: gene expression and clinical outcomes measured over 24 months. Extensive genetic monitoring and functional assessments are included.",
    },
    {
        "id": 8,
        "title": "Arthritis Pain Management Trial",
        "content": "A 12-week, randomized, double-blind comparison of a novel pain management approach versus standard therapy for moderate to severe osteoarthritis. The study will enroll 300 participants with knee osteoarthritis. Participants are assessed using WOMAC scores, pain scales, and functional measures at baseline and weeks 4, 8, and 12. Safety monitoring includes liver and kidney function tests. The trial allows previous pain medication use to continue.",
    },
]

def embed_text(text: str, dim: int = 128) -> list[float]:
    """Create a hash-based embedding (same as backend)."""
    # Create hash of text
    hash_bytes = hashlib.sha256(text.encode()).digest()
    
    # Create embedding by chunking hash
    embedding = []
    for i in range(dim):
        chunk_size = len(hash_bytes) // dim
        start = i * chunk_size
        end = start + chunk_size
        byte_val = sum(hash_bytes[start:end]) % 256
        embedding.append(byte_val / 256.0)
    
    # L2 normalize
    norm = math.sqrt(sum(x ** 2 for x in embedding))
    if norm > 0:
        embedding = [x / norm for x in embedding]
    
    return embedding

def init_qdrant():
    """Initialize Qdrant with sample documents."""
    # Connect to Qdrant
    client = QdrantClient(url="http://localhost:6333")
    
    collection_name = "clinical_protocols"
    vector_size = 128
    
    # Check if collection exists
    try:
        client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists. Skipping creation.")
        return
    except Exception:
        print(f"Creating collection {collection_name}...")
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        named_vectors={
            "text_vec": VectorParams(size=vector_size, distance=Distance.COSINE)
        }
    )
    
    # Insert sample documents
    points = []
    for doc in SAMPLE_DOCUMENTS:
        combined_text = f"{doc['title']} {doc['content']}"
        embedding = embed_text(combined_text)
        
        point = PointStruct(
            id=doc["id"],
            vector={
                "text_vec": embedding
            },
            payload={
                "title": doc["title"],
                "content": doc["content"],
            }
        )
        points.append(point)
    
    # Upload points to Qdrant
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"✓ Initialized {collection_name} with {len(points)} documents")

if __name__ == "__main__":
    init_qdrant()
