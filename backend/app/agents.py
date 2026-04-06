from __future__ import annotations

from typing import Annotated, List, Literal, TypedDict, Dict, Any
import operator
import math
import logging

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .config import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)


# --- Shared Agent State -----------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_node: str
    current_plan: List[str]
    retrieved_docs: List[Dict[str, Any]]
    final_answer: str
    trace: Annotated[List[str], operator.add]


# --- LLM and Tools Setup ----------------------------------------------------

llm = ChatOpenAI(
    model="tinyllama",
    base_url=settings.vllm_url,
    api_key="EMPTY",
)

qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

EMBED_DIM = 128

SAMPLE_DOCUMENTS = [
    {"id": 1, "title": "Phase 3 Randomized Trial of COVID-19 Vaccine", "content": "This Phase 3 randomized controlled trial evaluates the efficacy and safety of a novel COVID-19 vaccine. Participants are randomized 1:1 to receive either vaccine or placebo."},
    {"id": 2, "title": "Type 2 Diabetes Treatment Trial", "content": "A Phase 3 trial of a novel oral antidiabetic agent. Inclusion criteria: age 18-75 years, HbA1c 7.5%-11%. Primary objective: change in HbA1c from baseline."},
    {"id": 3, "title": "Cancer Immunotherapy Clinical Trial", "content": "A Phase 2 trial of a PD-1 inhibitor combined with chemotherapy for advanced non-small cell lung cancer. Primary endpoint: overall response rate."},
    {"id": 4, "title": "Alzheimer's Disease Biomarker Study", "content": "A longitudinal observational study recruiting 500 participants aged 65-85 years. Primary outcome: rate of cognitive decline over 5 years."},
    {"id": 5, "title": "Hypertension Management Trial", "content": "A cluster-randomized trial comparing intensive versus standard blood pressure control. Systolic BP targets: <120 mmHg (intensive) or <140 mmHg (standard)."},
    {"id": 6, "title": "Depression Treatment Clinical Trial", "content": "An 8-week double-blind placebo-controlled trial of a novel antidepressant. Primary outcome: change in MADRS score from baseline to week 8."},
    {"id": 7, "title": "Rare Disease Gene Therapy Study", "content": "A Phase 1/2 open-label study of an AAV-based gene therapy. Enroll up to 15 patients with confirmed genetic mutations. Primary objectives: safety and tolerability."},
    {"id": 8, "title": "Arthritis Pain Management Trial", "content": "A 12-week randomized double-blind comparison for moderate to severe osteoarthritis. Enroll 300 participants with knee osteoarthritis."},
]


def embed_text(text: str) -> List[float]:
    """Create hash-based embedding."""
    vec = [0.0] * EMBED_DIM
    tokens = text.lower().split()
    for tok in tokens:
        idx = hash(tok) % EMBED_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [(v / norm) for v in vec]


def qdrant_search(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Search Qdrant collection."""
    vector = embed_text(query)
    results = qdrant_client.search(
        collection_name=settings.clinical_collection,
        query_vector=vector,
        limit=limit,
    )
    docs = []
    for point in results:
        docs.append({
            "id": point.id,
            "score": point.score,
            "payload": point.payload or {},
        })
    return docs


# --- Agent Nodes --------------------------------------------------------


def supervisor_node(state: AgentState) -> dict:
    """Route to next agent."""
    messages = state.get("messages") or []
    retrieved = state.get("retrieved_docs") or []
    trace = state.get("trace") or []
    
    # Simple routing: if we have docs, analyze; otherwise retrieve
    if retrieved:
        trace.append("Supervisor: Retrieved context available, delegating to Analysis Agent.")
        return {"next_node": "AnalysisAgent", "trace": trace}
    else:
        user_query = getattr(messages[-1], 'content', '') if messages else ''
        if user_query:
            trace.append(f"Supervisor: Received question — '{user_query[:80]}...' — calling Retrieval Agent.")
        else:
            trace.append("Supervisor: No question text found, still calling Retrieval Agent.")
        return {"next_node": "RetrievalAgent", "trace": trace}


def retrieval_node(state: AgentState) -> dict:
    """Retrieve relevant documents."""
    messages = state.get("messages") or []
    trace = state.get("trace") or []
    
    if not messages:
        trace.append("Retrieval Agent: No user message found; skipping search and passing empty context.")
        return {"retrieved_docs": [], "next_node": "AnalysisAgent", "trace": trace}
    
    # Get query from latest message
    latest = messages[-1]
    query = getattr(latest, 'content', str(latest)) if latest else ""
    
    try:
        docs = qdrant_search(query) if query else []
        if docs:
            top = docs[0]
            title = (top.get("payload") or {}).get("title", "Untitled protocol")
            trace.append(
                f"Retrieval Agent: Found {len(docs)} relevant protocol snippet(s); top match is '{title}'."
            )
        else:
            trace.append("Retrieval Agent: No relevant protocol snippets found for this question.")
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        trace.append(f"Retrieval Agent: Encountered an error while searching the protocol library: {e}.")
        docs = []
    
    return {"retrieved_docs": docs, "next_node": "AnalysisAgent", "trace": trace}


def analysis_node(state: AgentState) -> dict:
    """Analyze documents and generate answer."""
    messages = state.get("messages") or []
    docs = state.get("retrieved_docs") or []
    trace = state.get("trace") or []
    
    # Build context
    context_lines = []
    for doc in docs:
        payload = doc.get("payload") or {}
        content = payload.get("content", "")
        if content:
            context_lines.append(content)
    
    context = "\n---\n".join(context_lines[:5]) if context_lines else "No relevant protocols found."
    if context_lines:
        trace.append(f"Analysis Agent: Using {min(len(context_lines), 5)} protocol snippet(s) as clinical context.")
    else:
        trace.append("Analysis Agent: No matching protocol context; answering based on general clinical knowledge.")
    
    # Create analysis messages
    analysis_messages = [
        SystemMessage(content="You are a clinical trial analyst. Answer using only the provided protocol context."),
        SystemMessage(content=f"Protocols:\n{context}"),
    ]
    
    # Add all previous messages
    for msg in messages:
        if msg:
            analysis_messages.append(msg)
    
    try:
        logger.info(f"=== ANALYSIS NODE: Invoking LLM with {len(analysis_messages)} messages ===")
        answer = llm.invoke(analysis_messages)
        logger.info(f"=== LLM returned object type: {type(answer)} ===")
        if answer and hasattr(answer, 'content'):
            answer_text = answer.content
            logger.info(f"=== LLM ANSWER: {answer_text[:300]}... ===")
            trace.append("Analysis Agent: Generated a draft answer based on the retrieved context and question.")
            return {
                "final_answer": answer_text,
                "next_node": "FINISH",
                "trace": trace,
            }
        else:
            logger.error(f"=== LLM returned object without content: {answer} ===")
    except Exception as e:
        logger.error(f"=== Analysis error: {str(e)} ===", exc_info=True)
        return {
            "final_answer": f"Error: {str(e)}",
            "next_node": "FINISH",
            "trace": trace + [
                "Analysis Agent: Failed to generate an answer due to an internal error."
            ],
        }
    
    logger.error("=== Analysis node: No answer returned ===")
    trace.append("Analysis Agent: No answer could be generated from the current context.")
    return {"final_answer": "[No answer generated]", "next_node": "FINISH", "trace": trace}


# --- Workflow Builder -------------------------------------------------------


def build_workflow() -> StateGraph:
    """Build the agent workflow."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("RetrievalAgent", retrieval_node)
    workflow.add_node("AnalysisAgent", analysis_node)
    
    workflow.set_entry_point("Supervisor")
    
    def route(state: AgentState) -> str:
        next_node = state.get("next_node", "FINISH")
        return END if next_node == "FINISH" else next_node
    
    workflow.add_conditional_edges("Supervisor", route)
    workflow.add_edge("RetrievalAgent", "AnalysisAgent")
    workflow.add_edge("AnalysisAgent", END)
    
    return workflow


# --- Qdrant Initialization --------------------------------------------------


def _init_qdrant():
    """Initialize Qdrant collection."""
    collection = settings.clinical_collection
    
    try:
        qdrant_client.get_collection(collection)
        print(f"✓ Qdrant collection '{collection}' exists")
        return
    except:
        pass
    
    try:
        print(f"Creating Qdrant collection '{collection}'...")
        qdrant_client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        
        points = []
        for doc in SAMPLE_DOCUMENTS:
            text = f"{doc['title']} {doc['content']}"
            vec = embed_text(text)
            points.append(PointStruct(
                id=doc["id"],
                vector=vec,
                payload={"title": doc["title"], "content": doc["content"]},
            ))
        
        qdrant_client.upsert(collection_name=collection, points=points)
        print(f"✓ Created with {len(points)} documents")
    except Exception as e:
        print(f"Warning: Could not initialize Qdrant: {e}")


_init_qdrant()
