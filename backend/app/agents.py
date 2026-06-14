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

from sentence_transformers import SentenceTransformer, CrossEncoder

# Initialize local HuggingFace embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 384

# Initialize CrossEncoder for Re-ranking
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except:
    reranker = None

def embed_text(text: str) -> List[float]:
    """Create semantic embedding using HuggingFace."""
    return embedder.encode(text).tolist()


def qdrant_search(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Search Qdrant collection."""
    vector = embed_text(query)
    results = qdrant_client.query_points(
        collection_name=settings.clinical_collection,
        query=vector,
        limit=limit,
    )
    docs = []
    for point in results.points:
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
        # Fetch top 20 candidates for re-ranking
        docs = qdrant_search(query, limit=20) if query else []
        if docs and reranker:
            trace.append(f"Retrieval Agent: Retrieved {len(docs)} candidates. Re-ranking using CrossEncoder...")
            pairs = [[query, doc["payload"].get("content", "")] for doc in docs]
            scores = reranker.predict(pairs)
            for idx, doc in enumerate(docs):
                doc["score"] = float(scores[idx])  # Overwrite vector score with rerank score
            # Sort descending and keep top 5
            docs = sorted(docs, key=lambda x: x["score"], reverse=True)[:5]
            
        if docs:
            top = docs[0]
            title = (top.get("payload") or {}).get("title", "Untitled protocol")
            trace.append(f"Retrieval Agent: Final top match after re-ranking is '{title}'. Passing {len(docs)} documents to Analysis.")
        else:
            trace.append("Retrieval Agent: No relevant protocol snippets found for this question.")
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        trace.append(f"Retrieval Agent: Encountered an error: {e}.")
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
                "next_node": "GraderAgent",
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

def grader_node(state: AgentState) -> dict:
    """Evaluate if the answer addresses the question based on the retrieved context."""
    messages = state.get("messages") or []
    answer = state.get("final_answer") or ""
    trace = state.get("trace") or []
    
    query = getattr(messages[-1], 'content', str(messages[-1])) if messages else ""
    
    grade_prompt = f"Does the following answer directly address the user's question? Respond strictly with 'YES' or 'NO'.\n\nQuestion: {query}\nAnswer: {answer}"
    
    try:
        response = llm.invoke([SystemMessage(content="You are a strict, objective grader."), HumanMessage(content=grade_prompt)])
        if "YES" in str(response.content).upper():
            trace.append("Grader Agent: Answer passed self-reflection. Approved for user.")
        else:
            trace.append("Grader Agent: Self-reflection failed! Answer does not fully address the question. Flagging answer.")
            answer += "\n\n*(Self-Reflection Grader Warning: This answer may not fully resolve your query based on the available protocols.)*"
    except Exception as e:
        logger.error(f"Grader error: {e}")
        
    return {"next_node": "FINISH", "final_answer": answer, "trace": trace}


# --- Workflow Builder -------------------------------------------------------


def build_workflow() -> StateGraph:
    """Build the agent workflow."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("RetrievalAgent", retrieval_node)
    workflow.add_node("AnalysisAgent", analysis_node)
    workflow.add_node("GraderAgent", grader_node)
    
    workflow.set_entry_point("Supervisor")
    
    def route(state: AgentState) -> str:
        next_node = state.get("next_node", "FINISH")
        return END if next_node == "FINISH" else next_node
    
    workflow.add_conditional_edges("Supervisor", route)
    workflow.add_edge("RetrievalAgent", "AnalysisAgent")
    workflow.add_edge("AnalysisAgent", "GraderAgent")
    workflow.add_edge("GraderAgent", END)
    
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
        print(f"✓ Created collection '{collection}'")
    except Exception as e:
        print(f"Warning: Could not initialize Qdrant: {e}")


_init_qdrant()
