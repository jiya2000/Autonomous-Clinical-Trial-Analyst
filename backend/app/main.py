from __future__ import annotations

import logging.config
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from .agents import build_workflow, AgentState

logger = logging.getLogger(__name__)


# Configure logging
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console'],
    },
})

app = FastAPI(title="Autonomous Clinical Trial Analyst")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.websocket("/ws/chat/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
    await websocket.accept()
    workflow = build_workflow().compile()

    try:
        while True:
            logger.info("=== WEBSOCKET: Waiting for message... ===")
            data = await websocket.receive_json()
            logger.info(f"=== WEBSOCKET: Received data: {data} ===")
            query = data.get("query", "")

            initial_state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "next_node": "Supervisor",
                "current_plan": [],
                "retrieved_docs": [],
                "final_answer": "",
                "trace": [],
            }

            try:
                logger.info(f"=== WEBSOCKET: Processing query: {query[:100]}... ===")
                # Run workflow and get final state
                final_state = await workflow.ainvoke(initial_state)
                logger.info(f"=== WORKFLOW COMPLETED: final_state keys = {list(final_state.keys())} ===")

                # Stream agent thoughts (reasoning trace) in order
                for step in final_state.get("trace", []):
                    await websocket.send_json({
                        "type": "thought",
                        "agent": "Agent",
                        "content": step,
                    })

                # Send the final answer in a single message (no token streaming)
                answer = final_state.get("final_answer", "No answer generated")
                logger.info(f"=== FINAL ANSWER from state: {answer[:300] if answer else 'EMPTY'}... ===")
                await websocket.send_json({
                    "type": "answer",
                    "content": answer,
                })

                await websocket.send_json({"type": "done"})
            except Exception as e:
                error_msg = str(e)
                logger.error(f"=== WEBSOCKET ERROR: {error_msg} ===", exc_info=True)
                await websocket.send_json({"type": "error", "content": f"Error: {error_msg}"})

    except WebSocketDisconnect:
        return
