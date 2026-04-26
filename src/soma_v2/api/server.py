"""
SOMA V2 — FastAPI HTTP Server
==============================
Exposes the SOMASwarm via a REST API.

Endpoints
---------
POST /mission             — dispatch a task; returns mission_id
GET  /mission/{id}        — poll task status + result
GET  /suspended           — list tasks waiting for HITL approval
POST /approve/{agent_id}  — resume a suspended task
GET  /metrics             — swarm performance statistics
GET  /health              — liveness probe
"""

import asyncio
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.background import BackgroundTasks
from pydantic import BaseModel

from ..main import SOMASwarm

app = FastAPI(
    title="SOMA V2 — Multi-Agent Kernel API",
    version="2.0.0",
    description="Heterogeneous multi-agent swarm backed by the OpenAI API.",
)

# ── swarm singleton ───────────────────────────────────────────────────────────
MODEL = os.getenv("SOMA_MODEL", "openai/gpt-4o-mini")
SLOTS = int(os.getenv("SOMA_SLOTS", "3"))
swarm: Optional[SOMASwarm] = None

# In-memory mission store
missions: Dict[str, Dict[str, Any]] = {}


# ── lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    global swarm
    swarm = SOMASwarm(model=MODEL, slots=SLOTS)
    print(f"SOMA V2 kernel online — model={MODEL} slots={SLOTS}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if swarm:
        await swarm.close()


# ── request / response models ─────────────────────────────────────────────────

class MissionRequest(BaseModel):
    task:         str
    urgency:      Optional[str] = "medium"
    forced_depth: Optional[str] = None


class MissionResponse(BaseModel):
    mission_id: str
    status:     str


# ── routes ────────────────────────────────────────────────────────────────────

@app.post("/mission", response_model=MissionResponse, summary="Dispatch a task")
async def dispatch_mission(
    req: MissionRequest,
    background_tasks: BackgroundTasks,
) -> MissionResponse:
    """
    Dispatch a natural-language task to the swarm.
    Returns a mission_id; poll GET /mission/{id} for the result.
    """
    mission_id = str(uuid.uuid4())
    missions[mission_id] = {"task": req.task, "status": "RUNNING", "result": None, "error": None}
    background_tasks.add_task(_run_mission, mission_id, req.task, req.urgency, req.forced_depth)
    return MissionResponse(mission_id=mission_id, status="RUNNING")


async def _run_mission(
    mission_id: str,
    task: str,
    urgency: str,
    forced_depth: Optional[str],
) -> None:
    try:
        result = await swarm.dispatch(task, urgency=urgency, forced_depth=forced_depth)
        missions[mission_id]["status"] = "COMPLETED"
        missions[mission_id]["result"] = result
    except Exception as exc:
        missions[mission_id]["status"] = "FAILED"
        missions[mission_id]["error"]  = str(exc)


@app.get("/mission/{mission_id}", summary="Poll mission status")
async def get_mission(mission_id: str) -> Dict[str, Any]:
    """Returns the current status and result of a dispatched mission."""
    if mission_id not in missions:
        raise HTTPException(status_code=404, detail="Mission not found")
    return missions[mission_id]


@app.get("/suspended", summary="List suspended tasks")
async def list_suspended() -> Dict[str, str]:
    """Returns all tasks currently paused for human-in-the-loop approval."""
    return swarm.get_suspended_tasks()


@app.post("/approve/{agent_id}", summary="Resume a suspended task")
async def approve_task(agent_id: str) -> Dict[str, str]:
    """Signals a suspended agent to resume execution."""
    success = swarm.approve(agent_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"No suspended task found for agent '{agent_id}'"
        )
    return {"status": "RESUMED", "agent_id": agent_id}


@app.get("/metrics", summary="Swarm performance metrics")
async def get_metrics() -> Dict[str, Any]:
    """Returns real-time kernel statistics: memory, director, bus."""
    return swarm.stats


@app.get("/health", summary="Liveness probe")
async def health_check() -> Dict[str, str]:
    return {"status": "online", "model": MODEL}
