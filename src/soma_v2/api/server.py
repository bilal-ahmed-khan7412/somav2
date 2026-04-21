from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import asyncio
import os

from ..main import SOMASwarm

app = FastAPI(title="SOMA V2 Swarm Kernel API")

# Global swarm instance
# Note: In production, you'd load the model from an environment variable
MODEL = os.getenv("SOMA_MODEL", "ollama/qwen2.5:3b")
swarm = SOMASwarm(model=MODEL)

# Storage for async missions
missions: Dict[str, Dict[str, Any]] = {}

class MissionRequest(BaseModel):
    task: str
    urgency: Optional[str] = "medium"

class ApprovalRequest(BaseModel):
    task_id: str

@app.on_event("startup")
async def startup_event():
    print(f"SOMA V2 Kernel booting with model: {MODEL}")

@app.post("/mission")
async def dispatch_mission(req: MissionRequest, background_tasks: BackgroundTasks):
    """
    Dispatch a natural language mission to the swarm.
    Returns a mission_id for tracking.
    """
    mission_id = str(uuid.uuid4())
    missions[mission_id] = {
        "task": req.task,
        "status": "RUNNING",
        "result": None,
        "error": None
    }
    
    # Run in background to avoid blocking the API
    background_tasks.add_task(_run_mission, mission_id, req.task, req.urgency)
    
    return {"mission_id": mission_id, "status": "RUNNING"}

async def _run_mission(mission_id: str, task: str, urgency: str):
    try:
        result = await swarm.dispatch(task, urgency=urgency)
        missions[mission_id]["status"] = "COMPLETED"
        missions[mission_id]["result"] = result
    except Exception as e:
        missions[mission_id]["status"] = "FAILED"
        missions[mission_id]["error"] = str(e)

@app.get("/mission/{mission_id}")
async def get_mission_status(mission_id: str):
    if mission_id not in missions:
        raise HTTPException(status_code=404, detail="Mission not found")
    return missions[mission_id]

@app.get("/suspended")
async def list_suspended_tasks():
    """Returns all tasks currently waiting for HITL approval."""
    return await swarm.get_suspended_tasks()

@app.post("/approve/{task_id}")
async def approve_task(task_id: str):
    """Signals a suspended task to resume execution."""
    success = await swarm.approve(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or not suspended")
    return {"status": "RESUMED", "task_id": task_id}

@app.get("/metrics")
async def get_metrics():
    """Returns real-time kernel performance statistics."""
    return swarm.stats

@app.get("/health")
async def health_check():
    return {"status": "online", "model": MODEL}
