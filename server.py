import asyncio
import aiohttp
import json
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import SOMA V2 Core
from src.soma_v2.core.director import AgentDirector
from src.soma_v2.memory.hierarchical import HierarchicalMemory

app = FastAPI(title="SOMA V2 Architecture API")

# Allow CORS for local HTML testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
memory = HierarchicalMemory()

async def real_llm_callback(task_type: str, prompt: str) -> str:
    """Real LLM Call to Ollama"""
    system_prompt = (
        "You are the SOMA V2 Deliberative Planner. "
        "You must output ONLY valid JSON. "
        "Do not include markdown blocks or any other text. "
        "The JSON should contain a 'plan' array with objects having 'step' and 'action' keys, "
        "and a 'reasoning' key explaining your strategy."
    )
    
    payload = {
        "model": "qwen2.5:3b",
        "prompt": f"{system_prompt}\n\nTask Type: {task_type}\nPrompt: {prompt}",
        "stream": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:11434/api/generate", json=payload, timeout=60) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data["response"]
                    
                    # Clean up if the LLM output markdown json blocks
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                        
                    return content
                else:
                    return json.dumps({"action": "error", "reason": f"Ollama HTTP {resp.status}"})
    except Exception as e:
        return json.dumps({"action": "error", "reason": str(e)})

# Initialize Director with REAL LLM
director = AgentDirector(llm_callback=real_llm_callback, memory=memory)
director.add_slot("agent_1", role="PEER")

@app.on_event("startup")
async def startup_event():
    await director.start()

@app.on_event("shutdown")
async def shutdown_event():
    await director.stop()

class TaskRequest(BaseModel):
    task: str

@app.post("/api/dispatch")
async def dispatch_task(req: TaskRequest):
    """
    Feed the text to the real SOMA V2 AgentDirector.
    It will run through the real Depth Classifier, allocate via A2A, and execute.
    """
    # Assign the task to the director
    outcome = await director.assign(event=req.task, urgency="critical")
    
    if outcome["status"] == "success":
        res = outcome["result"]
        return {
            "success": True,
            "task_id": outcome["task_id"],
            "assigned_to": outcome["assigned_to"],
            "depth": res["depth"],
            "depth_prob": res["depth_prob"],
            "agent_type": res["agent_type"],
            "latency_ms": res["latency_ms"],
            "raw_decision": str(res["decision"])
        }
    else:
        return {"success": False, "error": outcome.get("reason", "Unknown error")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
