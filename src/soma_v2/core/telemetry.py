import json
import time
import os
import logging
import threading
from typing import Any, Dict, Optional
from contextvars import ContextVar

logger = logging.getLogger("SOMA_V2.TELEMETRY")

# Global context for task_id propagation
current_task_id: ContextVar[Optional[str]] = ContextVar("current_task_id", default=None)

class TelemetryStore:
    """
    Structured JSONL tracer for SOMA V2.
    Captures the decision lifecycle of every task for audit and debugging.
    """
    
    def __init__(self, trace_dir: Optional[str] = None):
        self.trace_dir = trace_dir
        self._lock = threading.Lock()
        if self.trace_dir:
            os.makedirs(self.trace_dir, exist_ok=True)
            self.trace_file = os.path.join(self.trace_dir, f"trace_{int(time.time())}.jsonl")
            logger.info(f"Telemetry enabled: writing to {self.trace_file}")
        else:
            self.trace_file = None

    def log_event(self, event_type: str, data: Dict[str, Any], task_id: Optional[str] = None):
        """Logs a structured event to the trace file."""
        if not self.trace_file:
            return

        # Use contextvar as fallback for task_id
        tid = task_id or current_task_id.get()

        # Guard against key collisions with system fields
        clean_data = data.copy()
        if "event" in clean_data:
            clean_data["event_text"] = clean_data.pop("event")
        if "timestamp" in clean_data:
            clean_data["original_timestamp"] = clean_data.pop("timestamp")

        payload = {
            "timestamp": time.time(),
            "event": event_type,
            "task_id": tid,
            **clean_data
        }
        
        try:
            with self._lock:
                with open(self.trace_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to write telemetry event: {exc}")

class TaskTracer:
    """Context-bound tracer for a single task lifecycle."""
    def __init__(self, store: TelemetryStore, task_id: str):
        self.store = store
        self.task_id = task_id
        self.t0 = time.perf_counter()
        # Set context for deep-nested logging (e.g. LLM retries)
        self._ctx_token = current_task_id.set(task_id)

    def record(self, event_name: str, **kwargs):
        self.store.log_event(event_name, kwargs, task_id=self.task_id)

    def end(self, status: str, **kwargs):
        # Prefer provided latency_ms if any, else calculate
        latency_ms = kwargs.pop("latency_ms", (time.perf_counter() - self.t0) * 1000)
        self.store.log_event("task_end", {
            "status": status,
            "latency_ms": round(latency_ms, 2),
            **kwargs
        }, task_id=self.task_id)
        # Reset context
        current_task_id.reset(self._ctx_token)
