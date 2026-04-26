"""
SOMA V2 Kernel
==============
Heterogeneous dispatch kernel. Classifies each incoming task by depth
then routes to the appropriate agent type:

  simple  → ReactiveAgent    (D1 rule-route, zero LLM calls)
  medium  → RoutingAgent     (D2 single LLM call)
  complex → DeliberativeAgent (D3 multi-step DAG planner)
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .depth_classifier import DepthClassifier, DEPTH_SIMPLE, DEPTH_MEDIUM, DEPTH_COMPLEX
from ..agents.reactive    import ReactiveAgent
from ..agents.routing     import RoutingAgent
from ..agents.deliberative import DeliberativeAgent
from .tools import ToolRegistry

logger = logging.getLogger("SOMA_V2.KERNEL")


# ── LLM call wrapper: concurrency + timeout + retry ──────────────────────────

def _make_resilient_llm(
    llm_callback,
    telemetry=None,
    timeout_s: float = 30.0,
    max_retries: int = 2,
    max_concurrent: int = 3,
):
    """
    Wraps any llm_callback with:
      - asyncio Semaphore (default 3)  — prevents flooding the LLM API
      - per-call timeout (default 30s) — prevents hung calls
      - exponential backoff retry      — 1s, 2s on timeout/error
    Returns None if no callback supplied.
    """
    if llm_callback is None:
        return None

    _sem = asyncio.Semaphore(max_concurrent)

    async def _call(task_type: str, prompt: str) -> str:
        delay    = 1.0
        last_exc: Exception = RuntimeError("no attempts made")

        for attempt in range(max_retries + 1):
            t_start = time.perf_counter()
            try:
                async with _sem:
                    res = await asyncio.wait_for(
                        llm_callback(task_type, prompt),
                        timeout=timeout_s,
                    )
                    call_ms = (time.perf_counter() - t_start) * 1000
                    if telemetry:
                        telemetry.log_event("llm_call", {
                            "task_type": task_type,
                            "attempt":   attempt + 1,
                            "status":    "success",
                            "call_ms":   round(call_ms, 2),
                        })
                    return res
            except asyncio.TimeoutError:
                last_exc   = asyncio.TimeoutError(f"LLM timeout after {timeout_s}s")
                latency_ms = (time.perf_counter() - t_start) * 1000
                logger.warning(
                    f"LLM '{task_type}' timed out "
                    f"(attempt {attempt+1}/{max_retries+1}, {latency_ms:.0f}ms)"
                )
                if telemetry:
                    telemetry.log_event("llm_call", {
                        "task_type": task_type, "attempt": attempt + 1,
                        "status": "timeout", "latency_ms": round(latency_ms, 2),
                    })
            except Exception as exc:
                last_exc   = exc
                latency_ms = (time.perf_counter() - t_start) * 1000
                logger.warning(
                    f"LLM '{task_type}' failed: {exc} "
                    f"(attempt {attempt+1}/{max_retries+1})"
                )
                if telemetry:
                    telemetry.log_event("llm_call", {
                        "task_type": task_type, "attempt": attempt + 1,
                        "status": "error", "error": str(exc),
                        "latency_ms": round(latency_ms, 2),
                    })

            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= 2

        logger.error(f"LLM '{task_type}' exhausted retries")
        raise last_exc

    return _call


# ── kernel ────────────────────────────────────────────────────────────────────

class V2Kernel:
    """
    Single agent execution unit.

    Parameters
    ----------
    llm_callback : async (task_type, prompt) -> str
        Injected LLM interface — provider-agnostic.
    memory : HierarchicalMemory (optional)
        Shared memory for plan caching + episodic recall.
    tool_registry : ToolRegistry (optional)
        Registered callable tools available to the deliberative agent.
    telemetry : TelemetryStore (optional)
        Structured JSONL event tracer.
    agent_id : str
        Unique ID for this kernel instance.
    llm_timeout_s : float
        Per-call LLM timeout in seconds.
    llm_max_retries : int
        Number of retries on timeout or error.
    llm_max_concurrent : int
        Max simultaneous LLM calls from this kernel.
    min_depth_confidence : float
        Minimum classifier confidence to trust depth prediction.
    """

    def __init__(
        self,
        llm_callback=None,
        memory=None,
        tool_registry: Optional[ToolRegistry] = None,
        telemetry: Optional[Any] = None,
        agent_id: Optional[str] = None,
        llm_timeout_s: float   = 30.0,
        llm_max_retries: int   = 2,
        llm_max_concurrent: int = 3,
        min_depth_confidence: float = 0.60,
        # Kept for legacy compat — silently ignored
        model_path: str = "",
        base_csv:   str = "",
        **kwargs,
    ):
        self.agent_id      = agent_id or f"node_{uuid.uuid4().hex[:8]}"
        self.memory        = memory
        self.tool_registry = tool_registry
        self.telemetry     = telemetry

        _llm = _make_resilient_llm(
            llm_callback,
            telemetry=telemetry,
            timeout_s=llm_timeout_s,
            max_retries=llm_max_retries,
            max_concurrent=llm_max_concurrent,
        )

        self.classifier = DepthClassifier(
            model_path=model_path or "src/soma_v2/models/depth_classifier.joblib",
            base_csv=base_csv or "",
            min_confidence=min_depth_confidence,
        )

        self.reactive     = ReactiveAgent()
        self.routing      = RoutingAgent(llm_callback=_llm)
        self.deliberative = DeliberativeAgent(
            llm_callback=_llm,
            memory=memory,
            tool_registry=tool_registry,
            telemetry=telemetry,
            agent_id=self.agent_id,
        )

        self._dispatch_log: List[Dict[str, Any]] = []
        logger.info(
            f"V2Kernel '{self.agent_id}' ready "
            f"(timeout={llm_timeout_s}s retries={llm_max_retries})"
        )

    def get_suspended_node(self) -> Optional[Any]:
        return self.deliberative.get_suspended_node()

    def approve(self):
        self.deliberative.approve()

    # ── public API ────────────────────────────────────────────────────────────

    async def handle(
        self,
        event: str,
        agent_role: str = "PEER",
        confidence: float = 0.75,
        urgency: str = "medium",
        contested: bool = False,
        reroute_attempts: int = 0,
        forced_depth: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Classify and dispatch a single task event.
        Returns a result dict including depth, agent_type, decision, latency_ms.
        """
        t0 = time.perf_counter()

        from .telemetry import TaskTracer
        tracer = None
        if self.telemetry:
            tracer = TaskTracer(
                self.telemetry,
                task_id or f"local_{uuid.uuid4().hex[:6]}"
            )
            tracer.record("task_start", event=event, agent_id=self.agent_id, urgency=urgency)

        # Fast-path: routine status/ping tasks skip ML inference entirely
        is_routine = any(
            kw in event.lower()
            for kw in ["status", "ping", "verify", "heartbeat", "health"]
        )
        if not forced_depth and not contested and urgency in ("low", "medium") and is_routine:
            depth, depth_prob = DEPTH_SIMPLE, 1.0
            agent_type = "hybrid_router"
            logger.info(f"V2Kernel: fast-path routine → {depth}")
        elif forced_depth:
            depth, depth_prob = forced_depth, 1.0
            agent_type = None
        else:
            depth, depth_prob = self.classifier.predict(
                agent_role, confidence, urgency, contested, reroute_attempts,
                event_text=event,
            )
            agent_type = None

        logger.info(f"V2Kernel: depth={depth} p={depth_prob:.3f} event='{event[:60]}'")
        if tracer:
            tracer.record("kernel_dispatch", depth=depth, depth_prob=depth_prob)

        try:
            if depth == DEPTH_SIMPLE:
                result = await self.reactive.handle(event, agent_role, confidence, urgency)
                if not agent_type:
                    agent_type = "reactive"
            elif depth == DEPTH_COMPLEX:
                result = await self.deliberative.handle(
                    event, agent_role, confidence, urgency, task_id=task_id
                )
                agent_type = "deliberative"
            else:
                result = await self.routing.handle(event, agent_role, confidence, urgency)
                agent_type = "routing"

            latency_ms = (time.perf_counter() - t0) * 1000

            if not forced_depth:
                self.classifier.record_outcome(
                    agent_role, confidence, urgency, contested, reroute_attempts, depth,
                    event_text=event,
                )

            record = {
                "event":      event[:80],
                "depth":      depth,
                "depth_prob": round(depth_prob, 4),
                "agent_type": agent_type,
                "decision":   result,
                "latency_ms": round(latency_ms, 2),
            }
            self._dispatch_log.append(record)
            if tracer:
                tracer.end("success", depth=depth, latency_ms=latency_ms)
            return record

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            if tracer:
                tracer.end("failed", error=str(exc), latency_ms=latency_ms)
            raise

    async def handle_batch(
        self,
        events: List[Tuple[str, Dict]],
    ) -> List[Dict[str, Any]]:
        """Process a batch of (event_text, context_kwargs) concurrently."""
        return await asyncio.gather(*[
            self.handle(event, **kwargs) for event, kwargs in events
        ])

    @property
    def dispatch_summary(self) -> Dict[str, int]:
        counts = {DEPTH_SIMPLE: 0, DEPTH_MEDIUM: 0, DEPTH_COMPLEX: 0}
        for r in self._dispatch_log:
            counts[r["depth"]] = counts.get(r["depth"], 0) + 1
        return counts

    @property
    def classifier_stats(self) -> dict:
        return self.classifier.stats
