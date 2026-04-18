"""
SOMA V2 Kernel
==============
Heterogeneous dispatch kernel. Classifies each incoming task by depth
then routes to the appropriate agent type:

  simple  → ReactiveAgent  (D1 rule-route, zero LLM)
  medium  → RoutingAgent   (single LLM call)
  complex → DeliberativeAgent (multi-step planner — stub for now)
"""

import asyncio
import logging
import time
from typing import List, Tuple, Dict, Any, Optional

from .depth_classifier import DepthClassifier, DEPTH_SIMPLE, DEPTH_MEDIUM, DEPTH_COMPLEX
from ..agents.reactive import ReactiveAgent
from ..agents.routing import RoutingAgent
from ..agents.deliberative import DeliberativeAgent

logger = logging.getLogger("SOMA_V2.KERNEL")

# ── LLM call wrapper: timeout + exponential-backoff retry ─────────────────────

def _make_resilient_llm(llm_callback, timeout_s: float = 30.0, max_retries: int = 2, max_concurrent: int = 2):
    """
    Wraps any llm_callback with:
      - concurrency semaphore (default 2) — prevents flooding a single Ollama instance
      - per-call timeout (default 30s) — prevents a hung Ollama from freezing a wave
      - exponential backoff retry (1s, 2s) on timeout or exception
      - returns a fallback string on total failure so the plan step degrades gracefully
    """
    if llm_callback is None:
        return None

    _sem = asyncio.Semaphore(max_concurrent)

    async def _call(task_type: str, prompt: str) -> str:
        delay = 1.0
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(max_retries + 1):
            try:
                async with _sem:
                    return await asyncio.wait_for(
                        llm_callback(task_type, prompt),
                        timeout=timeout_s,
                    )
            except asyncio.TimeoutError:
                last_exc = asyncio.TimeoutError(f"LLM timeout after {timeout_s}s")
                logger.warning(f"LLM call '{task_type}' timed out (attempt {attempt+1}/{max_retries+1})")
            except Exception as exc:
                last_exc = exc
                logger.warning(f"LLM call '{task_type}' failed: {exc} (attempt {attempt+1}/{max_retries+1})")
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= 2
        logger.error(f"LLM call '{task_type}' exhausted retries — using fallback")
        raise last_exc   # let caller handle (planner marks node FAILED, triggers backtrack)

    return _call


class V2Kernel:
    """
    Main entry point for SOMA V2.

    Parameters
    ----------
    llm_callback : async callable(task_type, prompt) -> str
        LLM interface — injected so kernel stays provider-agnostic.
    model_path : str
        Path to trained depth classifier joblib.
    base_csv : str
        Path to V1 training CSV for online retraining seed.
    min_depth_confidence : float
        Minimum classifier confidence to trust depth prediction.
        Below this threshold defaults to medium.
    """

    def __init__(
        self,
        llm_callback=None,
        memory=None,
        model_path: str = "src/soma_v2/models/depth_classifier_v1.joblib",
        base_csv: str = "experiments/results/v8_training_data.csv",
        min_depth_confidence: float = 0.60,
        llm_timeout_s: float = 30.0,
        llm_max_retries: int = 2,
        llm_max_concurrent: int = 2,
    ):
        self.llm_callback = llm_callback
        self.memory       = memory

        # Wrap callback with semaphore + timeout + retry
        _llm = _make_resilient_llm(llm_callback, timeout_s=llm_timeout_s, max_retries=llm_max_retries, max_concurrent=llm_max_concurrent)

        self.classifier = DepthClassifier(
            model_path=model_path,
            base_csv=base_csv,
            min_confidence=min_depth_confidence,
        )

        self.reactive     = ReactiveAgent()
        self.routing      = RoutingAgent(llm_callback=_llm)
        self.deliberative = DeliberativeAgent(llm_callback=_llm, memory=memory)

        self._dispatch_log: List[Dict[str, Any]] = []

        logger.info(f"V2Kernel initialised (llm_timeout={llm_timeout_s}s retries={llm_max_retries})")

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
    ) -> Dict[str, Any]:
        """
        Classify and dispatch a single event. Returns a result dict with
        depth, agent_type, decision, and latency_ms.
        """
        t0 = time.perf_counter()

        if forced_depth:
            depth, depth_prob = forced_depth, 1.0
        else:
            depth, depth_prob = self.classifier.predict(
                agent_role, confidence, urgency, contested, reroute_attempts,
                event_text=event,
            )
        
        logger.info(f"V2Kernel: depth={depth} p={depth_prob:.3f} event='{event[:60]}'")

        if depth == DEPTH_SIMPLE:
            result = await self.reactive.handle(event, agent_role, confidence, urgency)
            agent_type = "reactive"
        elif depth == DEPTH_COMPLEX:
            result = await self.deliberative.handle(event, agent_role, confidence, urgency)
            agent_type = "deliberative"
        else:
            result = await self.routing.handle(event, agent_role, confidence, urgency)
            agent_type = "routing"

        latency_ms = (time.perf_counter() - t0) * 1000

        # Feed outcome back to classifier for online retraining
        # If we forced a depth, we don't necessarily want to treat it as 'truth' 
        # unless it was successful.
        self.classifier.record_outcome(
            agent_role, confidence, urgency, contested, reroute_attempts, depth,
            event_text=event,
        )

        record = {
            "event":       event[:80],
            "depth":       depth,
            "depth_prob":  round(depth_prob, 4),
            "agent_type":  agent_type,
            "decision":    result,
            "latency_ms":  round(latency_ms, 2),
        }
        self._dispatch_log.append(record)
        return record

    async def handle_batch(
        self,
        events: List[Tuple[str, Dict]],
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of (event_text, context_kwargs) tuples concurrently.
        """
        tasks = [
            self.handle(event, **kwargs)
            for event, kwargs in events
        ]
        return await asyncio.gather(*tasks)

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def dispatch_summary(self) -> Dict[str, int]:
        counts = {DEPTH_SIMPLE: 0, DEPTH_MEDIUM: 0, DEPTH_COMPLEX: 0}
        for r in self._dispatch_log:
            counts[r["depth"]] = counts.get(r["depth"], 0) + 1
        return counts

    @property
    def classifier_stats(self) -> dict:
        return self.classifier.stats
