"""
SOMA V2 — HierarchicalMemory
==============================
Unified facade over the two-tier memory system.

  Hot layer  (HotMemory)  — in-process LRU, <0.1ms, ephemeral working state
  Cold layer (ColdMemory) — ChromaDB / flat fallback, ~10-50ms, episodic recall

Cold writes are fully async: a single background worker thread drains a
queue so episode commits never block task execution (no per-task threads).
"""

from __future__ import annotations

import logging
import queue as _queue
import threading
import uuid
from typing import Any, Dict, List, Optional

from .hot  import HotMemory
from .cold import ColdMemory

logger = logging.getLogger("SOMA_V2.MEMORY")


class HierarchicalMemory:
    """
    Two-tier memory manager.

    Parameters
    ----------
    hot_capacity  : max LRU entries per agent namespace
    hot_ttl       : seconds before hot entries expire (0 = no expiry)
    cold_persist  : directory for ChromaDB persistence (None = in-memory)
    cold_collection : ChromaDB collection name
    """

    def __init__(
        self,
        hot_capacity:    int   = 256,
        hot_ttl:         float = 300.0,
        cold_persist:    Optional[str] = "./soma_cache",
        cold_collection: str  = "soma_episodes",
        cold_enabled:    bool = True,
    ) -> None:
        self.hot  = HotMemory(capacity=hot_capacity, default_ttl=hot_ttl)
        self.cold = ColdMemory(persist_dir=cold_persist, collection=cold_collection) if cold_enabled else None
        # Single background worker drains write queue — no per-task thread spawn
        self._write_q: _queue.Queue = _queue.Queue()
        self._worker = threading.Thread(target=self._cold_writer, daemon=True)
        self._worker.start()
        cold_info = self.cold.backend if self.cold else "disabled"
        logger.info(f"HierarchicalMemory ready — hot(cap={hot_capacity} ttl={hot_ttl}s) cold({cold_info})")

    def sync(self) -> None:
        """Block until all background cold-memory writes are complete."""
        self._write_q.join()

    def _cold_writer(self) -> None:
        while True:
            kwargs = self._write_q.get()
            if kwargs is None:
                break
            try:
                self.cold.record(**kwargs)
            except Exception as exc:
                logger.warning(f"HierarchicalMemory cold write failed: {exc}")
            self._write_q.task_done()

    # ── hot layer API ─────────────────────────────────────────────────────────

    def remember(
        self,
        agent_id: str,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Store a working-memory value for an agent (fast path)."""
        self.hot.set(agent_id, key, value, ttl=ttl)

    def recall_working(self, agent_id: str, key: str) -> Optional[Any]:
        """Retrieve a working-memory value (<0.1ms)."""
        return self.hot.get(agent_id, key)

    def working_context(self, agent_id: str) -> Dict[str, Any]:
        """All live working-memory entries for an agent."""
        return self.hot.get_all(agent_id)

    def forget(self, agent_id: str, key: str) -> bool:
        """Delete a specific working-memory key."""
        return self.hot.delete(agent_id, key)

    def clear_working(self, agent_id: str) -> int:
        """Flush all working memory for an agent (e.g. after task completes)."""
        return self.hot.flush(agent_id)

    # ── cold layer API ────────────────────────────────────────────────────────

    def commit_episode(
        self,
        event:      str,
        agent_id:   str,
        agent_type: str,
        action:     str,
        urgency:    str,
        success:    bool,
        extra:      Optional[Dict] = None,
    ) -> str:
        """Persist a completed task episode to cold memory."""
        episode_id = self.cold.record(
            event=event, agent_id=agent_id, agent_type=agent_type,
            action=action, urgency=urgency, success=success, extra=extra,
        )
        logger.debug(f"HierarchicalMemory: episode committed id={episode_id}")
        return episode_id

    def recall_similar(
        self,
        query:          str,
        n:              int = 5,
        only_successes: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve semantically similar past episodes from cold memory.
        Returns list of {event, metadata, distance} dicts, closest first.
        """
        if self.cold is None:
            return []
        return self.cold.recall(query, n=n, filter_success=only_successes)

    # ── combined helpers ──────────────────────────────────────────────────────

    def task_start(self, agent_id: str, task_id: str, event: str, context: Dict) -> None:
        """Called when an agent starts a task — seeds working memory."""
        self.remember(agent_id, "task_id",  task_id)
        self.remember(agent_id, "event",    event)
        self.remember(agent_id, "context",  context)
        self.remember(agent_id, "step",     0)

    def task_done(
        self,
        agent_id:   str,
        agent_type: str,
        action:     str,
        urgency:    str,
        success:    bool,
        extra:      Optional[Dict] = None,
    ) -> str:
        """Called when a task completes — enqueues cold write, clears hot."""
        event      = self.recall_working(agent_id, "event") or ""
        episode_id = uuid.uuid4().hex
        self.clear_working(agent_id)
        if self.cold is not None:
            self._write_q.put(dict(event=event, agent_id=agent_id, agent_type=agent_type,
                                   action=action, urgency=urgency, success=success, extra=extra))
        return episode_id

    # ── stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "hot":  self.hot.stats,
            "cold": self.cold.stats if self.cold else {"backend": "disabled"},
        }
