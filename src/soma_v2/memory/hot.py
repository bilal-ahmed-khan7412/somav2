"""
SOMA V2 — Hot Memory Layer
===========================
In-process LRU cache for agent working memory.
Zero external deps, sub-millisecond reads/writes.

Stores arbitrary key→value pairs per agent_id namespace.
Evicts least-recently-used entries when capacity is exceeded.
TTL is enforced lazily on read (no background sweeper needed at this scale).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class _Entry:
    value:      Any
    expires_at: float   # monotonic time; 0 = never


class HotMemory:
    """
    Per-agent LRU cache with optional TTL.

    Parameters
    ----------
    capacity : max entries per agent namespace
    default_ttl : seconds before an entry expires (0 = no expiry)
    """

    def __init__(self, capacity: int = 256, default_ttl: float = 300.0) -> None:
        self.capacity    = capacity
        self.default_ttl = default_ttl
        # agent_id -> OrderedDict[key -> _Entry]  (insertion-order = LRU order)
        self._store: Dict[str, OrderedDict] = {}

    # ── write ─────────────────────────────────────────────────────────────────

    def set(
        self,
        agent_id: str,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        ns = self._ns(agent_id)
        expires_at = (time.monotonic() + (ttl if ttl is not None else self.default_ttl)
                      if (ttl or self.default_ttl) else 0.0)
        if key in ns:
            ns.move_to_end(key)
        ns[key] = _Entry(value=value, expires_at=expires_at)
        if len(ns) > self.capacity:
            ns.popitem(last=False)  # evict LRU

    # ── read ──────────────────────────────────────────────────────────────────

    def get(self, agent_id: str, key: str) -> Optional[Any]:
        ns    = self._store.get(agent_id)
        if ns is None or key not in ns:
            return None
        entry = ns[key]
        if entry.expires_at and time.monotonic() > entry.expires_at:
            del ns[key]
            return None
        ns.move_to_end(key)
        return entry.value

    def get_all(self, agent_id: str) -> Dict[str, Any]:
        ns  = self._store.get(agent_id, {})
        now = time.monotonic()
        return {
            k: e.value for k, e in ns.items()
            if not e.expires_at or now <= e.expires_at
        }

    # ── delete ────────────────────────────────────────────────────────────────

    def delete(self, agent_id: str, key: str) -> bool:
        ns = self._store.get(agent_id)
        if ns and key in ns:
            del ns[key]
            return True
        return False

    def flush(self, agent_id: str) -> int:
        ns = self._store.pop(agent_id, {})
        return len(ns)

    # ── stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "namespaces": len(self._store),
            "total_keys": sum(len(ns) for ns in self._store.values()),
            "capacity":   self.capacity,
        }

    def _ns(self, agent_id: str) -> OrderedDict:
        if agent_id not in self._store:
            self._store[agent_id] = OrderedDict()
        return self._store[agent_id]
