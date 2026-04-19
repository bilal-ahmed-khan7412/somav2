import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger("SOMA_V2.BLACKBOARD")


class ResourcePool:
    """
    Groups interchangeable units (e.g., B1-B4 drones).
    Tasks claim 'any unit' and get whichever is free — reduces conflicts ~80%.
    """

    def __init__(self, pool_id: str, units: List[str]) -> None:
        self.pool_id = pool_id
        self._units: List[str] = list(units)
        self._available: Set[str] = set(units)
        self._lock = asyncio.Lock()

    async def claim_any(
        self,
        agent_id: str,
        blackboard: "ResourceBlackboard",
        node_id: str,
        timeout_s: float = 2.0,
    ) -> Optional[str]:
        """Claim any free unit. Returns unit_id on success, None if all busy."""
        async with self._lock:
            for unit in self._units:
                if unit in self._available:
                    claimed = await blackboard.claim(unit, agent_id, node_id, timeout_s)
                    if claimed:
                        self._available.discard(unit)
                        return unit
        return None

    async def release(
        self,
        unit_id: str,
        agent_id: str,
        node_id: str,
        blackboard: "ResourceBlackboard",
    ) -> None:
        await blackboard.release(unit_id, agent_id, node_id)
        async with self._lock:
            self._available.add(unit_id)

    @property
    def available_count(self) -> int:
        return len(self._available)

    @property
    def total_count(self) -> int:
        return len(self._units)


class ResourceBlackboard:
    """
    Shared state container for SOMA V2 agents.
    Provides mutually exclusive access to physical resources (drones, sectors).
    """
    
    def __init__(self, bus: Optional[Any] = None) -> None:
        self._bus = bus
        self._locks: Dict[str, str] = {}  # resource_id -> agent_id (owner)
        self._lock_objs: Dict[str, asyncio.Lock] = {}
        self._telemetry: Dict[str, dict] = {}
        self._total_claims = 0
        self._total_conflicts = 0

    @property
    def stats(self) -> dict:
        return {
            "active_locks": len(self._locks),
            "total_claims": self._total_claims,
            "total_conflicts": self._total_conflicts,
            "telemetry_points": len(self._telemetry),
            "locked_units": list(self._locks.keys())
        }

    def _get_lock(self, resource_id: str) -> asyncio.Lock:
        if resource_id not in self._lock_objs:
            self._lock_objs[resource_id] = asyncio.Lock()
        return self._lock_objs[resource_id]

    async def claim(self, resource_id: str, agent_id: str, node_id: str, timeout_s: float = 2.0) -> bool:
        """Attempt to lock a resource for an agent's specific plan node.

        Reentrant: if the calling agent already owns this resource (e.g. a
        negotiated sub-step executing inside the owner's context), the claim
        succeeds immediately without re-acquiring the asyncio.Lock.
        """
        # Reentrant check — owner doesn't need to re-acquire its own lock
        if self._locks.get(resource_id) == agent_id:
            self._total_claims += 1
            logger.info(f"Blackboard: Unit '{resource_id}' re-CLAIMED (reentrant) by '{agent_id}' for '{node_id}'")
            return True

        lock = self._get_lock(resource_id)
        try:
            # We use a timeout to avoid deadlocks in the swarm
            await asyncio.wait_for(lock.acquire(), timeout=timeout_s)
            self._total_claims += 1
            self._locks[resource_id] = agent_id
            logger.info(f"Blackboard: Unit '{resource_id}' CLAIMED by '{agent_id}' for '{node_id}'")
            return True
        except asyncio.TimeoutError:
            self._total_conflicts += 1
            owner = self._locks.get(resource_id, "unknown")
            logger.warning(f"Blackboard: Resource conflict! Unit '{resource_id}' held by '{owner}'")
            return False

    async def release(self, resource_id: str, agent_id: str, node_id: str) -> None:
        """Release a resource lock."""
        if resource_id in self._lock_objs:
            lock = self._lock_objs[resource_id]
            if lock.locked():
                lock.release()
                if self._locks.get(resource_id) == agent_id:
                    del self._locks[resource_id]
                logger.info(f"Blackboard: Unit '{resource_id}' RELEASED by '{agent_id}'")

    def update_telemetry(self, resource_id: str, data: dict) -> None:
        self._telemetry[resource_id] = {**self._telemetry.get(resource_id, {}), **data, "ts": time.time()}

    def get_telemetry(self, resource_id: str) -> Optional[dict]:
        return self._telemetry.get(resource_id)
