import abc
import asyncio
import logging

logger = logging.getLogger("SOMA_V2.ACTUATOR")


class Actuator(abc.ABC):
    """Base interface for physical world interaction."""

    @abc.abstractmethod
    async def execute_command(self, cmd_string: str) -> bool:
        """Parse and execute a raw command string. Returns True on success."""
        pass


class MockDroneActuator(Actuator):
    """
    Simulates drone/unit commands with realistic log output.

    Supported commands (case-insensitive):
      TAKEOFF <unit>
      GOTO <unit> <target>
      LAND <unit>
      SCAN <unit> <area>
      DEPLOY <unit> <payload>
      STATUS <unit>
    """

    async def execute_command(self, cmd_string: str) -> bool:
        raw   = cmd_string.strip()
        parts = raw.split()

        if not parts:
            logger.warning("MockDroneActuator: empty command")
            return False

        verb = parts[0].upper()

        if verb == "TAKEOFF":
            unit = parts[1] if len(parts) > 1 else "UNIT"
            print(f"  [ACTUATOR] {unit}: Powering motors... ASCENDING to 10m AGL.")
            await asyncio.sleep(0.5)
            return True

        if verb == "GOTO":
            unit   = parts[1] if len(parts) > 1 else "UNIT"
            target = " ".join(parts[2:]) if len(parts) > 2 else "TARGET"
            print(f"  [ACTUATOR] {unit}: Navigating to {target} via autonomous GPS path.")
            await asyncio.sleep(0.8)
            return True

        if verb == "LAND":
            unit = parts[1] if len(parts) > 1 else "UNIT"
            print(f"  [ACTUATOR] {unit}: Descent initiated — MOTORS DISARMED on touchdown.")
            await asyncio.sleep(0.5)
            return True

        if verb == "SCAN":
            unit = parts[1] if len(parts) > 1 else "UNIT"
            area = " ".join(parts[2:]) if len(parts) > 2 else "AREA"
            print(f"  [ACTUATOR] {unit}: Initiating sensor sweep of {area}.")
            await asyncio.sleep(0.6)
            return True

        if verb == "DEPLOY":
            unit    = parts[1] if len(parts) > 1 else "UNIT"
            payload = " ".join(parts[2:]) if len(parts) > 2 else "PAYLOAD"
            print(f"  [ACTUATOR] {unit}: Deploying {payload}.")
            await asyncio.sleep(0.4)
            return True

        if verb == "STATUS":
            unit = parts[1] if len(parts) > 1 else "UNIT"
            print(f"  [ACTUATOR] {unit}: ONLINE — battery 87%, comms nominal.")
            await asyncio.sleep(0.1)
            return True

        logger.warning(f"MockDroneActuator: unknown command verb '{verb}' in '{raw}'")
        return False
