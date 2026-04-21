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


class AirSimActuator(Actuator):
    """
    Real drone actuator via Microsoft AirSim.

    Maps SOMA [CMD] tokens to AirSim Python API calls.
    Requires AirSim Blocks (or any environment) running locally
    and `pip install airsim`.

    Supported commands:
      TAKEOFF <unit>
      GOTO    <unit> <x> <y> <z>   (z is negative = up in NED coords)
      LAND    <unit>
      SCAN    <unit> <area>         (hovers and captures image)
      STATUS  <unit>

    Multi-drone: AirSim identifies drones by vehicle_name.
    Unit names like B1, B2 map to vehicle names "Drone1", "Drone2" etc.
    Configure multiple drones in AirSim settings.json before use.
    """

    # Map SOMA unit IDs → AirSim vehicle names
    UNIT_MAP = {
        "B1": "Drone1", "B2": "Drone2",
        "B3": "Drone3", "B4": "Drone4",
        "A1": "Drone1", "A2": "Drone2",
    }
    DEFAULT_VEHICLE = "Drone1"
    TAKEOFF_ALT     = -3.0   # metres, NED (negative = up)
    CRUISE_SPEED    = 5.0    # m/s

    def __init__(self) -> None:
        try:
            import airsim
            self._airsim = airsim
            self._clients: dict[str, airsim.MultirotorClient] = {}
            # Verify primary connection
            main_client = self._get_client(self.DEFAULT_VEHICLE)
            main_client.confirmConnection()
            logger.info("AirSimActuator: initialized client pool")
        except ImportError:
            raise RuntimeError("airsim package not installed — run: pip install airsim")
        except Exception as exc:
            raise RuntimeError(f"AirSim not reachable — is Blocks running? ({exc})")

    def _get_client(self, vname: str):
        """Get or create a dedicated RPC client for a specific vehicle."""
        if vname not in self._clients:
            client = self._airsim.MultirotorClient()
            client.confirmConnection()
            # We don't enable API control here; we do it per-command
            self._clients[vname] = client
        return self._clients[vname]

    def _vehicle(self, unit: str) -> str:
        return self.UNIT_MAP.get(unit.upper(), self.DEFAULT_VEHICLE)

    async def execute_command(self, cmd_string: str) -> bool:
        loop  = asyncio.get_event_loop()
        raw   = cmd_string.strip()
        parts = raw.split()
        if not parts:
            return False

        verb = parts[0].upper()
        unit = parts[1] if len(parts) > 1 else "B1"
        vname = self._vehicle(unit)
        client = self._get_client(vname)

        try:
            if verb == "TAKEOFF":
                print(f"  >> TAKEOFF  {unit} | ascending to {abs(self.TAKEOFF_ALT):.0f}m ...")
                await loop.run_in_executor(None, self._do_takeoff, client, vname)
                print(f"  << TAKEOFF  {unit} | airborne")
                return True

            if verb == "GOTO":
                x = float(parts[2]) if len(parts) > 2 else 10.0
                y = float(parts[3]) if len(parts) > 3 else 0.0
                z = float(parts[4]) if len(parts) > 4 else self.TAKEOFF_ALT
                print(f"  >> GOTO     {unit} | flying to ({x:.0f}, {y:.0f}, {z:.0f}) ...")
                await loop.run_in_executor(None, self._do_goto, client, vname, x, y, z)
                print(f"  << GOTO     {unit} | position reached")
                return True

            if verb == "LAND":
                print(f"  >> LAND     {unit} | descending ...")
                await loop.run_in_executor(None, self._do_land, client, vname)
                print(f"  << LAND     {unit} | motors disarmed")
                return True

            if verb == "SCAN":
                area = " ".join(parts[2:]) if len(parts) > 2 else "AREA"
                print(f"  >> SCAN     {unit} | sensor sweep of {area} ...")
                await loop.run_in_executor(None, self._do_scan, client, vname)
                print(f"  << SCAN     {unit} | sweep complete")
                return True

            if verb == "DEPLOY":
                payload = " ".join(parts[2:]) if len(parts) > 2 else "PAYLOAD"
                print(f"  >> DEPLOY   {unit} | releasing {payload} ...")
                await loop.run_in_executor(None, self._do_deploy, client, vname)
                print(f"  << DEPLOY   {unit} | payload released")
                return True

            if verb == "STATUS":
                state = await loop.run_in_executor(None, client.getMultirotorState, vname)
                pos = state.kinematics_estimated.position
                logger.info(f"AirSim STATUS {vname}: pos=({pos.x_val:.1f},{pos.y_val:.1f},{pos.z_val:.1f})")
                return True

            logger.warning(f"AirSimActuator: unknown verb '{verb}'")
            return False

        except Exception as exc:
            logger.error(f"AirSimActuator: command '{raw}' failed: {exc}")
            return False

    # ── blocking helpers (run in executor) ───────────────────────────────────

    def _do_takeoff(self, client, vname: str) -> None:
        client.enableApiControl(True, vname)
        client.armDisarm(True, vname)
        client.takeoffAsync(vehicle_name=vname).join()
        client.moveToZAsync(self.TAKEOFF_ALT, self.CRUISE_SPEED, vehicle_name=vname).join()

    def _do_goto(self, client, vname: str, x: float, y: float, z: float) -> None:
        client.moveToPositionAsync(x, y, z, self.CRUISE_SPEED, vehicle_name=vname).join()

    def _do_land(self, client, vname: str) -> None:
        client.landAsync(vehicle_name=vname).join()
        client.armDisarm(False, vname)
        client.enableApiControl(False, vname)

    def _do_deploy(self, client, vname: str) -> None:
        import time
        client.hoverAsync(vehicle_name=vname).join()
        time.sleep(1.5)

    def _do_scan(self, client, vname: str) -> None:
        client.hoverAsync(vehicle_name=vname).join()
        responses = client.simGetImages(
            [self._airsim.ImageRequest("0", self._airsim.ImageType.Scene)],
            vehicle_name=vname,
        )
        if responses:
            logger.info(f"AirSim SCAN {vname}: captured {len(responses[0].image_data_uint8)} bytes")


class DeduplicatingActuator(Actuator):
    """
    Wraps any Actuator. Identical concurrent commands share one Future.
    TAKEOFF B1 issued by 5 agents simultaneously → executed once, all 5 share result.
    """

    def __init__(self, inner: Actuator) -> None:
        self._inner   = inner
        self._inflight: dict[str, asyncio.Future] = {}
        self._lock    = asyncio.Lock()

    async def execute_command(self, cmd_string: str) -> bool:
        key = cmd_string.strip().upper()
        async with self._lock:
            if key in self._inflight:
                fut = self._inflight[key]
            else:
                loop = asyncio.get_event_loop()
                fut  = loop.create_future()
                self._inflight[key] = fut
                asyncio.create_task(self._run(key, cmd_string, fut))
        return await asyncio.shield(fut)

    async def _run(self, key: str, cmd: str, fut: asyncio.Future) -> None:
        try:
            result = await self._inner.execute_command(cmd)
            if not fut.done():
                fut.set_result(result)
        except Exception as exc:
            if not fut.done():
                fut.set_exception(exc)
        finally:
            self._inflight.pop(key, None)
