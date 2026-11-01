"""Physics-informed digital twin of a 220 kV GIS circuit breaker bay."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class AssetState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    ISOLATED = "isolated"


@dataclass
class GISBayState:
    timestamp: float = 0.0
    sf6_pressure_bar: float = 6.5
    temp_contact_c: float = 55.0
    temp_ambient_c: float = 32.0
    load_current_a: float = 750.0
    contact_wear_pct: float = 5.0
    vibration_mms: float = 0.7
    breaker_closed: bool = True
    disconnector_open: bool = False
    earthing_closed: bool = False
    state: AssetState = AssetState.HEALTHY
    active_faults: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "sf6_pressure_bar": round(self.sf6_pressure_bar, 3),
            "temp_contact_c": round(self.temp_contact_c, 2),
            "load_current_a": round(self.load_current_a, 1),
            "contact_wear_pct": round(self.contact_wear_pct, 2),
            "vibration_mms": round(self.vibration_mms, 3),
            "breaker_closed": self.breaker_closed,
            "disconnector_open": self.disconnector_open,
            "earthing_closed": self.earthing_closed,
            "state": self.state.value,
            "active_faults": self.active_faults.copy(),
        }


class GISBayTwin:
    """Simple but realistic physics simulation for a GIS bay."""

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.state = GISBayState()
        self._fault_params: Dict[str, float] = {}

    def inject_fault(self, fault_type: str, severity: float = 1.0):
        self._fault_params[fault_type] = severity
        if fault_type not in self.state.active_faults:
            self.state.active_faults.append(fault_type)

    def clear_fault(self, fault_type: str):
        self._fault_params.pop(fault_type, None)
        if fault_type in self.state.active_faults:
            self.state.active_faults.remove(fault_type)

    def step(self, load_a: Optional[float] = None) -> GISBayState:
        s = self.state
        s.timestamp += self.dt
        if load_a is not None:
            s.load_current_a = load_a

        # SF6 leak dynamics
        leak = self._fault_params.get("sf6_leak", 0.0)
        if leak > 0:
            s.sf6_pressure_bar -= 0.008 * leak * self.dt
        else:
            # very slow natural variation
            s.sf6_pressure_bar += np.random.normal(0, 0.001)

        # Contact temperature (I²R + ambient)
        i_pu = s.load_current_a / 800.0
        heating = 12.0 * (i_pu ** 2)
        if "contact_overheating" in self._fault_params:
            heating *= (1.0 + 1.8 * self._fault_params["contact_overheating"])
        s.temp_contact_c += 0.05 * (s.temp_ambient_c + heating - s.temp_contact_c) * self.dt

        # Wear accumulation
        if s.breaker_closed:
            s.contact_wear_pct += 0.0003 * (i_pu ** 1.5) * self.dt
            if "contact_overheating" in self._fault_params:
                s.contact_wear_pct += 0.01 * self._fault_params["contact_overheating"] * self.dt

        # Vibration
        s.vibration_mms = 0.6 + 0.3 * i_pu + np.random.normal(0, 0.05)
        if "mechanical_binding" in self._fault_params:
            s.vibration_mms += 1.2 * self._fault_params["mechanical_binding"]

        # State machine
        if s.sf6_pressure_bar < 5.2 or s.temp_contact_c > 105 or s.contact_wear_pct > 40:
            s.state = AssetState.CRITICAL
        elif s.sf6_pressure_bar < 5.8 or s.temp_contact_c > 85 or s.contact_wear_pct > 25:
            s.state = AssetState.DEGRADED
        elif s.disconnector_open and s.earthing_closed:
            s.state = AssetState.ISOLATED
        else:
            s.state = AssetState.HEALTHY

        return s

    def apply_action(self, action: str) -> str:
        """Apply operator / agent action. Returns status message."""
        s = self.state
        if action == "trip_breaker":
            s.breaker_closed = False
            return "Breaker tripped (open)"
        elif action == "open_disconnector":
            if not s.breaker_closed:
                s.disconnector_open = True
                return "Disconnector opened"
            return "REJECTED: Open breaker first"
        elif action == "close_earthing":
            if s.disconnector_open:
                s.earthing_closed = True
                return "Earthing switch closed – bay isolated"
            return "REJECTED: Open disconnector first"
        elif action == "reset_faults":
            self._fault_params.clear()
            s.active_faults.clear()
            return "Faults cleared (simulation reset)"
        return f"Unknown action: {action}"
