"""Utility helpers for loading and saving user preferences."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_PREFERENCES_PATH = Path("config/user_prefs.json")


@dataclass
class ControllerPrefs:
    kc: float
    ti: float
    td: float
    kc_max: float
    ti_max: float
    td_max: float
    bias: float
    u_min: float | None
    u_max: float | None


@dataclass
class SimulationPrefs:
    duration: float
    step_time: float
    step_value: float
    sample_rate_hz: float


@dataclass
class AnchorPrefs:
    flow_pct: float
    gain: float
    time_constant: float
    dead_time: float


@dataclass
class Preferences:
    controller: ControllerPrefs
    simulation: SimulationPrefs
    anchors: List[AnchorPrefs]

    @classmethod
    def defaults(cls) -> "Preferences":
        return cls(
            controller=ControllerPrefs(
                kc=0.15,
                ti=5.0,
                td=0.0,
                kc_max=5.0,
                ti_max=10.0,
                td_max=1.0,
                bias=0.0,
                u_min=0.0,
                u_max=100.0,
            ),
            simulation=SimulationPrefs(
                duration=10.0,
                step_time=1.0,
                step_value=60.0,
                sample_rate_hz=250.0,
            ),
            anchors=[
                AnchorPrefs(10.0, 10.0, 0.3, 0.2),
                AnchorPrefs(30.0, 5.0, 0.2, 0.15),
                AnchorPrefs(100.0, 10.0, 0.2, 0.08),
            ],
        )


def load_preferences(path: Path | None = None) -> Preferences:
    target = path or DEFAULT_PREFERENCES_PATH
    if not target.exists():
        return Preferences.defaults()

    try:
        payload = json.loads(target.read_text())
    except (json.JSONDecodeError, OSError):
        return Preferences.defaults()

    try:
        controller_data = payload["controller"]
        simulation_data = payload["simulation"]
        anchors_data = payload["anchors"]
    except KeyError:
        return Preferences.defaults()

    try:
        controller = ControllerPrefs(**controller_data)
        simulation = SimulationPrefs(**simulation_data)
        anchors = [AnchorPrefs(**row) for row in anchors_data]
    except TypeError:
        return Preferences.defaults()

    return Preferences(controller=controller, simulation=simulation, anchors=anchors)


def save_preferences(preferences: Preferences, path: Path | None = None) -> None:
    target = path or DEFAULT_PREFERENCES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "controller": asdict(preferences.controller),
        "simulation": asdict(preferences.simulation),
        "anchors": [asdict(anchor) for anchor in preferences.anchors],
    }
    target.write_text(json.dumps(payload, indent=2))


__all__ = [
    "Preferences",
    "ControllerPrefs",
    "SimulationPrefs",
    "AnchorPrefs",
    "load_preferences",
    "save_preferences",
    "DEFAULT_PREFERENCES_PATH",
]
