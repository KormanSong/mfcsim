"""Simulation helpers for PID tuning and FOPDT plant evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .pid_models import DigitalIPDController, FOPDTAnchor, InterpolatedFOPDTModel


@dataclass
class ControllerParams:
    """Parameters for constructing a :class:`DigitalIPDController`."""

    kc: float
    ti: float
    td: float
    bias: float = 0.0
    u_min: Optional[float] = None
    u_max: Optional[float] = None


@dataclass
class AnchorParams:
    """Serializable representation of an :class:`FOPDTAnchor` (0-100% flow)."""

    flow_pct: float
    gain: float
    time_constant: float
    dead_time: float

    def to_anchor(self) -> FOPDTAnchor:
        return FOPDTAnchor(
            flow_pct=self.flow_pct,
            gain=self.gain,
            time_constant=self.time_constant,
            dead_time=self.dead_time,
        )


@dataclass
class SimulationConfig:
    """Configuration for a closed-loop step simulation (flows normalized 0-100%)."""

    duration: float = 10.0
    step_time: float = 1.0
    step_value: float = 60.0
    sample_rate_hz: float = 250.0
    controller: ControllerParams = field(
        default_factory=lambda: ControllerParams(kc=0.15, ti=5.0, td=0.0)
    )
    anchors: Sequence[AnchorParams] = field(
        default_factory=lambda: [
            AnchorParams(10.0, gain=10.0, time_constant=0.3, dead_time=0.2),
            AnchorParams(30.0, gain=5.0, time_constant=0.2, dead_time=0.15),
            AnchorParams(100.0, gain=10.0, time_constant=0.2, dead_time=0.08),
        ]
    )
    initial_output: float = 0.0
    initial_input: float = 0.0

    def build_anchors(self) -> List[FOPDTAnchor]:
        return [anchor.to_anchor() for anchor in self.anchors]


@dataclass
class SimulationResult:
    """Time-series results and metrics from a step simulation."""

    time: List[float]
    setpoint: List[float]
    output: List[float]
    control: List[float]
    metrics: Dict[str, float]


def run_step_response(config: SimulationConfig) -> SimulationResult:
    """Run a closed-loop step simulation and compute basic metrics."""

    sample_time = 1.0 / config.sample_rate_hz
    steps = max(1, int(config.duration * config.sample_rate_hz))

    controller = DigitalIPDController(
        kc=config.controller.kc,
        ti=config.controller.ti,
        td=config.controller.td,
        sample_time=sample_time,
        bias=config.controller.bias,
        u_min=config.controller.u_min,
        u_max=config.controller.u_max,
    )

    plant = InterpolatedFOPDTModel(
        anchors=config.build_anchors(),
        sample_time=sample_time,
        initial_output=config.initial_output,
    )

    plant.reset(output=config.initial_output, input_value=config.initial_input)
    controller.reset(initial_output=config.controller.bias, pv=plant.output)

    time: List[float] = []
    setpoints: List[float] = []
    outputs: List[float] = []
    controls: List[float] = []

    for index in range(steps):
        t = index * sample_time
        sp = config.step_value if t >= config.step_time else 0.0

        u = controller.update(sp, plant.output)
        y = plant.update(u)

        time.append(t)
        setpoints.append(sp)
        outputs.append(y)
        controls.append(u)

    metrics = compute_step_metrics(time, setpoints, outputs)
    return SimulationResult(time=time, setpoint=setpoints, output=outputs, control=controls, metrics=metrics)


def compute_step_metrics(time: Sequence[float], setpoint: Sequence[float], output: Sequence[float]) -> Dict[str, float]:
    """Compute step-response metrics for tuning guidance."""

    if not time:
        return {"response_time": 0.0, "overshoot_pct": 0.0}

    final_sp = setpoint[-1]
    if final_sp == 0:
        response_time = 0.0
    else:
        threshold = 0.98 * final_sp
        response_time = _find_first_crossing(time, output, threshold)

    peak_output = max(output)
    overshoot_pct = 0.0
    if final_sp != 0:
        overshoot_pct = max(0.0, (peak_output - final_sp) / abs(final_sp) * 100.0)

    return {
        "response_time": response_time,
        "overshoot_pct": overshoot_pct,
    }


def _find_first_crossing(time: Sequence[float], output: Sequence[float], threshold: float) -> float:
    comparison = (lambda y: y >= threshold) if threshold >= 0 else (lambda y: y <= threshold)
    for t, y in zip(time, output):
        if comparison(y):
            return float(t)
    return float(time[-1])


__all__ = [
    "ControllerParams",
    "AnchorParams",
    "SimulationConfig",
    "SimulationResult",
    "run_step_response",
    "compute_step_metrics",
]
