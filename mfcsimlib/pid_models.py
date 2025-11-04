"""Control and process-model utilities for the MFC simulator."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional


@dataclass
class DigitalIPDController:
    """Discrete velocity-form I-PD controller (P, D on PV; I on error).

    Parameters
    ----------
    kc:
        Controller gain.
    ti:
        Integral time constant (seconds). Use ``float("inf")`` to disable integral action.
    td:
        Derivative time constant (seconds). Use ``0`` to disable derivative action.
    sample_time:
        Controller execution period (seconds). Defaults to 1 / 250 Hz.
    bias:
        Output bias applied on reset.
    u_min, u_max:
        Optional actuator limits for the controller output.
    """

    kc: float
    ti: float
    td: float
    sample_time: float = 1.0 / 250.0
    bias: float = 0.0
    u_min: Optional[float] = None
    u_max: Optional[float] = None

    _output: float = field(init=False, default=0.0, repr=False)
    _pv_prev: float = field(init=False, default=0.0, repr=False)
    _pv_prev_prev: float = field(init=False, default=0.0, repr=False)
    _initialized: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.sample_time <= 0:
            raise ValueError("sample_time must be positive")
        if self.ti < 0:
            raise ValueError("ti must be non-negative")
        if self.td < 0:
            raise ValueError("td must be non-negative")
        if self.u_min is not None and self.u_max is not None and self.u_min > self.u_max:
            raise ValueError("u_min must be <= u_max")

    def reset(self, initial_output: Optional[float] = None, pv: float = 0.0) -> None:
        """Reset controller memory.

        Parameters
        ----------
        initial_output:
            Starting controller output (defaults to ``bias``).
        pv:
            Initial process variable measurement used to seed velocity terms.
        """

        self._output = self.bias if initial_output is None else initial_output
        self._pv_prev = pv
        self._pv_prev_prev = pv
        self._initialized = True

    @property
    def output(self) -> float:
        """Return the most recent controller output."""

        return self._output

    def update(self, setpoint: float, pv: float) -> float:
        """Advance the controller one sample and return the new output."""

        if not self._initialized:
            self.reset(pv=pv)

        error = setpoint - pv
        pv_delta = pv - self._pv_prev
        pv_accel = pv - 2.0 * self._pv_prev + self._pv_prev_prev

        integral_gain = 0.0 if self.ti == 0.0 or math.isinf(self.ti) else self.sample_time / self.ti
        derivative_gain = 0.0 if self.td == 0.0 else self.td / self.sample_time

        integral_term = integral_gain * error
        proportional_term = -self.kc * pv_delta
        derivative_term = -self.kc * derivative_gain * pv_accel

        delta_u = integral_term + proportional_term + derivative_term

        candidate_output = self._output + delta_u
        if self.u_min is not None:
            candidate_output = max(self.u_min, candidate_output)
        if self.u_max is not None:
            candidate_output = min(self.u_max, candidate_output)

        self._output = candidate_output
        self._pv_prev_prev = self._pv_prev
        self._pv_prev = pv

        return self._output


@dataclass(frozen=True)
class FOPDTSegment:
    """Parameter slice for a first-order plus dead-time model."""

    flow_min: float
    flow_max: float
    gain: float
    time_constant: float
    dead_time: float

    def __post_init__(self) -> None:
        if self.flow_max <= self.flow_min:
            raise ValueError("flow_max must be greater than flow_min")
        if self.time_constant < 0:
            raise ValueError("time_constant must be non-negative")
        if self.dead_time < 0:
            raise ValueError("dead_time must be non-negative")

    def contains(self, flow: float, *, inclusive_upper: bool = False) -> bool:
        upper_check = flow <= self.flow_max if inclusive_upper else flow < self.flow_max
        return self.flow_min <= flow and upper_check


@dataclass
class SegmentedFOPDTModel:
    """Piecewise FOPDT model selecting parameters per flow interval."""

    segments: Iterable[FOPDTSegment]
    sample_time: float = 1.0 / 250.0
    initial_output: float = 0.0

    _segments: List[FOPDTSegment] = field(init=False, repr=False)
    _output: float = field(init=False, default=0.0, repr=False)
    _max_dead_steps: int = field(init=False, default=0, repr=False)
    _input_history: Deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        segments = list(self.segments)
        if not segments:
            raise ValueError("segments must contain at least one FOPDTSegment")
        if self.sample_time <= 0:
            raise ValueError("sample_time must be positive")

        segments.sort(key=lambda s: (s.flow_min, s.flow_max))
        self._segments = segments

        max_dead_time = max(segment.dead_time for segment in segments)
        self._max_dead_steps = max(0, int(round(max_dead_time / self.sample_time)))
        self._input_history = deque([0.0] * (self._max_dead_steps + 1), maxlen=self._max_dead_steps + 1)
        self.reset()

    @property
    def output(self) -> float:
        """Return the latest process output value."""

        return self._output

    def reset(self, output: Optional[float] = None, input_value: float = 0.0) -> None:
        """Reset model state and input history."""

        self._output = self.initial_output if output is None else output
        self._input_history.clear()
        self._input_history.extend([input_value] * (self._max_dead_steps + 1))

    def update(self, input_signal: float, flow_reference: Optional[float] = None) -> float:
        """Advance the model one sample and return the new output."""

        flow_reference = self._output if flow_reference is None else flow_reference
        segment = self._select_segment(flow_reference)

        self._input_history.append(input_signal)
        dead_steps = int(round(segment.dead_time / self.sample_time))
        delayed_input = self._input_history[-(dead_steps + 1)] if dead_steps > 0 else input_signal

        alpha = 1.0 if segment.time_constant <= 0 else min(self.sample_time / segment.time_constant, 1.0)
        target = segment.gain * delayed_input
        self._output += alpha * (target - self._output)
        return self._output

    def _select_segment(self, flow: float) -> FOPDTSegment:
        for idx, segment in enumerate(self._segments):
            inclusive_upper = idx == len(self._segments) - 1
            if segment.contains(flow, inclusive_upper=inclusive_upper):
                return segment
        return self._segments[-1]


@dataclass(frozen=True)
class FOPDTAnchor:
    """Parameter anchor used for interpolation across flow percentages."""

    flow_pct: float
    gain: float
    time_constant: float
    dead_time: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.flow_pct <= 100.0:
            raise ValueError("flow_pct must be within 0-100%")
        if self.time_constant < 0:
            raise ValueError("time_constant must be non-negative")
        if self.dead_time < 0:
            raise ValueError("dead_time must be non-negative")


@dataclass
class InterpolatedFOPDTModel:
    """FOPDT model with parameters interpolated between anchor points."""

    anchors: Iterable[FOPDTAnchor]
    sample_time: float = 1.0 / 250.0
    initial_output: float = 0.0

    _anchors: List[FOPDTAnchor] = field(init=False, repr=False)
    _output: float = field(init=False, default=0.0, repr=False)
    _max_dead_steps: int = field(init=False, default=0, repr=False)
    _input_history: Deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        anchors = sorted(self.anchors, key=lambda anchor: anchor.flow_pct)
        if len(anchors) < 2:
            raise ValueError("anchors must contain at least two points for interpolation")
        for idx in range(1, len(anchors)):
            if math.isclose(anchors[idx - 1].flow_pct, anchors[idx].flow_pct):
                raise ValueError("anchors must have strictly increasing flow_pct values")
        if self.sample_time <= 0:
            raise ValueError("sample_time must be positive")

        self._anchors = anchors

        max_dead_time = max(anchor.dead_time for anchor in anchors)
        self._max_dead_steps = max(0, int(round(max_dead_time / self.sample_time)))
        self._input_history = deque([0.0] * (self._max_dead_steps + 1), maxlen=self._max_dead_steps + 1)
        self.reset()

    @property
    def output(self) -> float:
        """Return the latest process output value."""

        return self._output

    def reset(self, output: Optional[float] = None, input_value: float = 0.0) -> None:
        """Reset model state and input history."""

        self._output = self.initial_output if output is None else output
        self._input_history.clear()
        self._input_history.extend([input_value] * (self._max_dead_steps + 1))

    def update(self, input_signal: float, flow_reference: Optional[float] = None) -> float:
        """Advance the model one sample and return the new output."""

        flow_reference = self._output if flow_reference is None else flow_reference
        gain, time_constant, dead_time = self._interpolate_params(flow_reference)

        self._input_history.append(input_signal)
        dead_steps = int(round(dead_time / self.sample_time))
        dead_steps = min(dead_steps, self._max_dead_steps)
        delayed_input = self._input_history[-(dead_steps + 1)] if dead_steps > 0 else input_signal

        alpha = 1.0 if time_constant <= 0 else min(self.sample_time / time_constant, 1.0)
        target = gain * delayed_input
        self._output += alpha * (target - self._output)
        return self._output

    def _interpolate_params(self, flow_pct: float) -> tuple[float, float, float]:
        anchors = self._anchors
        if flow_pct <= anchors[0].flow_pct:
            anchor = anchors[0]
            return anchor.gain, anchor.time_constant, anchor.dead_time
        if flow_pct >= anchors[-1].flow_pct:
            anchor = anchors[-1]
            return anchor.gain, anchor.time_constant, anchor.dead_time

        for lower, upper in zip(anchors[:-1], anchors[1:]):
            if lower.flow_pct <= flow_pct <= upper.flow_pct:
                span = upper.flow_pct - lower.flow_pct
                if span <= 0:
                    return upper.gain, upper.time_constant, upper.dead_time
                ratio = (flow_pct - lower.flow_pct) / span
                gain = lower.gain + ratio * (upper.gain - lower.gain)
                time_constant = lower.time_constant + ratio * (upper.time_constant - lower.time_constant)
                dead_time = lower.dead_time + ratio * (upper.dead_time - lower.dead_time)
                return gain, time_constant, dead_time

        last = anchors[-1]
        return last.gain, last.time_constant, last.dead_time


__all__ = [
    "DigitalIPDController",
    "FOPDTSegment",
    "SegmentedFOPDTModel",
    "FOPDTAnchor",
    "InterpolatedFOPDTModel",
]
