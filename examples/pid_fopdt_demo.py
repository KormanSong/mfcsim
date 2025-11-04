"""Quick simulation to visualize the digital I-PD controller with an interpolated FOPDT plant."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from mfcsimlib.simulation import (
    AnchorParams,
    ControllerParams,
    SimulationConfig,
    SimulationResult,
    run_step_response,
)


def run_simulation(
    duration: float = 10.0,
    step_time: float = 1.0,
    step_value: float = 60.0,
    sample_rate_hz: float = 250.0,
) -> SimulationResult:
    """Simulate a step response and return the structured result."""

    config = SimulationConfig(
        duration=duration,
        step_time=step_time,
        step_value=step_value,
        sample_rate_hz=sample_rate_hz,
        controller=ControllerParams(kc=0.15, ti=5.0, td=0.0, bias=0.0, u_min=0.0, u_max=100.0),
        anchors=[
            AnchorParams(10.0, gain=10.0, time_constant=0.3, dead_time=0.2),
            AnchorParams(30.0, gain=5.0, time_constant=0.2, dead_time=0.15),
            AnchorParams(100.0, gain=10.0, time_constant=0.2, dead_time=0.08),
        ],
    )
    return run_step_response(config)


def create_figure(result: SimulationResult) -> go.Figure:
    """Build a Plotly figure for the PID vs plant step response."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Setpoint",
            x=result.time,
            y=result.setpoint,
            mode="lines",
            line=dict(color="#2c7fb8", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Plant Output",
            x=result.time,
            y=result.output,
            mode="lines",
            line=dict(color="#31a354"),
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Controller Output",
            x=result.time,
            y=result.control,
            mode="lines",
            line=dict(color="#b30000"),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Digital I-PD Controller vs Interpolated FOPDT Plant",
        xaxis_title="Time (s)",
        yaxis=dict(title="Flow (%)"),
        yaxis2=dict(title="Controller Output", overlaying="y", side="right", showgrid=False),
        template="plotly_white",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1),
    )
    return fig


def main() -> None:
    result = run_simulation()
    figure = create_figure(result)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    output_path = reports_dir / "pid_fopdt_demo.html"
    figure.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Saved PID/FOPDT demo plot to {output_path}")
    print("Step metrics:")
    for name, value in sorted(result.metrics.items()):
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
