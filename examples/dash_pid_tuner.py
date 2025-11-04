"""Dash application for interactive PID + interpolated FOPDT tuning."""

from __future__ import annotations

import io
from datetime import datetime
from dataclasses import asdict
from typing import Any, Dict, List

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, dash_table, no_update
from dash.exceptions import PreventUpdate

from mfcsimlib.preferences import (
    Preferences,
    ControllerPrefs,
    SimulationPrefs,
    AnchorPrefs,
    load_preferences,
    save_preferences,
)
from mfcsimlib.simulation import (
    AnchorParams,
    ControllerParams,
    SimulationConfig,
    SimulationResult,
    run_step_response,
)


def _default_anchor_records(config: SimulationConfig) -> List[Dict[str, Any]]:
    return [
        {
            "flow_pct": anchor.flow_pct,
            "gain": anchor.gain,
            "time_constant": anchor.time_constant,
            "dead_time": anchor.dead_time,
        }
        for anchor in config.anchors
    ]


def _build_figure(result: SimulationResult) -> go.Figure:
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
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig


def _format_metrics(metrics: Dict[str, float]) -> html.Table:
    label_map = {
        "response_time": "Response Time (s)",
        "overshoot_pct": "Overshoot (%)",
    }
    rows = []
    for key in ["response_time", "overshoot_pct"]:
        value = metrics.get(key, float("nan"))
        rows.append(
            html.Tr([
                html.Th(label_map.get(key, key)),
                html.Td(f"{value:.4f}"),
            ])
        )
    return html.Table(rows, className="metrics-table")


def _result_to_payload(result: SimulationResult) -> Dict[str, Any]:
    return {
        "time": result.time,
        "setpoint": result.setpoint,
        "output": result.output,
        "control": result.control,
        "metrics": result.metrics,
    }


def _parse_float(value: Any) -> float:
    if value is None or value == "":
        raise ValueError("Missing numeric value")
    return float(value)


def _parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _sanitize_max(value: Any, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return numeric if numeric > 0 else fallback


def _preferences_to_payload(prefs: Preferences) -> Dict[str, Any]:
    return {
        "controller": asdict(prefs.controller),
        "simulation": asdict(prefs.simulation),
        "anchors": [asdict(anchor) for anchor in prefs.anchors],
    }


def build_controller_row(
    *,
    label: str,
    slider_id: str,
    input_id: str,
    display_id: str,
    min_value: float,
    max_value: float,
    step: float,
    value: float,
) -> html.Div:
    return html.Div(
        [
            html.Label(label, className="controller-label"),
            html.Span(
                dcc.Input(
                    id=slider_id,
                    type="range",
                    min=min_value,
                    max=max_value,
                    step=step,
                    value=value,
                    className="controller-range",
                ),
                className="controller-range-wrapper",
            ),
            html.Div(
                [
                    html.Span(f"{value:.3f}", id=display_id, className="controller-display"),
                    dcc.Input(
                        id=input_id,
                        type="number",
                        value=value,
                        step=step,
                        className="controller-input",
                    ),
                ],
                className="controller-value-row",
            ),
        ],
        className="controller-row",
    )


def _load_anchors(rows: List[Dict[str, Any]]) -> List[AnchorParams]:
    anchors: List[AnchorParams] = []
    for idx, row in enumerate(rows):
        try:
            anchor = AnchorParams(
                flow_pct=float(row["flow_pct"]),
                gain=float(row["gain"]),
                time_constant=float(row["time_constant"]),
                dead_time=float(row["dead_time"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid anchor row {idx + 1}: {exc}") from exc
        if not 0.0 <= anchor.flow_pct <= 100.0:
            raise ValueError(f"Anchor {idx + 1}: flow_pct must be within 0-100%")
        anchors.append(anchor)

    anchors.sort(key=lambda a: a.flow_pct)
    if len(anchors) < 2:
        raise ValueError("At least two anchors are required for interpolation")

    for prev, curr in zip(anchors, anchors[1:]):
        if curr.flow_pct <= prev.flow_pct:
            raise ValueError("Anchor flow_pct values must be strictly increasing")

    return anchors


def _append_anchor(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return [
            {
                "flow_pct": 10.0,
                "gain": 0.8,
                "time_constant": 0.3,
                "dead_time": 0.08,
            },
            {
                "flow_pct": 30.0,
                "gain": 0.85,
                "time_constant": 0.35,
                "dead_time": 0.1,
            },
            {
                "flow_pct": 100.0,
                "gain": 0.95,
                "time_constant": 0.45,
                "dead_time": 0.12,
            },
        ]

    rows = list(rows)
    if len(rows) == 1:
        base = float(rows[0]["flow_pct"])
        next_flow = min(base + 10.0, 100.0)
        if next_flow <= base:
            return rows
        rows.append(
            {
                "flow_pct": next_flow,
                "gain": float(rows[0].get("gain", 1.0)),
                "time_constant": float(rows[0].get("time_constant", 0.3)),
                "dead_time": float(rows[0].get("dead_time", 0.05)),
            }
        )
        return rows

    flows = [float(row["flow_pct"]) for row in rows]
    gaps = [upper - lower for lower, upper in zip(flows, flows[1:])]
    max_gap_idx = max(range(len(gaps)), key=lambda idx: gaps[idx], default=None)
    if max_gap_idx is None:
        return rows

    gap = gaps[max_gap_idx]
    if gap <= 0.0:
        return rows

    lower_row = rows[max_gap_idx]
    upper_row = rows[max_gap_idx + 1]
    new_flow = flows[max_gap_idx] + gap / 2.0
    if new_flow <= flows[max_gap_idx] or new_flow >= flows[max_gap_idx + 1]:
        return rows

    lower_gain = float(lower_row.get("gain", 1.0))
    upper_gain = float(upper_row.get("gain", 1.0))
    lower_tc = float(lower_row.get("time_constant", 0.3))
    upper_tc = float(upper_row.get("time_constant", 0.3))
    lower_dead = float(lower_row.get("dead_time", 0.05))
    upper_dead = float(upper_row.get("dead_time", 0.05))

    new_row = {
        "flow_pct": new_flow,
        "gain": lower_gain + (upper_gain - lower_gain) / 2.0,
        "time_constant": lower_tc + (upper_tc - lower_tc) / 2.0,
        "dead_time": lower_dead + (upper_dead - lower_dead) / 2.0,
    }
    rows.insert(max_gap_idx + 1, new_row)
    return rows


_PREFERENCES = load_preferences()

DEFAULT_CONFIG = SimulationConfig(
    duration=_PREFERENCES.simulation.duration,
    step_time=_PREFERENCES.simulation.step_time,
    step_value=_PREFERENCES.simulation.step_value,
    sample_rate_hz=_PREFERENCES.simulation.sample_rate_hz,
    controller=ControllerParams(
        kc=_PREFERENCES.controller.kc,
        ti=_PREFERENCES.controller.ti,
        td=_PREFERENCES.controller.td,
        bias=_PREFERENCES.controller.bias,
        u_min=_PREFERENCES.controller.u_min,
        u_max=_PREFERENCES.controller.u_max,
    ),
    anchors=[
        AnchorParams(
            flow_pct=anchor.flow_pct,
            gain=anchor.gain,
            time_constant=anchor.time_constant,
            dead_time=anchor.dead_time,
        )
        for anchor in _PREFERENCES.anchors
    ],
)

DEFAULT_RESULT = run_step_response(DEFAULT_CONFIG)

DEFAULT_CONTROLLER_MAX = {
    "kc": _PREFERENCES.controller.kc_max,
    "ti": _PREFERENCES.controller.ti_max,
    "td": _PREFERENCES.controller.td_max,
}

app = Dash(__name__)
app.title = "MFC PID Tuning"

app.layout = html.Div(
    [
        html.H1("MFC PID Tuning Tool", className="page-title"),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="step-graph", figure=_build_figure(DEFAULT_RESULT)),
                        html.Div(id="metrics-panel", children=_format_metrics(DEFAULT_RESULT.metrics), className="metrics-card"),
                    ],
                    className="results-card",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2("Controller", className="card-title"),
                                html.Div(
                                    [
                                        build_controller_row(
                                            label="Kc",
                                            slider_id="kc-slider",
                                            input_id="kc",
                                            display_id="kc-display",
                                            min_value=0.0,
                                            max_value=DEFAULT_CONTROLLER_MAX["kc"],
                                            step=0.01,
                                            value=DEFAULT_CONFIG.controller.kc,
                                        ),
                                        build_controller_row(
                                            label="Ti (s)",
                                            slider_id="ti-slider",
                                            input_id="ti",
                                            display_id="ti-display",
                                            min_value=0.0,
                                            max_value=DEFAULT_CONTROLLER_MAX["ti"],
                                            step=0.05,
                                            value=DEFAULT_CONFIG.controller.ti,
                                        ),
                                        build_controller_row(
                                            label="Td (s)",
                                            slider_id="td-slider",
                                            input_id="td",
                                            display_id="td-display",
                                            min_value=0.0,
                                            max_value=DEFAULT_CONTROLLER_MAX["td"],
                                            step=0.005,
                                            value=DEFAULT_CONFIG.controller.td,
                                        ),
                                        html.Div(
                                            [
                                                html.Div([
                                                    html.Label("Bias"),
                                                    dcc.Input(id="bias", type="number", value=DEFAULT_CONFIG.controller.bias, step=0.1),
                                                ], className="control-field"),
                                                html.Div([
                                                    html.Label("u_min"),
                                                    dcc.Input(id="u_min", type="number", value=DEFAULT_CONFIG.controller.u_min, step=0.1),
                                                ], className="control-field"),
                                                html.Div([
                                                    html.Label("u_max"),
                                                    dcc.Input(id="u_max", type="number", value=DEFAULT_CONFIG.controller.u_max, step=0.1),
                                                ], className="control-field"),
                                                html.Div([
                                                    html.Label("Kc Max"),
                                                    dcc.Input(id="kc-max", type="number", value=DEFAULT_CONTROLLER_MAX["kc"], min=0, step=0.1, className="table-input"),
                                                ], className="control-field"),
                                                html.Div([
                                                    html.Label("Ti Max"),
                                                    dcc.Input(id="ti-max", type="number", value=DEFAULT_CONTROLLER_MAX["ti"], min=0, step=0.1, className="table-input"),
                                                ], className="control-field"),
                                                html.Div([
                                                    html.Label("Td Max"),
                                                    dcc.Input(id="td-max", type="number", value=DEFAULT_CONTROLLER_MAX["td"], min=0, step=0.01, className="table-input"),
                                                ], className="control-field"),
                                            ],
                                            className="uniform-grid",
                                        ),
                                    ],
                                    className="uniform-grid",
                                ),
                            ],
                            className="card controller-card",
                        ),
                        html.Div(
                            [
                                html.H2("Simulation", className="card-title"),
                                html.Table(
                                    [
                                        html.Thead(
                                            html.Tr(
                                                [
                                                    html.Th("Parameter"),
                                                    html.Th("Value"),
                                                ]
                                            )
                                        ),
                                        html.Tbody(
                                            [
                                                html.Tr([
                                                    html.Td("Duration (s)"),
                                                    html.Td(dcc.Input(id="duration", type="number", value=DEFAULT_CONFIG.duration, min=0.1, step=0.5, className="table-input")),
                                                ]),
                                                html.Tr([
                                                    html.Td("Step Time (s)"),
                                                    html.Td(dcc.Input(id="step_time", type="number", value=DEFAULT_CONFIG.step_time, min=0, step=0.1, className="table-input")),
                                                ]),
                                                html.Tr([
                                                    html.Td("Step Value (%)"),
                                                    html.Td(dcc.Input(id="step_value", type="number", value=DEFAULT_CONFIG.step_value, step=1.0, className="table-input")),
                                                ]),
                                                html.Tr([
                                                    html.Td("Sample Rate (Hz)"),
                                                    html.Td(dcc.Input(id="sample_rate", type="number", value=DEFAULT_CONFIG.sample_rate_hz, min=1, step=10, className="table-input")),
                                                ]),
                                            ]
                                        ),
                                    ],
                                    className="sim-table",
                                ),
                                html.Div(
                                    [
                                        html.Button("Run Simulation", id="run-button", n_clicks=0, className="primary"),
                                        html.Button("Add Anchor", id="add-anchor", n_clicks=0),
                                        html.Button("Download CSV", id="download-button", n_clicks=0),
                                        html.Button("Save Settings", id="save-settings", n_clicks=0, className="secondary"),
                                    ],
                                    className="sim-actions",
                                ),
                                html.Div(id="save-status", className="status"),
                                html.Div(id="status-message", className="status", children="Ready."),
                            ],
                            className="card sim-card",
                        ),
                    ],
                    className="side-panel",
                ),
            ],
            className="layout-grid",
        ),
        html.Div(
            [
                html.H2("Anchors", className="card-title"),
                dash_table.DataTable(
                    id="anchors-table",
                    data=_default_anchor_records(DEFAULT_CONFIG),
                    editable=True,
                    row_deletable=True,
                    columns=[
                        {"name": "Flow %", "id": "flow_pct", "type": "numeric"},
                        {"name": "Gain", "id": "gain", "type": "numeric"},
                        {"name": "Time Constant", "id": "time_constant", "type": "numeric"},
                        {"name": "Dead Time", "id": "dead_time", "type": "numeric"},
                    ],
                    style_table={"overflowX": "auto"},
                ),
            ],
            className="card anchors-card",
        ),
        dcc.Store(id="result-store", data=_result_to_payload(DEFAULT_RESULT)),
        dcc.Store(id="prefs-store", data=_preferences_to_payload(_PREFERENCES)),
        dcc.Download(id="download-data"),
    ]
)


@app.callback(
    Output("anchors-table", "data"),
    Input("add-anchor", "n_clicks"),
    State("anchors-table", "data"),
    prevent_initial_call=True,
)
def handle_add_anchor(n_clicks: int, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if n_clicks <= 0:
        raise PreventUpdate
    return _append_anchor(list(rows or []))


@app.callback(
    Output("kc", "value"),
    Output("kc-slider", "value"),
    Output("kc-slider", "max"),
    Output("kc-display", "children"),
    Input("kc-slider", "value"),
    Input("kc", "value"),
    Input("kc-max", "value"),
    prevent_initial_call=True,
)
def sync_kc(slider_value: Any, input_value: Any, max_value: Any):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    max_numeric = _sanitize_max(max_value, DEFAULT_CONTROLLER_MAX["kc"])

    def _current() -> float:
        try:
            if slider_value not in (None, ""):
                return float(slider_value)
        except (TypeError, ValueError):
            pass
        try:
            if input_value not in (None, ""):
                return float(input_value)
        except (TypeError, ValueError):
            pass
        return 0.0

    triggered = ctx.triggered_id
    if triggered == "kc-slider":
        if slider_value is None:
            raise PreventUpdate
        try:
            numeric = float(slider_value)
        except (TypeError, ValueError) as exc:
            raise PreventUpdate from exc
    elif triggered == "kc":
        if input_value in (None, ""):
            raise PreventUpdate
        try:
            numeric = float(input_value)
        except (TypeError, ValueError) as exc:
            raise PreventUpdate from exc
    elif triggered == "kc-max":
        numeric = _current()
    else:
        raise PreventUpdate

    numeric = _clamp(numeric, 0.0, max_numeric)
    display = f"{numeric:.3f}"
    return numeric, numeric, max_numeric, display


@app.callback(
    Output("ti", "value"),
    Output("ti-slider", "value"),
    Output("ti-slider", "max"),
    Output("ti-display", "children"),
    Input("ti-slider", "value"),
    Input("ti", "value"),
    Input("ti-max", "value"),
    prevent_initial_call=True,
)
def sync_ti(slider_value: Any, input_value: Any, max_value: Any):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    max_numeric = _sanitize_max(max_value, DEFAULT_CONTROLLER_MAX["ti"])

    def _current() -> float:
        try:
            if slider_value not in (None, ""):
                return float(slider_value)
        except (TypeError, ValueError):
            pass
        try:
            if input_value not in (None, ""):
                return float(input_value)
        except (TypeError, ValueError):
            pass
        return 0.0

    triggered = ctx.triggered_id
    if triggered == "ti-slider":
        if slider_value is None:
            raise PreventUpdate
        try:
            numeric = float(slider_value)
        except (TypeError, ValueError) as exc:
            raise PreventUpdate from exc
    elif triggered == "ti":
        if input_value in (None, ""):
            raise PreventUpdate
        try:
            numeric = float(input_value)
        except (TypeError, ValueError) as exc:
            raise PreventUpdate from exc
    elif triggered == "ti-max":
        numeric = _current()
    else:
        raise PreventUpdate

    numeric = _clamp(numeric, 0.0, max_numeric)
    display = f"{numeric:.3f}"
    return numeric, numeric, max_numeric, display


@app.callback(
    Output("td", "value"),
    Output("td-slider", "value"),
    Output("td-slider", "max"),
    Output("td-display", "children"),
    Input("td-slider", "value"),
    Input("td", "value"),
    Input("td-max", "value"),
    prevent_initial_call=True,
)
def sync_td(slider_value: Any, input_value: Any, max_value: Any):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    max_numeric = _sanitize_max(max_value, DEFAULT_CONTROLLER_MAX["td"])

    def _current() -> float:
        try:
            if slider_value not in (None, ""):
                return float(slider_value)
        except (TypeError, ValueError):
            pass
        try:
            if input_value not in (None, ""):
                return float(input_value)
        except (TypeError, ValueError):
            pass
        return 0.0

    triggered = ctx.triggered_id
    if triggered == "td-slider":
        if slider_value is None:
            raise PreventUpdate
        try:
            numeric = float(slider_value)
        except (TypeError, ValueError) as exc:
            raise PreventUpdate from exc
    elif triggered == "td":
        if input_value in (None, ""):
            raise PreventUpdate
        try:
            numeric = float(input_value)
        except (TypeError, ValueError) as exc:
            raise PreventUpdate from exc
    elif triggered == "td-max":
        numeric = _current()
    else:
        raise PreventUpdate

    numeric = _clamp(numeric, 0.0, max_numeric)
    display = f"{numeric:.3f}"
    return numeric, numeric, max_numeric, display


@app.callback(
    Output("step-graph", "figure"),
    Output("metrics-panel", "children"),
    Output("status-message", "children"),
    Output("result-store", "data"),
    Input("run-button", "n_clicks"),
    Input("kc-slider", "value"),
    Input("ti-slider", "value"),
    Input("td-slider", "value"),
    State("bias", "value"),
    State("u_min", "value"),
    State("u_max", "value"),
    State("duration", "value"),
    State("step_time", "value"),
    State("step_value", "value"),
    State("sample_rate", "value"),
    State("anchors-table", "data"),
    prevent_initial_call=False,
)
def run_simulation_callback(
    _n_clicks: int,
    kc_value: Any,
    ti_value: Any,
    td_value: Any,
    bias: Any,
    u_min: Any,
    u_max: Any,
    duration: Any,
    step_time: Any,
    step_value: Any,
    sample_rate: Any,
    anchors_rows: List[Dict[str, Any]],
):
    try:
        controller = ControllerParams(
            kc=_parse_float(kc_value),
            ti=_parse_float(ti_value),
            td=_parse_float(td_value),
            bias=_parse_float(bias),
            u_min=_parse_optional_float(u_min),
            u_max=_parse_optional_float(u_max),
        )
        config = SimulationConfig(
            duration=_parse_float(duration),
            step_time=_parse_float(step_time),
            step_value=_parse_float(step_value),
            sample_rate_hz=_parse_float(sample_rate),
            controller=controller,
            anchors=_load_anchors(list(anchors_rows or [])),
        )
        result = run_step_response(config)
    except ValueError as exc:  # invalid input or simulation configuration
        return no_update, no_update, f"Error: {exc}", no_update

    figure = _build_figure(result)
    metrics_table = _format_metrics(result.metrics)
    status = f"Last updated {datetime.now().strftime('%H:%M:%S')}"
    return figure, metrics_table, status, _result_to_payload(result)


@app.callback(
    Output("save-status", "children"),
    Output("prefs-store", "data"),
    Input("save-settings", "n_clicks"),
    State("kc", "value"),
    State("ti", "value"),
    State("td", "value"),
    State("kc-max", "value"),
    State("ti-max", "value"),
    State("td-max", "value"),
    State("bias", "value"),
    State("u_min", "value"),
    State("u_max", "value"),
    State("duration", "value"),
    State("step_time", "value"),
    State("step_value", "value"),
    State("sample_rate", "value"),
    State("anchors-table", "data"),
    prevent_initial_call=True,
)
def save_settings(
    n_clicks: int,
    kc_value: Any,
    ti_value: Any,
    td_value: Any,
    kc_max: Any,
    ti_max: Any,
    td_max: Any,
    bias_value: Any,
    u_min_value: Any,
    u_max_value: Any,
    duration_value: Any,
    step_time_value: Any,
    step_value_value: Any,
    sample_rate_value: Any,
    anchors_rows: List[Dict[str, Any]],
):
    if not n_clicks:
        raise PreventUpdate

    try:
        controller = ControllerPrefs(
            kc=_parse_float(kc_value),
            ti=_parse_float(ti_value),
            td=_parse_float(td_value),
            kc_max=_sanitize_max(kc_max, DEFAULT_CONTROLLER_MAX["kc"]),
            ti_max=_sanitize_max(ti_max, DEFAULT_CONTROLLER_MAX["ti"]),
            td_max=_sanitize_max(td_max, DEFAULT_CONTROLLER_MAX["td"]),
            bias=_parse_float(bias_value),
            u_min=_parse_optional_float(u_min_value),
            u_max=_parse_optional_float(u_max_value),
        )
        simulation = SimulationPrefs(
            duration=_parse_float(duration_value),
            step_time=_parse_float(step_time_value),
            step_value=_parse_float(step_value_value),
            sample_rate_hz=_parse_float(sample_rate_value),
        )
        anchor_params = _load_anchors(list(anchors_rows or []))
        anchors = [
            AnchorPrefs(
                flow_pct=anchor.flow_pct,
                gain=anchor.gain,
                time_constant=anchor.time_constant,
                dead_time=anchor.dead_time,
            )
            for anchor in anchor_params
        ]
    except ValueError as exc:
        return f"Save failed: {exc}", no_update

    prefs = Preferences(controller=controller, simulation=simulation, anchors=anchors)
    save_preferences(prefs)

    global _PREFERENCES
    _PREFERENCES = prefs

    DEFAULT_CONTROLLER_MAX["kc"] = controller.kc_max
    DEFAULT_CONTROLLER_MAX["ti"] = controller.ti_max
    DEFAULT_CONTROLLER_MAX["td"] = controller.td_max

    timestamp = datetime.now().strftime("%H:%M:%S")
    return f"Settings saved at {timestamp}", _preferences_to_payload(prefs)


@app.callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    State("result-store", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks: int, payload: Dict[str, Any]):
    if n_clicks <= 0:
        raise PreventUpdate
    if not payload:
        raise PreventUpdate

    buffer = io.StringIO()
    buffer.write("time,setpoint,output,control\n")
    for row in zip(payload["time"], payload["setpoint"], payload["output"], payload["control"]):
        buffer.write(",".join(f"{value}" for value in row))
        buffer.write("\n")

    filename = f"pid_fopdt_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return dict(content=buffer.getvalue(), filename=filename)


if __name__ == "__main__":
    app.run(debug=True)
