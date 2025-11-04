# mfcsim
<<<<<<< HEAD
Simulation of MFC(Mass Flow Controller)
=======

Interactive Mass Flow Controller (MFC) simulation playground featuring a digital I-PD controller and an anchor-interpolated first-order-plus-dead-time (FOPDT) process model. Flow is normalized to 0–100% so you can compare scenarios independent of absolute sccm ranges while specifying gains, time constants, and delays at key setpoints. Use the library to prototype new control logic and the Dash tooling to tune parameters against representative scenarios.

## Quick Start

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
python examples/dash_pid_tuner.py
```

Open the printed URL (defaults to http://127.0.0.1:8050/) to adjust controller gains (via synchronized sliders/inputs), anchor definitions, and review response time & overshoot metrics. Use **Save Settings** in the UI to persist your tuning choices to `config/user_prefs.json` so they reload next time. For a static HTML summary run:

```bash
python examples/pid_fopdt_demo.py
```

The script stores a Plotly visualization under `reports/pid_fopdt_demo.html` and prints headline performance metrics to the console.
>>>>>>> 6206433 (Initial commit)
