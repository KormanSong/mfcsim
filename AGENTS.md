# Repository Guidelines

## Project Structure & Module Organization
- `main.py` provides the CLI entry point; keep quick demos or smoke tests here and defer reusable logic to the library.
- `mfcsimlib/` hosts reusable modules; prefer grouping related simulation helpers into cohesive files rather than expanding `main.py`.
- `mfcsimlib/config.py` is reserved for runtime settings; document defaults inside the module docstring and expose a `load_config()` helper when configuration logic lands.
- `mfcsimlib/pid_models.py` (add alongside new control logic) should encapsulate PID controllers plus interpolated FOPDT models so tuning logic stays testable.
- `mfcsimlib/simulation.py` centralizes 0–100% normalized configs/results and exposes anchor-based interpolation helpers for CLI/Dash reuse.
- `examples/` collects quick simulations such as `pid_fopdt_demo.py` that generate Plotly visual checks under `reports/`.
- `examples/dash_pid_tuner.py` spins up the interactive Dash tuning app; keep layout logic here and reuse library utilities for math.
- `config/user_prefs.json` persists UI choices (controller gains, slider maxima, simulation defaults, anchors) saved from the Dash tool.
- Add new assets or datasets under `assets/` (create on demand) to avoid mixing resources with code.

## Build, Test, and Development Commands
- `uv venv .venv && source .venv/bin/activate` – create a Python 3.12+ virtual environment managed by `uv` for reproducible tooling.
- `uv pip install -e .[dev]` – install the project in editable mode; define `[project.optional-dependencies.dev]` in `pyproject.toml` when new tooling is needed.
- `python main.py` – run the current simulation entry point.
- `python examples/pid_fopdt_demo.py` – run the controller/plant demo; inspect the resulting `reports/pid_fopdt_demo.html` for a quick visual sanity check.
- `python examples/dash_pid_tuner.py` – launch the Dash tuning UI (defaults to http://127.0.0.1:8050) for interactive parameter sweeps over 0–100% normalized flow.
- Inside the tuner, use the `Save Settings` button to persist current parameters to `config/user_prefs.json`; they reload automatically next run.
- Use the sliders in the controller card for quick Kc/Ti/Td sweeps; numeric inputs stay in sync for precise edits.
- `ruff check .` and `ruff format .` – lint and format once `ruff` is added to dev dependencies.

## Coding Style & Naming Conventions
- Follow `ruff` defaults (PEP 8) with 4-space indentation; prefer explicit imports and type hints on public functions.
- Name modules with lowercase underscores (`mfcsimlib/signal_router.py`), classes in `CamelCase`, and functions in `snake_case`.
- Keep public APIs documented with concise docstrings stating purpose, inputs, and side effects.

## Testing Guidelines
- Use `pytest` for unit and integration tests; place them under `tests/` mirroring the package layout (`tests/test_config.py`).
- Name test functions `test_<behavior>` and prefer fixtures for common setup.
- Run `pytest --maxfail=1 --disable-warnings -q` before pushing; target high coverage on `mfcsimlib/` modules where business logic lives.

## Commit & Pull Request Guidelines
- Use Imperative, present-tense commit subjects (`Add config loader`) and keep bodies for rationale or follow-up notes.
- Scope PRs narrowly with clear descriptions, testing notes, and links to related issues or tickets.
- Include CLI output or screenshots when behavior changes; summarize validation steps so reviewers can reproduce quickly.

## Environment & Security Notes
- Never commit secrets; rely on environment variables loaded via `.env` (list keys in `.env.example`).
- Pin external dependencies in `pyproject.toml` to guard against supply-chain drift and run `pip check` before release tags.
