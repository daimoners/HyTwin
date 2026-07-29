"""dashboard/app.py — compatibility re-export (delegates to network_app)."""
from dashboard.network_app import (  # noqa: F401
    run_network_dashboard as run,
    create_app,
    NetworkSimulationWorker as SimulationWorker,
)
