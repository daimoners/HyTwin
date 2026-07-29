"""HyTwin Dashboard — multi-site Network Control Room.

All functionality lives in ``dashboard.network_app``.  This module re-exports
the public API so that ``from dashboard import create_app, run`` keeps working.
"""

from .network_app import (  # noqa: F401
    create_app,
    run_network_dashboard as run,
    NetworkSimulationWorker as SimulationWorker,
    DEFAULT_CONFIG,
    DEFAULT_RL_MODEL,
)
