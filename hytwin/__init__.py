"""
HyTwin 2.0 — H2-based Energy Grid Digital Twin Framework
=========================================================
A modular, scalable framework for simulating, monitoring, and optimising
hydrogen-based energy grids through digital twin technology and reinforcement
learning.

Architecture layers:
  core/          — Event bus, state manager, simulation clock, component registry
  models/        — Physics-based models for each grid component
  sensors/       — Virtual sensor layer (noise, drift, delay, faults)
  weather/       — Stochastic weather and solar irradiance generation
  digital_twin/  — Digital twin nodes, state estimation, grid twin
  rl/            — Gymnasium environment, reward functions, RL trainer
  simulation/    — Simulation engine, scenario runner
  data/          — Time-series store, export utilities
  visualization/ — Dashboard and plotting utilities
"""

__version__ = "2.0.0"
__author__ = "HyTwin Development Team"
