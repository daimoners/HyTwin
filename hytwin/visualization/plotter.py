"""
Dashboard Plotter
==================
Matplotlib-based plots and real-time dashboard for simulation results.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from ..digital_twin.grid_twin import GridState

logger = logging.getLogger(__name__)


def plot_simulation_results(
    records: "pd.DataFrame | List[GridState]",
    title: str = "HyTwin — Simulation Results",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Comprehensive 6-panel dashboard of simulation output.

    Parameters
    ----------
    records : DataFrame or list of GridState
    title : str
    save_path : str, optional  — save figure to file
    show : bool                — call plt.show()
    """
    if not isinstance(records, pd.DataFrame):
        from dataclasses import asdict
        records = pd.DataFrame([asdict(gs) for gs in records])

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    gs_layout = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ts = records.get("timestamp", range(len(records)))

    # --- Panel 1: Power flows ---
    ax1 = fig.add_subplot(gs_layout[0, 0])
    ax1.fill_between(ts, records["wind_power_kw"], alpha=0.6, label="Wind [kW]", color="#3498db")
    ax1.fill_between(ts, records["pv_power_kw"], alpha=0.6, label="PV [kW]", color="#f39c12")
    ax1.plot(ts, records["load_kw"], "k--", lw=1.5, label="Load [kW]")
    ax1.set_ylabel("Power [kW]")
    ax1.set_title("Power Generation vs Load")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    # --- Panel 2: H2 system ---
    ax2 = fig.add_subplot(gs_layout[0, 1])
    ax2_twin = ax2.twinx()
    ax2.plot(ts, records["h2_storage_kg"], color="#27ae60", lw=2, label="H₂ storage [kg]")
    ax2_twin.plot(ts, records["h2_pressure_bar"], color="#8e44ad", lw=1.5, ls="--", label="Pressure [bar]")
    ax2.set_ylabel("H₂ Mass [kg]", color="#27ae60")
    ax2_twin.set_ylabel("Pressure [bar]", color="#8e44ad")
    ax2.set_title("Hydrogen Storage")
    lines1, l1 = ax2.get_legend_handles_labels()
    lines2, l2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, l1 + l2, fontsize=7)
    ax2.grid(alpha=0.3)

    # --- Panel 3: Electrolyzer + Fuel cell ---
    ax3 = fig.add_subplot(gs_layout[1, 0])
    ax3.plot(ts, records["electrolyzer_power_kw"], color="#e74c3c", lw=1.5, label="Electrolyzer [kW]")
    ax3.plot(ts, records["fuel_cell_power_kw"], color="#1abc9c", lw=1.5, label="Fuel Cell [kW]")
    ax3.set_ylabel("Power [kW]")
    ax3.set_title("H₂ Conversion Equipment")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # --- Panel 4: Grid exchange ---
    ax4 = fig.add_subplot(gs_layout[1, 1])
    colors_grid = ["#e74c3c" if v > 0 else "#2ecc71" for v in records["grid_exchange_kw"]]
    ax4.bar(range(len(records)), records["grid_exchange_kw"], color=colors_grid, alpha=0.7, width=1.0)
    ax4.axhline(0, color="black", lw=0.8)
    ax4.set_ylabel("Grid Exchange [kW]\n(+import / -export)")
    ax4.set_title("Grid Interaction")
    ax4.set_xlabel("Step")
    ax4.grid(alpha=0.3, axis="y")

    # --- Panel 5: KPIs ---
    ax5 = fig.add_subplot(gs_layout[2, 0])
    ax5.plot(ts, records["renewable_fraction"] * 100, color="#3498db", label="Renewable %")
    ax5.plot(ts, records["grid_self_sufficiency"] * 100, color="#2ecc71", label="Self-sufficiency %")
    if "overall_health" in records.columns:
        ax5.plot(ts, records["overall_health"] * 100, color="#e67e22", lw=1, ls="--", label="Health %")
    ax5.set_ylabel("[%]")
    ax5.set_ylim(-5, 110)
    ax5.set_title("System KPIs")
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.3)

    # --- Panel 6: H2 SOC ---
    ax6 = fig.add_subplot(gs_layout[2, 1])
    ax6.plot(ts, records["h2_soc"], color="#8e44ad", lw=2)
    ax6.axhline(0.5, color="gray", ls="--", lw=1, label="Target SOC=0.5")
    ax6.fill_between(ts, records["h2_soc"], 0.5, alpha=0.2,
                     where=(records["h2_soc"] > 0.5), color="#27ae60")
    ax6.fill_between(ts, records["h2_soc"], 0.5, alpha=0.2,
                     where=(records["h2_soc"] < 0.5), color="#e74c3c")
    ax6.set_ylabel("SOC [0..1]")
    ax6.set_ylim(-0.05, 1.05)
    ax6.set_title("Hydrogen Tank State of Charge")
    ax6.legend(fontsize=7)
    ax6.grid(alpha=0.3)

    for ax in [ax1, ax2, ax3, ax5, ax6]:
        ax.set_xlabel("Step")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)

    if show:
        plt.show()

    return fig


def plot_sensor_comparison(
    true_values: List[float],
    measured_values: List[float],
    label: str = "Signal",
    unit: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Two-panel comparison: overlay + error histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    steps = range(len(true_values))

    ax1.plot(steps, true_values, "b-", lw=1.5, label="True value")
    ax1.plot(steps, measured_values, "r-", lw=1, alpha=0.7, label="Sensor reading")
    ax1.set_xlabel("Step")
    ax1.set_ylabel(f"{label} [{unit}]")
    ax1.set_title(f"Sensor vs True — {label}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    errors = np.array(measured_values) - np.array(true_values)
    ax2.hist(errors, bins=40, color="#3498db", edgecolor="white", alpha=0.8)
    ax2.axvline(0, color="k", lw=1.5)
    ax2.set_xlabel(f"Error [{unit}]")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Measurement Error Distribution\n"
                  f"μ={errors.mean():.3f}, σ={errors.std():.3f}")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
