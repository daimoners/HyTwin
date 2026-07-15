"""
Time-Series Recorder
======================
Captures simulation output into in-memory and optionally on-disk records
for post-processing, plotting, and training data sets.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TimeSeriesRecorder:
    """
    Records ``GridState`` snapshots to a list of dicts (in memory)
    and optionally streams to a CSV file.

    Parameters
    ----------
    csv_path : str | Path, optional
        If provided, records are appended to this CSV file in real time.
    max_records : int
        Maximum number of records kept in memory.
    """

    def __init__(
        self,
        csv_path: Optional[str | Path] = None,
        max_records: int = 100_000,
    ) -> None:
        self._records: List[Dict[str, Any]] = []
        self._max = max_records
        self._csv_path = Path(csv_path) if csv_path else None
        self._csv_file = None
        self._csv_writer = None

        if self._csv_path:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")

    # ------------------------------------------------------------------

    def record(
        self,
        step: int,
        grid_state,
        weather: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store one step's data."""
        row: Dict[str, Any] = {"step": step}

        # GridState fields
        for f in fields(grid_state):
            v = getattr(grid_state, f.name)
            if isinstance(v, datetime):
                v = v.isoformat()
            elif isinstance(v, list):
                v = ";".join(str(x) for x in v)
            row[f.name] = v

        # Key weather variables
        if weather:
            for k in ("wind_speed_ms", "ghi_wm2", "temperature_c", "cloud_cover"):
                if k in weather:
                    row[f"wx_{k}"] = weather[k]

        self._records.append(row)
        if len(self._records) > self._max:
            self._records.pop(0)

        # Stream to CSV
        if self._csv_file:
            if self._csv_writer is None:
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=list(row.keys())
                )
                self._csv_writer.writeheader()
            self._csv_writer.writerow(row)
            self._csv_file.flush()

    def to_dataframe(self):
        """Return records as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._records)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2, default=str)

    def summary(self) -> Dict[str, float]:
        """Return aggregate KPIs over the recorded history."""
        if not self._records:
            return {}
        df = self.to_dataframe()
        return {
            "mean_renewable_fraction": float(df["renewable_fraction"].mean()),
            "mean_self_sufficiency": float(df["grid_self_sufficiency"].mean()),
            "mean_h2_soc": float(df["h2_soc"].mean()),
            "total_grid_import_kwh": float(
                df[df["grid_exchange_kw"] > 0]["grid_exchange_kw"].sum() / 60.0
            ),
            "total_wind_kwh": float(df["wind_power_kw"].sum() / 60.0),
            "total_pv_kwh": float(df["pv_power_kw"].sum() / 60.0),
            "mean_health": float(df["overall_health"].mean()),
        }

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    def __len__(self) -> int:
        return len(self._records)

    def __del__(self) -> None:
        self.close()
