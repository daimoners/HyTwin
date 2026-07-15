#!/usr/bin/env python3
"""
Debug: Track RNG consumption in actual run_one scenarios
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.control_comparison.run_control_scenarios import run_one

def detailed_weather_comparison():
    """
    Run none vs rule with same seed and trace where weather diverges.
    """
    print("\n" + "=" * 80)
    print("DETAILED WEATHER COMPARISON: none vs rule (same seed)")
    print("=" * 80)
    
    config_path = str(ROOT / "config" / "advanced_stress.yaml")
    seed = 887
    steps = 20  # Short run for debugging
    
    print(f"\nRunning scenario=none with seed={seed}...")
    df_none = run_one("none", config_path, steps, 600, 0.0, seed, None)
    
    print(f"Running scenario=rule with seed={seed}...")
    df_rule = run_one("rule", config_path, steps, 600, 0.0, seed, None)
    
    print("\n" + "-" * 80)
    print("Step-by-step comparison")
    print("-" * 80)
    print(f"{'Step':>5} {'Timestamp':<20} {'Wind_none':>12} {'Wind_rule':>12} {'Diff':>12}")
    print("-" * 80)
    
    all_wind_match = True
    first_divergence_step = None
    
    for i in range(min(len(df_none), len(df_rule))):
        w_none = df_none.iloc[i]["wind_power_kw"]
        w_rule = df_rule.iloc[i]["wind_power_kw"]
        ts = df_none.iloc[i]["timestamp"]
        diff = abs(w_none - w_rule)
        
        if diff > 1e-10:
            all_wind_match = False
            if first_divergence_step is None:
                first_divergence_step = i
            marker = " ← DIVERGENCE"
        else:
            marker = ""
        
        print(f"{i:5d} {str(ts):<20} {w_none:12.4f} {w_rule:12.4f} {diff:12.2e}{marker}")
    
    print("\n" + "=" * 80)
    if all_wind_match:
        print("✓ PASS: Wind power is IDENTICAL across scenarios")
    else:
        print(f"✗ FAIL: Wind power DIVERGES at step {first_divergence_step}")
        print(f"\nFirst 5 wind values (none): {df_none['wind_power_kw'].head().values}")
        print(f"First 5 wind values (rule): {df_rule['wind_power_kw'].head().values}")
    
    print("=" * 80 + "\n")
    return all_wind_match

if __name__ == "__main__":
    success = detailed_weather_comparison()
    sys.exit(0 if success else 1)
