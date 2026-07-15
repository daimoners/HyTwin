#!/usr/bin/env python3
"""
Final Validation: Verify weather reproducibility in actual run_one() calls
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import the actual run_one function
from experiments.control_comparison.run_control_scenarios import run_one

def test_scenario_reproducibility():
    """
    Verify that running the same scenario twice with same seed produces
    IDENTICAL environmental outputs (wind, pv, load, prices).
    """
    print("\n" + "=" * 80)
    print("FINAL VALIDATION: Scenario Reproducibility with run_one()")
    print("=" * 80)
    
    config_path = str(ROOT / "config" / "advanced_stress.yaml")
    
    # Run scenario 1: mode="none" with seed 999
    print("\nRun 1: scenario=none, seed=999, steps=50...")
    df1 = run_one(
        mode="none",
        config_path=config_path,
        steps=50,
        dt=600,
        speed_factor=0.0,
        seed=999,
        rl_model_path=None,
    )
    print(f"  Returned {len(df1)} rows")
    
    # Run scenario 2: mode="none" with SAME seed
    print("\nRun 2: scenario=none, seed=999, steps=50 (identical to Run 1)...")
    df2 = run_one(
        mode="none",
        config_path=config_path,
        steps=50,
        dt=600,
        speed_factor=0.0,
        seed=999,
        rl_model_path=None,
    )
    print(f"  Returned {len(df2)} rows")
    
    # Run scenario 3: mode="rule" with SAME seed
    print("\nRun 3: scenario=rule, seed=999, steps=50 (same seed, different controller)...")
    df3 = run_one(
        mode="rule",
        config_path=config_path,
        steps=50,
        dt=600,
        speed_factor=0.0,
        seed=999,
        rl_model_path=None,
    )
    print(f"  Returned {len(df3)} rows")
    
    # Compare environmental variables
    print("\n" + "-" * 80)
    print("COMPARISON: Environmental variables should be IDENTICAL")
    print("-" * 80)
    
    env_vars = ["wind_power_kw", "pv_power_kw", "load_kw", "energy_price_eur_kwh"]
    all_pass = True
    
    for var in env_vars:
        if var not in df1.columns:
            print(f"  ✗ {var}: NOT FOUND in dataframe")
            continue
        
        v1 = df1[var].values
        v2 = df2[var].values
        v3 = df3[var].values
        
        # Run 1 vs Run 2 (identical config+seed)
        match_12 = np.allclose(v1, v2, atol=1e-10, rtol=1e-10)
        max_diff_12 = np.max(np.abs(v1 - v2)) if match_12 else np.max(np.abs(v1 - v2))
        
        # Run 1 vs Run 3 (same seed, different controller) - should still match for env vars
        match_13 = np.allclose(v1, v3, atol=1e-10, rtol=1e-10)
        max_diff_13 = np.max(np.abs(v1 - v3)) if match_13 else np.max(np.abs(v1 - v3))
        
        status_12 = "✓" if match_12 else "✗"
        status_13 = "✓" if match_13 else "✗"
        
        print(f"\n  {var}:")
        print(f"    Run 1 vs Run 2 (identical): {status_12} max_diff={max_diff_12:.2e}")
        print(f"    Run 1 vs Run 3 (rule_vs_none): {status_13} max_diff={max_diff_13:.2e}")
        
        all_pass = all_pass and match_12 and match_13
    
    # Summary
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ PASS: Environmental reproducibility is GUARANTEED")
        print("  Fair comparison of controllers is now possible!")
        return True
    else:
        print("✗ FAIL: Some environmental variables still diverge")
        print("  Controller comparison results may be invalid")
        return False

if __name__ == "__main__":
    success = test_scenario_reproducibility()
    sys.exit(0 if success else 1)
