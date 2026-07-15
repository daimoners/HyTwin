#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate RL H2 behavior from control comparison CSV outputs."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to run horizon dir, e.g. output/.../5d",
    )
    parser.add_argument(
        "--min-el-on-ratio",
        type=float,
        default=0.05,
        help="Minimum fraction of steps with electrolyzer_power_kw > 1 kW",
    )
    parser.add_argument(
        "--min-fc-on-ratio",
        type=float,
        default=0.05,
        help="Minimum fraction of steps with fuel_cell_power_kw > 1 kW",
    )
    parser.add_argument(
        "--min-soc-delta",
        type=float,
        default=0.02,
        help="Minimum SOC dynamic range (max-min) to avoid flat/no-H2-use behavior",
    )
    parser.add_argument(
        "--max-grid-cost-vs-rule-ratio",
        type=float,
        default=1.30,
        help="Maximum allowed ratio RL_cost / Rule_cost when rule CSV exists",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def ratio_on(series: pd.Series, threshold: float = 1.0) -> float:
    return float((series > threshold).mean())


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    data_dir = run_dir / "data"

    rl_path = data_dir / "timeseries_rl.csv"
    rule_path = data_dir / "timeseries_rule.csv"

    rl_df = load_csv(rl_path)

    el_on_ratio = ratio_on(rl_df["electrolyzer_power_kw"], threshold=1.0)
    fc_on_ratio = ratio_on(rl_df["fuel_cell_power_kw"], threshold=1.0)
    soc_delta = float(rl_df["h2_soc"].max() - rl_df["h2_soc"].min())
    rl_cost = float(rl_df["step_cost_eur"].sum())

    print("=" * 72)
    print("RL H2 BEHAVIOR VALIDATION")
    print("=" * 72)
    print(f"run_dir               : {run_dir}")
    print(f"steps                 : {len(rl_df)}")
    print(f"el_on_ratio (>1 kW)   : {el_on_ratio:.3f}")
    print(f"fc_on_ratio (>1 kW)   : {fc_on_ratio:.3f}")
    print(f"h2_soc_delta          : {soc_delta:.3f}")
    print(f"rl_cost_total_eur     : {rl_cost:.2f}")

    checks: list[tuple[bool, str]] = []
    checks.append(
        (el_on_ratio >= float(args.min_el_on_ratio),
         f"EL usage ratio >= {args.min_el_on_ratio:.3f}")
    )
    checks.append(
        (fc_on_ratio >= float(args.min_fc_on_ratio),
         f"FC usage ratio >= {args.min_fc_on_ratio:.3f}")
    )
    checks.append(
        (soc_delta >= float(args.min_soc_delta),
         f"H2 SOC delta >= {args.min_soc_delta:.3f}")
    )

    if rule_path.exists():
        rule_df = load_csv(rule_path)
        rule_cost = float(rule_df["step_cost_eur"].sum())
        if rule_cost > 1e-9:
            cost_ratio = rl_cost / rule_cost
            print(f"rule_cost_total_eur   : {rule_cost:.2f}")
            print(f"rl_vs_rule_cost_ratio : {cost_ratio:.3f}")
            checks.append(
                (cost_ratio <= float(args.max_grid_cost_vs_rule_ratio),
                 f"RL/Rule cost ratio <= {args.max_grid_cost_vs_rule_ratio:.3f}")
            )

    print("-" * 72)
    all_ok = True
    for ok, label in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}")
        all_ok = all_ok and ok

    print("-" * 72)
    if all_ok:
        print("RESULT: PASS — RL shows non-trivial H2 behavior under current thresholds")
        return 0

    print("RESULT: FAIL — RL policy appears degenerate or under-trained for H2 usage")
    return 1


if __name__ == "__main__":
    sys.exit(main())
