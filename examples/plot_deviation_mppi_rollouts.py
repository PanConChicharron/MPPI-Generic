#!/usr/bin/env python3
"""Plot MPPI rollout analysis dumps captured at off-track timesteps.

The closed-loop dubins_stadium_path_tracking_example writes one dump per off-track
episode (same CSV layout as dubins_stadium_mppi_rollout_analysis_example) and lists
prefixes in <log_stem>_meta.csv under mppi_analysis_prefixes.

Usage:
  python3 examples/plot_deviation_mppi_rollouts.py dubins_stadium_path_tracking_log.csv
  python3 examples/plot_deviation_mppi_rollouts.py dubins_stadium_path_tracking_log.csv --no-show
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def load_meta(log_csv: Path) -> dict[str, str]:
    meta_path = log_csv.with_name(log_csv.stem + "_meta.csv")
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing {meta_path} (run path tracking example first)")
    out: dict[str, str] = {}
    with meta_path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["key"].strip()] = row["value"].strip()
    return out


def prefixes_from_meta(meta: dict[str, str]) -> list[str]:
    raw = meta.get("mppi_analysis_prefixes", "").strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(";") if p.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log_csv", type=Path, help="Temporal tracking log from dubins_stadium_path_tracking_example")
    p.add_argument("--no-show", action="store_true", help="Pass through to plot_mppi_rollout_analysis.py")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    if not args.log_csv.is_file():
        print(f"error: not a file: {args.log_csv}", file=sys.stderr)
        return 1

    plot_script = Path(__file__).resolve().parent / "plot_mppi_rollout_analysis.py"
    if not plot_script.is_file():
        print(f"error: missing {plot_script}", file=sys.stderr)
        return 1

    meta = load_meta(args.log_csv)
    prefixes = prefixes_from_meta(meta)
    if not prefixes:
        print(
            "error: no mppi_analysis_prefixes in meta (vehicle may have stayed on track)",
            file=sys.stderr,
        )
        return 1

    th = meta.get("track_departure_distance_threshold_m", "?")
    print(f"Plotting {len(prefixes)} MPPI dumps (off-track threshold {th} m)", file=sys.stderr)

    plot_args = [sys.executable, str(plot_script)]
    if args.no_show:
        plot_args.append("--no-show")
    if args.show:
        plot_args.append("--show")

    failed = 0
    for prefix in prefixes:
        print(f"\n--- {prefix} ---", file=sys.stderr)
        rc = subprocess.call([*plot_args, prefix])
        if rc != 0:
            failed += 1

    if failed:
        print(f"\n{failed}/{len(prefixes)} plots failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
