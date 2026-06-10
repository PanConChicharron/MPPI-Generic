#!/usr/bin/env python3
"""Visualize the top MPPI rollouts at a chosen simulation step (purple=low, green=high weight).

Usage:
  python3 scripts/mppi/plot_mppi_rollouts_at_step.py first_order_dubins_two_lane_double_park_log.csv --step 42
  python3 scripts/mppi/plot_mppi_rollouts_at_step.py dubins_stadium_mppi_rollout_analysis --step 0
  python3 scripts/mppi/plot_mppi_rollouts_at_step.py first_order_dubins_two_lane_double_park_log_rollouts/step_000042
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from mppi_plot_utils import (
    load_centerline,
    load_combined,
    load_costs,
    load_meta,
    load_rollout_segments,
    resolve_step_prefix,
    sort_rollouts_for_draw,
    weight_to_purple_green,
    weight_to_rollout_linewidths,
    weights_for_rollout_ids,
)


def display_available() -> bool:
    if sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Temporal log CSV, rollouts directory, or analysis prefix")
    p.add_argument("--step", type=int, default=0, help="Simulation step to visualize")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--zoom-pad", type=float, default=2.0)
    args = p.parse_args()

    if args.path.is_dir() and args.path.name.startswith("step_"):
        step_dir = args.path
    elif args.path.is_dir():
        step_dir = resolve_step_prefix(args.path, args.step)
    elif args.path.suffix == ".csv":
        step_dir = resolve_step_prefix(args.path, args.step)
    else:
        base = args.path if args.path.suffix == "" else args.path.with_suffix("")
        step_dir = Path(str(base)) if args.step == 0 else resolve_step_prefix(args.path, args.step)

    meta_path = step_dir / "meta.csv"
    costs_path = step_dir / "costs.csv"
    rollouts_path = step_dir / "rollouts_xy.csv"
    combined_path = step_dir / "combined.csv"

    for req in (meta_path, costs_path, rollouts_path, combined_path):
        if not req.is_file():
            print(f"error: missing {req}", file=sys.stderr)
            return 1

    want_show = args.show or (not args.no_show and display_available())
    try:
        import matplotlib

        if not want_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError as e:
        print("error: need matplotlib (pip install matplotlib)", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    meta = load_meta(meta_path)
    costs = load_costs(costs_path)
    segments, seg_ids, _ = load_rollout_segments(rollouts_path)
    combined = load_combined(combined_path)

    centerline_path = step_dir.parent.parent / (step_dir.parent.name.replace("_rollouts", "") + "_centerline.csv")
    if not centerline_path.is_file():
        centerline_path = step_dir.parent / "centerline.csv"
    if not centerline_path.is_file():
        centerline_path = step_dir.with_name(step_dir.name.replace("step_", "") + "_centerline.csv")
    cpx, cpy = load_centerline(centerline_path)
    if cpx is None:
        alt = Path(str(step_dir).rsplit("_rollouts", 1)[0] + "_centerline.csv")
        cpx, cpy = load_centerline(alt)

    seg_weights = weights_for_rollout_ids(costs, seg_ids)
    segments, seg_weights = sort_rollouts_for_draw(segments, seg_weights)
    colors = weight_to_purple_green(seg_weights)

    fig, ax = plt.subplots(figsize=(11, 9))
    if cpx is not None:
        ax.plot(cpx, cpy, "r-", linewidth=1.4, label="ref centerline", zorder=1)

    lc = LineCollection(segments, colors=colors, linewidths=weight_to_rollout_linewidths(seg_weights), zorder=2)
    ax.add_collection(lc)
    ax.plot(combined["x"], combined["y"], "k-", linewidth=2.0, label="MPPI combined", zorder=5)
    ax.scatter(combined["x"][0], combined["y"][0], c="lime", s=55, edgecolors="k", zorder=6, label="start")

    all_xy = np.vstack(segments + [np.column_stack([combined["x"], combined["y"]])])
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    pad = max(args.zoom_pad, 0.05 * max(x_max - x_min, y_max - y_min, 1e-3))
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    sim_step = int(meta.get("sim_step", args.step))
    sim_time = meta.get("sim_time", float("nan"))
    ax.set_title(
        f"Top {len(segments)} rollouts @ step {sim_step}  t={sim_time:.2f}s  "
        f"λ={meta.get('lambda', 0):.3g}  color: purple (low) → green (high weight)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "purple_green", [(0.50, 0.00, 0.55), (0.00, 0.78, 0.20)]))
    sm.set_array(seg_weights)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("normalized weight")

    out_png = args.output if args.output is not None else step_dir / "rollouts_viz.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Drew {len(segments)} rollouts from {step_dir}")
    print(f"Wrote {out_png}")
    if want_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
