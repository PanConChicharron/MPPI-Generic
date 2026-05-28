#!/usr/bin/env python3
"""Visualize one MPPI iteration: all rollouts (alpha = weight), costs, combined trajectory.

Usage:
  python3 examples/plot_mppi_rollout_analysis.py dubins_circle_mppi_rollout_analysis
  python3 examples/plot_mppi_rollout_analysis.py dubins_circle_mppi_rollout_analysis --no-show
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from mppi_rollout_plot import build_rollout_analysis_figure


def display_available() -> bool:
    if sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix", type=Path, help="Log file prefix (no extension)")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--alpha-min", type=float, default=0.02, help="Alpha for worst rollouts (by rank)")
    p.add_argument("--alpha-max", type=float, default=0.55, help="Alpha for best rollouts (by rank)")
    p.add_argument(
        "--rollout-cmap",
        type=str,
        default=None,
        help="Colormap override (default: viridis for weight encoding, coolwarm_r for cost)",
    )
    p.add_argument(
        "--zoom-pad",
        type=float,
        default=2.0,
        help="Padding [m] around rollout bounding box (use --zoom-pad=-1 for full centerline view)",
    )
    p.add_argument("--no-inset", action="store_true", help="Skip the small full-track overview inset")
    args = p.parse_args()

    want_show = args.show or (not args.no_show and display_available())
    try:
        import matplotlib

        if not want_show:
            matplotlib.use("Agg")
    except ImportError as e:
        print("error: need matplotlib (pip install matplotlib)", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    base = args.prefix if args.prefix.suffix == "" else args.prefix.with_suffix("")
    fig = build_rollout_analysis_figure(
        base,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        zoom_pad=args.zoom_pad,
        show_inset=not args.no_inset,
        rollout_cmap=args.rollout_cmap,
    )

    out_png = args.output if args.output is not None else Path(str(base) + "_viz.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_png}")
    if want_show:
        import matplotlib.pyplot as plt

        print("Close plot window to exit.")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
