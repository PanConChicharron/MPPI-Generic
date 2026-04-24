#!/usr/bin/env python3
# Plots: (1) x-y path with reference, (2) key states vs time, (3) steer + controls + brake, (4) baseline cost.
# Requires: numpy, matplotlib
#
# Examples that produce this CSV:
#   racer_dubins_temporal_mppi_example_straight_line, racer_dubins_temporal_mppi_example_racetrack
#
# Usage:
#   python3 plot_racer_dubins_temporal_mppi.py racer_dubins_mppi_log.csv -o mppi_viz.png
#   python3 plot_racer_dubins_temporal_mppi.py racer_dubins_racetrack_mppi_log.csv -o racetrack_viz.png
#   If <csv_stem>_centerline.csv exists (from racetrack example), a solid red closed centerline is drawn;
#   the dashed ref from the log is s(t) in time and may be a sub-lap segment.

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(
        description="Visualize RacerDubins temporal MPPI log (CSV from straight_line or racetrack example binaries).",
    )
    p.add_argument("csv", type=Path, help="Path to exported CSV (see example binary)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <csv_stem>_viz.png). Use '-' to only display.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default: 150)",
    )
    args = p.parse_args()
    if not args.csv.is_file():
        print(f"error: not a file: {args.csv}", file=sys.stderr)
        return 1

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("error: need matplotlib. Install with: pip install matplotlib", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    # t,px,py,yaw,vx,steer,brake,u0,u1,ref_x,ref_y,ref_yaw,ref_v,baseline
    data = np.loadtxt(
        args.csv,
        delimiter=",",
        skiprows=1,
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t = data[:, 0]
    px, py, yaw, vx = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    steer, brake = data[:, 5], data[:, 6]
    u0, u1 = data[:, 7], data[:, 8]
    rpx, rpy, ryaw, ref_v = data[:, 9], data[:, 10], data[:, 11], data[:, 12]
    baseline = data[:, 13]

    out_png = args.output
    if out_png is None:
        out_png = args.csv.with_name(args.csv.stem + "_viz.png")

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1, 1], width_ratios=[1, 1])

    # --- Plan view: actual path + optional full closed centerline (racetrack example)
    # Red dashed (ref_x,ref_y) vs time is only a partial segment if sim_time * v_ref < lap length.
    centerline_path = args.csv.with_name(args.csv.stem + "_centerline.csv")
    have_cl = centerline_path.is_file()

    ax_xy = fig.add_subplot(gs[0, :])
    sc = ax_xy.scatter(
        px,
        py,
        c=t,
        s=10,
        cmap="viridis",
        label="position (time → color)",
    )
    fig.colorbar(sc, ax=ax_xy, label="time [s]", shrink=0.8)
    ax_xy.plot(px, py, "k-", linewidth=0.7, alpha=0.4, label="path (line)")
    if have_cl:
        cxy = np.loadtxt(centerline_path, delimiter=",", skiprows=1)
        if cxy.ndim == 1:
            cxy = cxy.reshape(1, -1)
        cpx, cpy = cxy[:, 0], cxy[:, 1]
        ax_xy.plot(cpx, cpy, "r-", linewidth=1.4, label="ref centerline (1 lap, closed)", alpha=0.95, zorder=2)
    ax_xy.plot(
        rpx,
        rpy,
        "r--",
        linewidth=1.1,
        label="ref s(t) from log" + (" (sub-lap if sim short)" if have_cl else ""),
        alpha=0.65,
    )
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    if have_cl:
        ax_xy.set_title("Path: vehicle vs closed centerline; dashed = ref in log vs time (may be partial arc)")
    else:
        ax_xy.set_title("Path in plane (reference vs closed-loop)")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best")

    # --- States
    for ax, yv, name, c in [
        (fig.add_subplot(gs[1, 0]), px, "pos x [m]", "C0"),
        (fig.add_subplot(gs[1, 1]), py, "pos y [m]", "C0"),
        (fig.add_subplot(gs[2, 0]), yaw, "yaw [rad]", "C1"),
        (fig.add_subplot(gs[2, 1]), vx, "vel x [m/s]", "C2"),
    ]:
        ax.plot(t, yv, color=c, linewidth=1.2)
        if name == "pos x [m]":
            ax.plot(t, rpx, "r--", alpha=0.6, label="ref")
        elif name == "pos y [m]":
            ax.plot(t, rpy, "r--", alpha=0.6, label="ref")
        elif name == "yaw [rad]":
            ax.plot(t, ryaw, "r--", alpha=0.6, label="ref")
        elif name == "vel x [m/s]":
            ax.plot(t, ref_v, "r--", alpha=0.6, label="ref")
        ax.set_ylabel(name)
        ax.set_xlabel("t [s]")
        ax.grid(True, alpha=0.3)
        if name.startswith("pos") or name in ("yaw [rad]", "vel x [m/s]"):
            ax.legend(loc="best", fontsize=7)

    ax_st = fig.add_subplot(gs[3, 0])
    ax_st.plot(t, steer, color="C3", label="steer angle [state]")
    ax_st.set_ylabel("steer angle (state)")
    ax_st.set_xlabel("t [s]")
    ax_st.grid(True, alpha=0.3)
    ax_br = fig.add_subplot(gs[3, 1])
    ax_br.plot(t, u0, label="u throttle/brake", color="C4")
    ax_br.plot(t, u1, label="u steer cmd", color="C5", alpha=0.85)
    ax_br.plot(t, brake, ":", color="0.35", linewidth=1.0, label="brake state")
    ax_br.set_ylabel("control / brake")
    ax_br.set_xlabel("t [s]")
    ax_br.legend(loc="best", fontsize=7)
    ax_br.grid(True, alpha=0.3)

    # Second figure: baseline (often easier to read separate)
    fig2, axb = plt.subplots(1, 1, figsize=(10, 2.5), constrained_layout=True)
    axb.plot(t, baseline, color="0.2", linewidth=1.0)
    axb.set_ylabel("MPPI baseline cost")
    axb.set_xlabel("t [s]")
    axb.set_title("Nominal trajectory cost (after optimization)")
    axb.grid(True, alpha=0.3)

    if str(out_png) == "-":
        print("Use a GUI backend to show interactively; saving skipped (-o -).", file=sys.stderr)
        return 0

    base = out_png
    if base.suffix.lower() not in (".png",):
        base = base.with_suffix(".png")
    fig.savefig(base, dpi=args.dpi)
    fig2.savefig(base.with_name(base.stem + "_baseline.png"), dpi=args.dpi)
    plt.close(fig)
    plt.close(fig2)
    print(f"Wrote {base}")
    print(f"Wrote {base.with_name(base.stem + '_baseline.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
