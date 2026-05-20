#!/usr/bin/env python3
# Plots: (1) x-y path with reference, (2) key states vs time, (3) controls, (4) arc length / lateral, (5) baseline.
# Requires: numpy, matplotlib
#
# Supported CSV logs (header row required for dubins format; legacy logs still work by column count):
#   dubins_path_tracking_example.cu         — straight line
#   dubins_circle_path_tracking_example.cu  — closed circle
#   Columns: u_accel, u_steer, arc_s, lat_err (+ *_centerline.csv)
#   Legacy RacerDubins examples (if present) — throttle/steer, optional path_u_*, arc_s
#
# Usage:
#   python3 examples/plot_racer_dubins_temporal_mppi.py dubins_path_tracking_log.csv
#   python3 examples/plot_racer_dubins_temporal_mppi.py log.csv -o viz.png
#   python3 examples/plot_racer_dubins_temporal_mppi.py log.csv --no-show

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


def perimeter_from_centerline(cpx: np.ndarray, cpy: np.ndarray) -> float:
    dx = np.diff(cpx)
    dy = np.diff(cpy)
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


def arc_s_from_xy_on_polyline(px: np.ndarray, py: np.ndarray, cpx: np.ndarray, cpy: np.ndarray) -> np.ndarray:
    """Closest-point arc length on a closed polyline."""
    s_vert = np.zeros(len(cpx))
    s_vert[1:] = np.cumsum(np.sqrt(np.diff(cpx) ** 2 + np.diff(cpy) ** 2))
    perimeter = s_vert[-1]
    out = np.zeros(len(px))
    for i, (x, y) in enumerate(zip(px, py)):
        best_d2 = np.inf
        best_s = 0.0
        for e in range(len(cpx) - 1):
            ax, ay = cpx[e], cpy[e]
            bx, by = cpx[e + 1], cpy[e + 1]
            ex, ey = bx - ax, by - ay
            elen2 = ex * ex + ey * ey
            t = 0.0 if elen2 < 1e-12 else np.clip(((x - ax) * ex + (y - ay) * ey) / elen2, 0.0, 1.0)
            qx, qy = ax + t * ex, ay + t * ey
            d2 = (x - qx) ** 2 + (y - qy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_s = s_vert[e] + t * (s_vert[e + 1] - s_vert[e])
        out[i] = best_s % perimeter if perimeter > 0 else 0.0
    return out


def display_available() -> bool:
    if sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def load_log(csv_path: Path) -> dict:
    """Load CSV with optional header; return column arrays and metadata."""
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    ncol = len(rows[0]) if rows else 0
    data = np.array([[float(x) for x in row] for row in rows], dtype=float)
    names = [h.strip() for h in header]
    name_to_idx = {n: i for i, n in enumerate(names)}

    def col(name: str, fallback: int) -> np.ndarray:
        if name in name_to_idx:
            return data[:, name_to_idx[name]]
        if 0 <= fallback < ncol:
            return data[:, fallback]
        raise KeyError(f"column {name!r} not in log (have {names})")

    out: dict = {"names": names, "ncol": ncol, "t": col("t", 0)}

    # Detect dubins bicycle log (accel/steer controls)
    is_dubins = "u_accel" in name_to_idx or (
        ncol == 18 and "u_throttle" not in name_to_idx and "ref_x" in name_to_idx
    )

    out["is_dubins"] = is_dubins
    out["px"] = col("pos_x", 1)
    out["py"] = col("pos_y", 2)
    out["yaw"] = col("yaw", 3)
    out["vx"] = col("vel_x", 4)
    out["steer"] = col("steer_angle", 5)

    if is_dubins:
        out["brake"] = col("brake_state", 6) if "brake_state" in name_to_idx else np.zeros_like(out["t"])
        out["u0"] = col("u_accel", 7)
        out["u1"] = col("u_steer", 8)
        out["nom_u0"] = col("nom_u_accel", 9) if "nom_u_accel" in name_to_idx else None
        out["nom_u1"] = col("nom_u_steer", 10) if "nom_u_steer" in name_to_idx else None
        out["path_u0"] = out["path_u1"] = None
        out["rpx"] = col("ref_x", 11)
        out["rpy"] = col("ref_y", 12)
        out["ryaw"] = col("ref_yaw", 13)
        out["ref_v"] = col("ref_v", 14)
        out["arc_s"] = col("arc_s", 15) if "arc_s" in name_to_idx else None
        out["lat_err"] = col("lat_err", 16) if "lat_err" in name_to_idx else None
        out["baseline"] = col("baseline", 17)
        out["u0_label"] = "u accel [m/s²]"
        out["u1_label"] = "u steer [rad]"
    elif ncol >= 19:
        out["brake"] = col("brake_state", 6)
        out["u0"] = col("u_throttle", 7)
        out["u1"] = col("u_steer", 8)
        out["nom_u0"] = col("nom_u_throttle", 9)
        out["nom_u1"] = col("nom_u_steer", 10)
        out["path_u0"] = col("path_u_throttle", 11)
        out["path_u1"] = col("path_u_steer", 12)
        out["rpx"] = col("ref_x", 13)
        out["rpy"] = col("ref_y", 14)
        out["ryaw"] = col("ref_yaw", 15)
        out["ref_v"] = col("ref_v", 16)
        out["arc_s"] = col("arc_s", 17)
        out["lat_err"] = None
        out["baseline"] = col("baseline", 18)
        out["u0_label"] = "u throttle (applied)"
        out["u1_label"] = "u steer (applied)"
    elif ncol >= 18:
        out["brake"] = col("brake_state", 6)
        out["u0"] = col("u_throttle", 7)
        out["u1"] = col("u_steer", 8)
        out["nom_u0"] = col("nom_u_throttle", 9)
        out["nom_u1"] = col("nom_u_steer", 10)
        out["path_u0"] = col("path_u_throttle", 11)
        out["path_u1"] = col("path_u_steer", 12)
        out["rpx"] = col("ref_x", 13)
        out["rpy"] = col("ref_y", 14)
        out["ryaw"] = col("ref_yaw", 15)
        out["ref_v"] = col("ref_v", 16)
        out["arc_s"] = None
        out["lat_err"] = None
        out["baseline"] = col("baseline", 17)
        out["u0_label"] = "u throttle (applied)"
        out["u1_label"] = "u steer (applied)"
    else:
        out["brake"] = col("brake_state", 6) if ncol > 6 else np.zeros_like(out["t"])
        out["u0"] = col("u_throttle", 7)
        out["u1"] = col("u_steer", 8)
        out["nom_u0"] = out["nom_u1"] = out["path_u0"] = out["path_u1"] = None
        out["rpx"] = col("ref_x", 9)
        out["rpy"] = col("ref_y", 10)
        out["ryaw"] = col("ref_yaw", 11)
        out["ref_v"] = col("ref_v", 12)
        out["arc_s"] = out["lat_err"] = None
        out["baseline"] = col("baseline", 13)
        out["u0_label"] = "u0"
        out["u1_label"] = "u1"

    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Visualize Dubins path-tracking or legacy RacerDubins temporal MPPI CSV logs.",
    )
    p.add_argument("csv", type=Path, help="Path to exported CSV from example binary")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save main figure PNG (default: <csv_stem>_viz.png). Use '-' for display only.",
    )
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving (default: 150)")
    p.add_argument("--show", action="store_true", help="Open interactive plot windows")
    p.add_argument("--no-show", action="store_true", help="Do not open windows; only write PNG files")
    args = p.parse_args()
    if not args.csv.is_file():
        print(f"error: not a file: {args.csv}", file=sys.stderr)
        return 1

    want_show = args.show or (not args.no_show and display_available())
    save_files = args.output != Path("-")
    if args.output is None:
        save_files = True

    try:
        import matplotlib

        if not want_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("error: need matplotlib. Install with: pip install matplotlib", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    log = load_log(args.csv)
    t = log["t"]
    px, py, yaw, vx = log["px"], log["py"], log["yaw"], log["vx"]
    steer, brake = log["steer"], log["brake"]
    u0, u1 = log["u0"], log["u1"]
    nom_u0, nom_u1 = log["nom_u0"], log["nom_u1"]
    path_u0, path_u1 = log["path_u0"], log["path_u1"]
    rpx, rpy, ryaw, ref_v = log["rpx"], log["rpy"], log["ryaw"], log["ref_v"]
    arc_s = log["arc_s"]
    lat_err = log["lat_err"]
    baseline = log["baseline"]
    is_dubins = log["is_dubins"]
    u0_label, u1_label = log["u0_label"], log["u1_label"]

    out_png = args.csv.with_name(args.csv.stem + "_viz.png")
    if args.output is not None:
        out_png = args.output

    centerline_path = args.csv.with_name(args.csv.stem + "_centerline.csv")
    have_cl = centerline_path.is_file()
    cpx = cpy = None
    perimeter = None
    if have_cl:
        cxy = np.loadtxt(centerline_path, delimiter=",", skiprows=1)
        if cxy.ndim == 1:
            cxy = cxy.reshape(1, -1)
        cpx, cpy = cxy[:, 0], cxy[:, 1]
        perimeter = perimeter_from_centerline(cpx, cpy)

    if arc_s is None and have_cl and cpx is not None:
        arc_s = arc_s_from_xy_on_polyline(px, py, cpx, cpy)
        print(f"note: no arc_s in log; estimated from {centerline_path.name}", file=sys.stderr)

    title_suffix = " (Dubins bicycle)" if is_dubins else ""
    n_progress_rows = 2 if lat_err is not None else 1
    fig = plt.figure(figsize=(12, 11 + 0.8 * (n_progress_rows - 1)), constrained_layout=True)
    gs = fig.add_gridspec(
        4 + n_progress_rows,
        2,
        height_ratios=[1.2, 1, 1, 1] + [0.75] * n_progress_rows,
        width_ratios=[1, 1],
    )

    ax_xy = fig.add_subplot(gs[0, :])
    sc = ax_xy.scatter(px, py, c=t, s=10, cmap="viridis", label="position (time → color)")
    fig.colorbar(sc, ax=ax_xy, label="time [s]", shrink=0.8)
    ax_xy.plot(px, py, "k-", linewidth=0.7, alpha=0.4, label="vehicle path")
    if have_cl and cpx is not None:
        ax_xy.plot(cpx, cpy, "r-", linewidth=1.4, label="ref centerline", alpha=0.95, zorder=2)
    ax_xy.plot(rpx, rpy, "r--", linewidth=1.1, label="ref (horizon t=0)", alpha=0.65)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title(f"Plan view{title_suffix}")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best")

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
    ax_st.legend(loc="best", fontsize=7)

    ax_br = fig.add_subplot(gs[3, 1])
    ax_br.plot(t, u0, label=u0_label, color="C4", linewidth=1.2)
    ax_br.plot(t, u1, label=u1_label, color="C5", alpha=0.85, linewidth=1.2)
    if nom_u0 is not None:
        nom0_lbl = "nom u accel (MPPI mean)" if is_dubins else "nom u throttle (MPPI mean)"
        nom1_lbl = "nom u steer (MPPI mean)"
        ax_br.plot(t, nom_u0, "--", color="C4", linewidth=1.0, alpha=0.75, label=nom0_lbl)
        ax_br.plot(t, nom_u1, "--", color="C5", linewidth=1.0, alpha=0.75, label=nom1_lbl)
    if path_u0 is not None:
        ax_br.plot(t, path_u0, ":", color="C4", linewidth=1.0, alpha=0.65, label="path u throttle")
        ax_br.plot(t, path_u1, ":", color="C5", linewidth=1.0, alpha=0.65, label="path u steer")
    if not is_dubins and np.any(brake != 0):
        ax_br.plot(t, brake, ":", color="0.35", linewidth=1.0, label="brake state")
    ax_br.set_ylabel("controls")
    ax_br.set_xlabel("t [s]")
    ax_br.legend(loc="best", fontsize=6)
    ax_br.grid(True, alpha=0.3)

    row_prog = 4
    if arc_s is not None:
        ax_s = fig.add_subplot(gs[row_prog, :])
        ax_s.plot(t, arc_s, color="C6", linewidth=1.2, label="arc s (projected)")
        v_ref = float(np.median(ref_v)) if len(ref_v) else 3.0
        if perimeter and perimeter > 0:
            s_unwrap = np.zeros(len(arc_s))
            s_unwrap[0] = arc_s[0]
            for i in range(1, len(arc_s)):
                ds = arc_s[i] - arc_s[i - 1]
                if ds < -0.5 * perimeter:
                    ds += perimeter
                elif ds > 0.5 * perimeter:
                    ds -= perimeter
                s_unwrap[i] = s_unwrap[i - 1] + ds
            ax_s.plot(
                t,
                s_unwrap,
                "-.",
                color="C6",
                alpha=0.65,
                linewidth=1.0,
                label="arc s (unwrapped)",
            )
            ax_s.plot(
                t,
                v_ref * t,
                "r--",
                alpha=0.55,
                linewidth=1.0,
                label=f"v_ref·t ({v_ref:.1f} m/s)",
            )
        else:
            ax_s.plot(t, v_ref * t, "r--", alpha=0.55, linewidth=1.0, label="v_ref·t")
        ax_s.set_ylabel("arc length [m]")
        ax_s.set_xlabel("t [s]")
        ax_s.set_title("Progress along path")
        ax_s.legend(loc="best", fontsize=7)
        ax_s.grid(True, alpha=0.3)
        row_prog += 1

    if lat_err is not None:
        ax_lat = fig.add_subplot(gs[row_prog, :])
        ax_lat.plot(t, lat_err, color="C7", linewidth=1.2, label="signed lateral error")
        ax_lat.axhline(0.0, color="0.4", linewidth=0.8, linestyle=":")
        ax_lat.set_ylabel("lateral [m] (+ = left of path)")
        ax_lat.set_xlabel("t [s]")
        ax_lat.set_title("Projection: signed lateral offset from path")
        ax_lat.legend(loc="best", fontsize=7)
        ax_lat.grid(True, alpha=0.3)

    fig2, axb = plt.subplots(1, 1, figsize=(10, 2.5), constrained_layout=True)
    axb.plot(t, baseline, color="0.2", linewidth=1.0)
    axb.set_ylabel("MPPI baseline cost")
    axb.set_xlabel("t [s]")
    axb.set_title("Nominal trajectory cost (after optimization)")
    axb.grid(True, alpha=0.3)

    if save_files:
        base = out_png if out_png.suffix.lower() == ".png" else out_png.with_suffix(".png")
        fig.savefig(base, dpi=args.dpi)
        baseline_png = base.with_name(base.stem + "_baseline.png")
        fig2.savefig(baseline_png, dpi=args.dpi)
        print(f"Wrote {base}")
        print(f"Wrote {baseline_png}")

    if want_show:
        print("Close all plot windows to exit.", file=sys.stderr)
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)
        if not save_files:
            print("error: no display and -o - given; use --show on a machine with a GUI.", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
