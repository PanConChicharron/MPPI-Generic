#!/usr/bin/env python3
"""Visualize one MPPI iteration: all rollouts (alpha = weight), costs, combined trajectory.

Usage:
  python3 examples/plot_mppi_rollout_analysis.py dubins_circle_mppi_rollout_analysis
  python3 examples/plot_mppi_rollout_analysis.py dubins_circle_mppi_rollout_analysis --no-show
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


def load_meta(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["key"]] = float(row["value"])
    return out


def load_costs(path: Path) -> dict[str, np.ndarray]:
    idx, raw, unnorm, norm = [], [], [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            idx.append(int(row["rollout_index"]))
            raw.append(float(row["raw_cost"]))
            unnorm.append(float(row["unnormalized_importance"]))
            norm.append(float(row["normalized_weight"]))
    return {
        "index": np.asarray(idx, dtype=int),
        "raw_cost": np.asarray(raw),
        "unnormalized": np.asarray(unnorm),
        "normalized": np.asarray(norm),
    }


def load_rollout_segments(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Build LineCollection segments and rollout indices (one segment per rollout)."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    rollout_ids = data[:, 0].astype(int)
    steps = data[:, 1].astype(int)
    xy = data[:, 2:4]

    unique_ids = np.unique(rollout_ids)
    segments: list[np.ndarray] = []
    seg_ids: list[int] = []
    for rid in unique_ids:
        mask = rollout_ids == rid
        order = np.argsort(steps[mask])
        pts = xy[mask][order]
        if len(pts) >= 2:
            segments.append(pts)
            seg_ids.append(int(rid))
    return segments, np.asarray(seg_ids, dtype=int)




def load_combined(path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {
        k: [] for k in ("step", "t", "x", "y", "yaw", "vel", "steer", "u_accel", "u_steer")
    }
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cols["step"].append(int(row["step"]))
            cols["t"].append(float(row["t"]))
            cols["x"].append(float(row["x"]))
            cols["y"].append(float(row["y"]))
            cols["yaw"].append(float(row["yaw"]))
            cols["vel"].append(float(row["vel_x"]))
            cols["steer"].append(float(row["steer"]))
            cols["u_accel"].append(float(row["u_accel"]))
            cols["u_steer"].append(float(row["u_steer"]))
    return {k: np.asarray(v) for k, v in cols.items()}


def weight_to_alpha(weights: np.ndarray, alpha_min: float = 0.03, alpha_max: float = 0.55) -> np.ndarray:
    """Map normalized MPPI weights to line alpha (sqrt stretch helps low-weight rollouts)."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return w
    w_max = float(np.max(w))
    if w_max <= 0:
        return np.full_like(w, alpha_min)
    t = np.sqrt(np.clip(w / w_max, 0.0, 1.0))
    return alpha_min + (alpha_max - alpha_min) * t


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
    p.add_argument("--alpha-min", type=float, default=0.03, help="Alpha for lowest-weight rollout")
    p.add_argument("--alpha-max", type=float, default=0.55, help="Alpha for highest-weight rollout")
    p.add_argument("--zoom-pad", type=float, default=2.0,
                   help="Padding [m] around rollout bounding box (use --zoom-pad=-1 for full centerline view)")
    p.add_argument("--no-inset", action="store_true", help="Skip the small full-track overview inset")
    args = p.parse_args()

    base = args.prefix if args.prefix.suffix == "" else args.prefix.with_suffix("")
    meta_path = Path(str(base) + "_meta.csv")
    costs_path = Path(str(base) + "_costs.csv")
    rollouts_path = Path(str(base) + "_rollouts_xy.csv")
    combined_path = Path(str(base) + "_combined.csv")
    centerline_path = Path(str(base) + "_centerline.csv")

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
    segments, seg_ids = load_rollout_segments(rollouts_path)
    combined = load_combined(combined_path)

    raw = costs["raw_cost"]
    weights = costs["normalized"]
    baseline = meta.get("baseline", float(np.min(raw)))
    normalizer = meta.get("normalizer", float(np.sum(costs["unnormalized"])))

    # weights[rollout_index] — index column is 0..N-1
    w_by_rollout = np.zeros(int(meta.get("num_rollouts", len(weights))), dtype=float)
    for i, r in enumerate(costs["index"]):
        if 0 <= r < w_by_rollout.size:
            w_by_rollout[r] = weights[i]
    seg_weights = w_by_rollout[seg_ids]
    seg_alphas = weight_to_alpha(seg_weights, args.alpha_min, args.alpha_max)

    cpx = cpy = None
    if centerline_path.is_file():
        cxy = np.loadtxt(centerline_path, delimiter=",", skiprows=1)
        if cxy.ndim == 1:
            cxy = cxy.reshape(1, -1)
        cpx, cpy = cxy[:, 0], cxy[:, 1]

    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.55, 0.95, 0.85], width_ratios=[1.2, 1, 1],
                          left=0.06, right=0.97, top=0.94, bottom=0.06, hspace=0.42, wspace=0.28)

    ax_xy = fig.add_subplot(gs[0, :])
    if cpx is not None:
        ax_xy.plot(cpx, cpy, "r-", linewidth=1.5, label="ref centerline", zorder=1)

    lc = LineCollection(
        segments,
        colors=(0.15, 0.35, 0.65, 1.0),
        linewidths=0.45,
        zorder=2,
    )
    lc.set_alpha(seg_alphas)
    ax_xy.add_collection(lc)

    ax_xy.plot(combined["x"], combined["y"], "k-", linewidth=2.2, label="MPPI combined (optimal u)", zorder=5)
    ax_xy.scatter(combined["x"][0], combined["y"][0], c="lime", s=60, edgecolors="k", zorder=6, label="start")

    seg_index_lookup: dict[int, int] = {int(rid): i for i, rid in enumerate(seg_ids)}

    # Rollout 0 is the noise-free / pre-iteration nominal rollout in this codebase's GaussianDistribution.
    if 0 in seg_index_lookup:
        si = seg_index_lookup[0]
        ax_xy.plot(
            segments[si][:, 0], segments[si][:, 1],
            color="orange", linewidth=1.6, linestyle="--",
            label="rollout #0 (noise-free / pre-iter nominal)", zorder=3,
        )

    best_w_i = int(costs["index"][int(np.argmax(weights))])
    if best_w_i in seg_index_lookup:
        si = seg_index_lookup[best_w_i]
        ax_xy.plot(
            segments[si][:, 0], segments[si][:, 1],
            color="magenta", linewidth=1.8,
            label=f"highest-weight rollout #{best_w_i}", zorder=4,
        )

    best_c_i = int(costs["index"][int(np.argmin(raw))])

    # Zoom to rollout bounding box (with the start point + combined trajectory so they stay in frame).
    all_xy = np.vstack(segments + [np.column_stack([combined["x"], combined["y"]])])
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    if args.zoom_pad >= 0:
        pad = max(args.zoom_pad, 0.05 * max(x_max - x_min, y_max - y_min, 1e-3))
        ax_xy.set_xlim(x_min - pad, x_max + pad)
        ax_xy.set_ylim(y_min - pad, y_max + pad)
    else:
        ax_xy.autoscale()
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    n_drawn = len(segments)
    n_total = int(meta.get("num_rollouts", len(raw)))
    span_m = float(np.hypot(x_max - x_min, y_max - y_min))
    ax_xy.set_title(
        f"All MPPI rollouts (n={n_drawn}/{n_total}, line α ∝ √weight)  "
        f"v₀={meta.get('init_vel_x', 0):.2f} m/s  λ={meta.get('lambda', 0):.3g}  "
        f"rollout cluster ≈ {span_m:.2f} m"
    )
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best", fontsize=8)

    # Small overview inset showing the full track + the zoomed rectangle. Uses the modern
    # Axes.inset_axes API (axes_grid1.inset_locator interferes with the interactive toolbar).
    if not args.no_inset and cpx is not None and args.zoom_pad >= 0:
        ax_inset = ax_xy.inset_axes([0.02, 0.02, 0.18, 0.30])
        ax_inset.plot(cpx, cpy, "r-", linewidth=0.9)
        ax_inset.plot(combined["x"], combined["y"], "k-", linewidth=1.2)
        rect_x = [x_min - pad, x_max + pad, x_max + pad, x_min - pad, x_min - pad]
        rect_y = [y_min - pad, y_min - pad, y_max + pad, y_max + pad, y_min - pad]
        ax_inset.plot(rect_x, rect_y, color="orange", linewidth=1.0)
        ax_inset.set_aspect("equal")
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_title("overview", fontsize=7, pad=2)

    ax_cost = fig.add_subplot(gs[1, 0])
    ax_cost.hist(raw, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax_cost.axvline(baseline, color="crimson", linewidth=2, label=f"baseline = {baseline:.2g}")
    ax_cost.set_xlabel("raw trajectory cost")
    ax_cost.set_ylabel("count")
    ax_cost.set_title("Cost distribution")
    ax_cost.legend(fontsize=8)
    ax_cost.grid(True, alpha=0.3)

    ax_w = fig.add_subplot(gs[1, 1])
    ax_w.hist(weights, bins=60, color="seagreen", edgecolor="white", alpha=0.85)
    ax_w.set_xlabel("normalized importance weight")
    ax_w.set_ylabel("count")
    ax_w.set_title(f"Weight distribution  (Σw = {weights.sum():.2f}, Z = {normalizer:.2g})")
    ax_w.grid(True, alpha=0.3)

    ax_sc = fig.add_subplot(gs[1, 2])
    sc = ax_sc.scatter(raw, weights, c=weights, s=8, cmap="viridis", alpha=0.65)
    ax_sc.set_xlabel("raw cost")
    ax_sc.set_ylabel("normalized weight")
    ax_sc.set_title("Cost vs weight")
    ax_sc.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax_sc, label="weight")

    # Optimal control commands over the MPPI horizon. combined.csv carries u_accel/u_steer (the
    # MPPI-optimal command) and the realized steer state, which lags the command through the
    # bicycle's first-order steer dynamics (visible at the corner-entry curvature step).
    sub = gs[2, :].subgridspec(2, 1, hspace=0.15)
    ax_ua = fig.add_subplot(sub[0, 0])
    ax_us = fig.add_subplot(sub[1, 0], sharex=ax_ua)
    t = combined["t"]

    ax_ua.plot(t, combined["u_accel"], color="tab:blue", linewidth=1.7, label="u_accel (cmd)")
    ax_ua.axhline(0.0, color="0.7", linewidth=0.6, linestyle="--")
    ax_ua.set_ylabel("u_accel [m/s²]")
    ax_ua.set_title("Optimal control over MPPI horizon")
    ax_ua.grid(True, alpha=0.3)
    ax_ua.legend(loc="best", fontsize=8)
    plt.setp(ax_ua.get_xticklabels(), visible=False)

    ax_us.plot(t, combined["u_steer"], color="tab:red", linewidth=1.7, label="u_steer (cmd)")
    if "steer" in combined:
        ax_us.plot(t, combined["steer"], color="0.4", linewidth=1.2, linestyle="--", label="steer state")
    ax_us.axhline(0.0, color="0.7", linewidth=0.6, linestyle="--")
    ax_us.set_xlabel("t [s]")
    ax_us.set_ylabel("u_steer [rad]")
    ax_us.grid(True, alpha=0.3)
    ax_us.legend(loc="best", fontsize=8)

    n_at_min = int(np.sum(np.isclose(raw, raw.min(), rtol=0, atol=1e-3)))
    fig.suptitle(
        f"MPPI iteration — {n_drawn} rollouts drawn  |  best cost {raw.min():.6g} (idx {best_c_i}, "
        f"{n_at_min} rollouts within 1e-3)  |  max weight {weights.max():.3g} (idx {best_w_i})",
        fontsize=10,
    )

    out_png = args.output if args.output is not None else Path(str(base) + "_viz.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Drew {n_drawn} rollouts (alpha from weight, range [{args.alpha_min}, {args.alpha_max}])")
    print(f"Wrote {out_png}")
    if want_show:
        print("Close plot window to exit.")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
