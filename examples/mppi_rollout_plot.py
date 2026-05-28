"""Shared matplotlib rendering for one MPPI iteration dump (CSV prefix)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_meta(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(newline="") as f:
        import csv

        for row in csv.DictReader(f):
            key = row["key"].strip()
            try:
                out[key] = float(row["value"])
            except ValueError:
                pass
    return out


def load_costs(path: Path) -> dict[str, np.ndarray]:
    import csv

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


def unwrap_dyaw(yaw0: float, yaw1: float) -> float:
    dy = float(yaw1) - float(yaw0)
    return float((dy + np.pi) % (2.0 * np.pi) - np.pi)


def finite_raw_costs(raw: np.ndarray) -> np.ndarray:
    r = np.asarray(raw, dtype=float)
    return r[np.isfinite(r) & (r < 1.0e20)]


def suggest_lambda(raw: np.ndarray, num_rollouts: int) -> float:
    """Heuristic λ so exp(-Δcost/λ) is not flat across rollouts (ESS well below N)."""
    fin = finite_raw_costs(raw)
    if fin.size == 0:
        return 30.0
    spread = float(np.max(fin) - np.min(fin))
    if spread < 1.0e-6:
        return 30.0
    n = max(int(num_rollouts), 2)
    return float(np.clip(spread / np.log(float(n)), 5.0, 500.0))


def load_rollout_segments(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    segments, seg_ids, _ = load_rollout_segments_with_yaw(path)
    return segments, seg_ids


def load_rollout_segments_with_yaw(path: Path) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    rollout_ids = data[:, 0].astype(int)
    steps = data[:, 1].astype(int)
    xy = data[:, 2:4]
    yaw = data[:, 4]

    unique_ids = np.unique(rollout_ids)
    segments: list[np.ndarray] = []
    seg_ids: list[int] = []
    seg_dyaw: list[float] = []
    for rid in unique_ids:
        mask = rollout_ids == rid
        order = np.argsort(steps[mask])
        pts = xy[mask][order]
        yaws = yaw[mask][order]
        if len(pts) >= 2:
            segments.append(pts)
            seg_ids.append(int(rid))
            seg_dyaw.append(unwrap_dyaw(yaws[0], yaws[-1]))
    return segments, np.asarray(seg_ids, dtype=int), np.asarray(seg_dyaw, dtype=float)


def load_reference_horizon(path: Path) -> dict[str, np.ndarray]:
    import csv

    cols: dict[str, list[float]] = {k: [] for k in ("k", "t", "arc_s", "x", "y", "yaw", "v")}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cols["k"].append(int(row["k"]))
            cols["t"].append(float(row["t"]))
            cols["arc_s"].append(float(row["arc_s"]))
            cols["x"].append(float(row["x"]))
            cols["y"].append(float(row["y"]))
            cols["yaw"].append(float(row["yaw"]))
            cols["v"].append(float(row["v"]))
    return {k: np.asarray(v) for k, v in cols.items()}


def load_log_reference_point(log_csv: Path, step_1based: int) -> dict[str, float] | None:
    """Instantaneous ref (t=0 of horizon) from closed-loop tracking log row."""
    import csv

    row_index = step_1based - 2
    if row_index < 0:
        return None
    with log_csv.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, row in enumerate(reader):
            if i != row_index or len(row) < 14:
                continue
            try:
                return {
                    "t": float(row[0]),
                    "x": float(row[11]),
                    "y": float(row[12]),
                    "yaw": float(row[13]),
                    "v": float(row[14]) if len(row) > 14 else float("nan"),
                }
            except ValueError:
                return None
    return None


def load_combined(path: Path) -> dict[str, np.ndarray]:
    import csv

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


def _rank01(values: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] by rank (ties broken by sort order)."""
    v = np.asarray(values, dtype=float)
    if v.size <= 1:
        return np.zeros_like(v) if v.size else v
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, v.size)
    return ranks


def pick_rollout_encoding(
    raw_costs: np.ndarray,
    weights: np.ndarray,
    *,
    weight_spread_tol: float = 0.015,
) -> tuple[np.ndarray, str, str, float, float, str]:
    """
    Choose visual score in [0, 1] (1 = best).

    When normalized weights are nearly flat (common with large λ), color/α use raw cost
  instead so rollouts remain distinguishable.
    """
    w = np.asarray(weights, dtype=float)
    raw = np.asarray(raw_costs, dtype=float)
    finite = np.isfinite(raw)
    if not np.any(finite):
        return np.ones_like(w), "weight", "MPPI weight", 1.0, 1.0, "viridis"

    w_spread = (float(np.max(w)) - float(np.min(w))) / (float(np.mean(w)) + 1e-30)
    if w_spread >= weight_spread_tol:
        w_max = float(np.max(w))
        score = np.clip(w / w_max, 0.0, 1.0) if w_max > 0 else np.ones_like(w)
        return score, "weight", "MPPI weight (color & α)", 0.0, w_max, "viridis"

    c = raw.copy()
    c[~finite] = float(np.nanmax(c[finite])) + 1.0
    c_min = float(np.min(c))
    c_max = float(np.max(c))
    if c_max - c_min < 1e-9:
        return np.ones_like(c), "cost", "trajectory cost (flat)", c_min, c_max, "coolwarm_r"
    score = 1.0 - (c - c_min) / (c_max - c_min)
    return score, "cost", "trajectory cost (color & α, cool=best)", c_min, c_max, "coolwarm_r"


def scores_to_rgba(
    scores: np.ndarray,
    *,
    encoding: str,
    alpha_min: float = 0.02,
    alpha_max: float = 0.55,
    cmap_name: str = "viridis",
    cbar_vmin: float = 0.0,
    cbar_vmax: float = 1.0,
) -> np.ndarray:
    """Per-rollout RGBA. Uses rank-based α so the full range is used even for tight cost clusters."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return np.zeros((0, 4), dtype=float)

    rank_t = _rank01(s)
    cmap = plt.get_cmap(cmap_name)
    if encoding == "cost":
        # Color by actual cost (low = good = cool end of coolwarm_r).
        norm = Normalize(vmin=cbar_vmin, vmax=cbar_vmax)
        raw_for_color = cbar_vmax - s * (cbar_vmax - cbar_vmin)
        rgba = np.asarray(cmap(norm(raw_for_color)), dtype=float)
    else:
        rgba = np.asarray(cmap(Normalize(vmin=0.0, vmax=1.0)(s)), dtype=float)

  # Rank-based α: best rollouts opaque, worst nearly invisible (avoids solid blob).
    rgba[:, 3] = alpha_min + (alpha_max - alpha_min) * rank_t
    return rgba


def scores_to_linewidths(scores: np.ndarray, lw_min: float = 0.15, lw_max: float = 1.0) -> np.ndarray:
    t = _rank01(np.asarray(scores, dtype=float))
    return lw_min + (lw_max - lw_min) * t


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    s = float(np.sum(w * w))
    return (1.0 / s) if s > 0 else 0.0


def prefix_paths(prefix: Path | str) -> dict[str, Path]:
    base = Path(prefix)
    if base.suffix:
        base = base.with_suffix("")
    s = str(base)
    return {
        "base": base,
        "meta": Path(s + "_meta.csv"),
        "costs": Path(s + "_costs.csv"),
        "rollouts": Path(s + "_rollouts_xy.csv"),
        "combined": Path(s + "_combined.csv"),
        "centerline": Path(s + "_centerline.csv"),
        "reference": Path(s + "_reference.csv"),
    }


def build_rollout_analysis_figure(
    prefix: Path | str,
    *,
    alpha_min: float = 0.02,
    alpha_max: float = 0.55,
    zoom_pad: float = 2.0,
    show_inset: bool = True,
    suptitle_extra: str = "",
    rollout_cmap: str | None = None,
    log_csv: Path | None = None,
    sim_step: int | None = None,
    highlight_ref_turn: bool = True,
):
    """Return a matplotlib Figure for one MPPI dump prefix (same layout as plot_mppi_rollout_analysis.py)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    paths = prefix_paths(prefix)
    for req in ("meta", "costs", "rollouts", "combined"):
        if not paths[req].is_file():
            raise FileNotFoundError(f"missing {paths[req]}")

    meta = load_meta(paths["meta"])
    costs = load_costs(paths["costs"])
    segments, seg_ids, seg_dyaw = load_rollout_segments_with_yaw(paths["rollouts"])
    combined = load_combined(paths["combined"])

    raw = costs["raw_cost"]
    raw_finite = finite_raw_costs(raw)
    weights = costs["normalized"]
    baseline = meta.get("baseline", float(np.min(raw)))
    normalizer = meta.get("normalizer", float(np.sum(costs["unnormalized"])))

    n_rollouts = int(meta.get("num_rollouts", len(weights)))
    w_by_rollout = np.zeros(n_rollouts, dtype=float)
    c_by_rollout = np.full(n_rollouts, np.inf, dtype=float)
    for i, r in enumerate(costs["index"]):
        if 0 <= r < n_rollouts:
            w_by_rollout[r] = weights[i]
            c_by_rollout[r] = raw[i]
    seg_weights = w_by_rollout[seg_ids]
    seg_costs = c_by_rollout[seg_ids]
    ess = effective_sample_size(weights)

    score, enc_mode, cbar_label, cbar_vmin, cbar_vmax, cmap_auto = pick_rollout_encoding(
        seg_costs, seg_weights
    )
    cmap_use = rollout_cmap if rollout_cmap is not None else cmap_auto
    seg_rgba = scores_to_rgba(
        score,
        encoding=enc_mode,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        cmap_name=cmap_use,
        cbar_vmin=cbar_vmin,
        cbar_vmax=cbar_vmax,
    )
    seg_lw = scores_to_linewidths(score)

    cpx = cpy = None
    if paths["centerline"].is_file():
        cxy = np.loadtxt(paths["centerline"], delimiter=",", skiprows=1)
        if cxy.ndim == 1:
            cxy = cxy.reshape(1, -1)
        cpx, cpy = cxy[:, 0], cxy[:, 1]

    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.55, 0.95, 0.85],
        width_ratios=[1.2, 1, 1],
        left=0.06,
        right=0.97,
        top=0.94,
        bottom=0.06,
        hspace=0.42,
        wspace=0.28,
    )

    ax_xy = fig.add_subplot(gs[0, :])
    if cpx is not None:
        ax_xy.plot(cpx, cpy, "r-", linewidth=1.5, label="track centerline", zorder=1)

    lc = LineCollection(segments, colors=seg_rgba, linewidths=seg_lw, zorder=2)
    ax_xy.add_collection(lc)

    ref_h = None
    ref_dyaw = 0.0
    if paths["reference"].is_file():
        ref_h = load_reference_horizon(paths["reference"])
        ref_dyaw = unwrap_dyaw(ref_h["yaw"][0], ref_h["yaw"][-1])
        ax_xy.plot(
            ref_h["x"],
            ref_h["y"],
            color="lime",
            linewidth=2.4,
            linestyle="-",
            label="cost reference (horizon)",
            zorder=4,
            alpha=0.95,
        )
    elif log_csv is not None and sim_step is not None and log_csv.is_file():
        ref_pt = load_log_reference_point(log_csv, sim_step)
        if ref_pt is not None:
            ax_xy.scatter(
                ref_pt["x"],
                ref_pt["y"],
                c="lime",
                s=80,
                edgecolors="k",
                linewidths=0.8,
                label="ref (log t=0)",
                zorder=4,
            )
    if enc_mode == "cost":
        sm = ScalarMappable(cmap=plt.get_cmap(cmap_use), norm=Normalize(vmin=cbar_vmin, vmax=cbar_vmax))
    else:
        sm = ScalarMappable(cmap=plt.get_cmap(cmap_use), norm=Normalize(vmin=0.0, vmax=cbar_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_xy, fraction=0.025, pad=0.01, aspect=30)
    cbar.set_label(cbar_label)

    seg_index_lookup: dict[int, int] = {int(rid): i for i, rid in enumerate(seg_ids)}

    if highlight_ref_turn and ref_h is not None and abs(ref_dyaw) > 0.05:
        turn_eps = 0.12
        ref_sign = float(np.sign(ref_dyaw))
        turn_mask = np.sign(seg_dyaw) == ref_sign
        turn_mask &= np.abs(seg_dyaw) > turn_eps
        n_turn = int(np.sum(turn_mask))
        if n_turn > 0:
            turn_segs = [segments[i] for i in range(len(segments)) if turn_mask[i]]
            lc_turn = LineCollection(
                turn_segs,
                colors=(0.95, 0.45, 0.05, 0.35),
                linewidths=0.7,
                zorder=3,
            )
            ax_xy.add_collection(lc_turn)
            turn_costs = seg_costs[turn_mask]
            turn_ids = seg_ids[turn_mask]
            best_turn_local = int(np.argmin(turn_costs))
            best_turn_id = int(turn_ids[best_turn_local])
            si = int(np.where(seg_ids == best_turn_id)[0][0])
            ax_xy.plot(
                segments[si][:, 0],
                segments[si][:, 1],
                color="darkorange",
                linewidth=2.2,
                label=f"best ref-turn rollout #{best_turn_id} (n={n_turn})",
                zorder=4,
            )

    ax_xy.plot(combined["x"], combined["y"], "k-", linewidth=2.2, label="MPPI combined (optimal u)", zorder=5)
    ax_xy.scatter(combined["x"][0], combined["y"][0], c="lime", s=60, edgecolors="k", zorder=6, label="start")

    fin_idx = np.where(np.isfinite(raw) & (raw < 1.0e20))[0]
    if fin_idx.size > 0:
        best_fin = int(fin_idx[np.argmin(raw[fin_idx])])
        best_c_i = int(costs["index"][best_fin])
    else:
        best_c_i = int(costs["index"][int(np.argmin(raw))])

    if 0 in seg_index_lookup:
        si = seg_index_lookup[0]
        ax_xy.plot(
            segments[si][:, 0],
            segments[si][:, 1],
            color="orange",
            linewidth=1.6,
            linestyle="--",
            label="rollout #0 (noise-free / pre-iter nominal)",
            zorder=3,
        )

    if best_c_i in seg_index_lookup:
        si = seg_index_lookup[best_c_i]
        ax_xy.plot(
            segments[si][:, 0],
            segments[si][:, 1],
            color="magenta",
            linewidth=1.8,
            label=f"lowest-cost rollout #{best_c_i}",
            zorder=4,
        )

    best_w_i = int(costs["index"][int(np.argmax(weights))])
    if enc_mode == "weight" and best_w_i != best_c_i and best_w_i in seg_index_lookup:
        si = seg_index_lookup[best_w_i]
        ax_xy.plot(
            segments[si][:, 0],
            segments[si][:, 1],
            color="deepskyblue",
            linewidth=1.4,
            linestyle=":",
            label=f"highest-weight rollout #{best_w_i}",
            zorder=4,
        )

    all_xy = np.vstack(segments + [np.column_stack([combined["x"], combined["y"]])])
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    if zoom_pad >= 0:
        pad = max(zoom_pad, 0.05 * max(x_max - x_min, y_max - y_min, 1e-3))
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
    sim_t = meta.get("sim_t", float("nan"))
    sim_step = meta.get("sim_step", float("nan"))
    time_lbl = ""
    if not np.isnan(sim_t):
        time_lbl = f"  log step {int(sim_step)} t={sim_t:.2f}s"
    enc_note = "cost" if enc_mode == "cost" else "weight"
    ess_pct = 100.0 * ess / max(n_total, 1)
    cost_spread = float(np.max(raw_finite) - np.min(raw_finite)) if raw_finite.size else 0.0
    lam_suggest = suggest_lambda(raw, n_total)
    turn_note = ""
    if ref_h is not None and abs(ref_dyaw) > 0.05:
        n_match = int(np.sum((np.sign(seg_dyaw) == np.sign(ref_dyaw)) & (np.abs(seg_dyaw) > 0.12)))
        turn_note = f"  ref-turn Δyaw={ref_dyaw:.2f} rad ({n_match}/{len(seg_dyaw)} rollouts)"
    ax_xy.set_title(
        f"All MPPI rollouts (n={n_drawn}/{n_total}, color/α by {enc_note})  "
        f"λ={meta.get('lambda', 0):.3g}  ESS≈{ess:.0f} ({ess_pct:.0f}%)  Δcost≈{cost_spread:.3g}  "
        f"try λ≈{lam_suggest:.0f}{turn_note}{time_lbl}"
    )
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best", fontsize=8)

    if show_inset and cpx is not None and zoom_pad >= 0:
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
    w_spread_pct = 100.0 * (float(np.max(weights)) - float(np.min(weights))) / (float(np.mean(weights)) + 1e-30)
    ax_w.set_title(
        f"Weight distribution  (Σw={weights.sum():.2f}, spread={w_spread_pct:.2f}%, ESS≈{ess:.0f})"
    )
    if enc_mode == "cost":
        ax_w.text(
            0.03,
            0.97,
            "weights ~ uniform → rollouts colored by cost",
            transform=ax_w.transAxes,
            fontsize=7,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
    ax_w.grid(True, alpha=0.3)

    ax_sc = fig.add_subplot(gs[1, 2])
    sc = ax_sc.scatter(raw, weights, c=weights, s=8, cmap="viridis", alpha=0.65)
    ax_sc.set_xlabel("raw cost")
    ax_sc.set_ylabel("normalized weight")
    ax_sc.set_title("Cost vs weight")
    ax_sc.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax_sc, label="weight")

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
    ax_us.plot(t, combined["steer"], color="0.4", linewidth=1.2, linestyle="--", label="steer state")
    ax_us.axhline(0.0, color="0.7", linewidth=0.6, linestyle="--")
    ax_us.set_xlabel("t [s]")
    ax_us.set_ylabel("u_steer [rad]")
    ax_us.grid(True, alpha=0.3)
    ax_us.legend(loc="best", fontsize=8)

    n_at_min = int(np.sum(np.isclose(raw, raw.min(), rtol=0, atol=1e-3)))
    extra = f"  |  {suptitle_extra}" if suptitle_extra else ""
    fig.suptitle(
        f"MPPI iteration — {n_drawn} rollouts  |  best cost {raw.min():.6g} (idx {best_c_i})  |  "
        f"max weight {weights.max():.3g} (idx {best_w_i}){extra}",
        fontsize=10,
    )
    return fig
