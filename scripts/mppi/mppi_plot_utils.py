"""Shared CSV loaders for MPPI rollout / temporal plotting tools."""
from __future__ import annotations

import csv
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


def load_rollout_segments(path: Path) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return LineCollection segments, rollout ids, and normalized weight per segment."""
    trajectories = load_rollout_trajectories(path)
    segments: list[np.ndarray] = []
    seg_ids: list[int] = []
    rollout_ids_flat: list[int] = []
    for rid, xy in trajectories.items():
        if xy.shape[0] >= 2:
            segments.append(xy)
            seg_ids.append(int(rid))
        rollout_ids_flat.extend([int(rid)] * xy.shape[0])
    return segments, np.asarray(seg_ids, dtype=int), np.asarray(rollout_ids_flat, dtype=int)


def load_rollout_trajectories(path: Path) -> dict[int, np.ndarray]:
    """Load per-rollout (x, y) polylines keyed by rollout_index."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.size == 0:
        return {}
    if data.ndim == 1:
        data = data.reshape(1, -1)
    rollout_ids = data[:, 0].astype(int)
    steps = data[:, 1].astype(int)
    xy = data[:, 2:4]

    out: dict[int, np.ndarray] = {}
    for rid in np.unique(rollout_ids):
        mask = rollout_ids == rid
        order = np.argsort(steps[mask])
        pts = xy[mask][order]
        if len(pts) >= 1:
            out[int(rid)] = pts
    return out


def compute_weighted_xy_trajectory(
    trajectories: dict[int, np.ndarray],
    weights_by_rollout: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """MPPI-style weighted average of rollout (x, y) paths for lambda retuning visualization."""
    if not trajectories:
        return np.asarray([]), np.asarray([])

    max_len = max(traj.shape[0] for traj in trajectories.values())
    xs = np.full((len(trajectories), max_len), np.nan, dtype=float)
    ys = np.full((len(trajectories), max_len), np.nan, dtype=float)
    ws = np.zeros(len(trajectories), dtype=float)
    for j, (rid, traj) in enumerate(trajectories.items()):
        n = traj.shape[0]
        xs[j, :n] = traj[:, 0]
        ys[j, :n] = traj[:, 1]
        ws[j] = max(weights_by_rollout.get(int(rid), 0.0), 0.0)

    w_sum = float(np.sum(ws))
    if w_sum <= 0.0:
        candidates = [rid for rid in trajectories if weights_by_rollout.get(int(rid), 0.0) > 0.0]
        if not candidates:
            candidates = list(trajectories.keys())
        best_rid = max(candidates, key=lambda rid: weights_by_rollout.get(int(rid), 0.0))
        traj = trajectories[best_rid]
        return traj[:, 0].copy(), traj[:, 1].copy()

    x_out = np.zeros(max_len, dtype=float)
    y_out = np.zeros(max_len, dtype=float)
    for t in range(max_len):
        w_t = ws.copy()
        w_t[np.isnan(xs[:, t])] = 0.0
        denom = float(np.sum(w_t))
        if denom <= 0.0:
            break
        x_out[t] = float(np.nansum(w_t * xs[:, t]) / denom)
        y_out[t] = float(np.nansum(w_t * ys[:, t]) / denom)
    valid = ~(np.isnan(x_out) | np.isnan(y_out))
    return x_out[valid], y_out[valid]


def load_rollout_controls(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load per-rollout control sequences keyed by rollout_index."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.size == 0:
        return {}
    if data.ndim == 1:
        data = data.reshape(1, -1)
    rollout_ids = data[:, 0].astype(int)
    steps = data[:, 1].astype(int)
    u_accel = data[:, 2]
    u_steer = data[:, 3]

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for rid in np.unique(rollout_ids):
        mask = rollout_ids == rid
        order = np.argsort(steps[mask])
        out[int(rid)] = (u_accel[mask][order], u_steer[mask][order])
    return out


def compute_weighted_control_trajectory(
    controls: dict[int, tuple[np.ndarray, np.ndarray]],
    weights_by_rollout: dict[int, float],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted average of sampled control sequences (Σ wᵢ·uᵢ), aligned with combined.csv time axis."""
    if not controls or dt <= 0.0:
        return np.asarray([]), np.asarray([]), np.asarray([])

    max_len = max(len(u_a) for u_a, _ in controls.values())
    ua = np.full((len(controls), max_len), np.nan, dtype=float)
    us = np.full((len(controls), max_len), np.nan, dtype=float)
    ws = np.zeros(len(controls), dtype=float)
    for j, (rid, (u_a, u_s)) in enumerate(controls.items()):
        n = len(u_a)
        ua[j, :n] = u_a
        us[j, :n] = u_s
        ws[j] = max(weights_by_rollout.get(int(rid), 0.0), 0.0)

    w_sum = float(np.sum(ws))
    if w_sum <= 0.0:
        candidates = [rid for rid in controls if weights_by_rollout.get(int(rid), 0.0) > 0.0]
        if not candidates:
            candidates = list(controls.keys())
        best_rid = max(candidates, key=lambda rid: weights_by_rollout.get(int(rid), 0.0))
        u_a, u_s = controls[best_rid]
        t = (np.arange(len(u_a), dtype=float) + 1.0) * dt
        return t, u_a.copy(), u_s.copy()

    u_a_out = np.zeros(max_len, dtype=float)
    u_s_out = np.zeros(max_len, dtype=float)
    for t in range(max_len):
        w_t = ws.copy()
        w_t[np.isnan(ua[:, t])] = 0.0
        denom = float(np.sum(w_t))
        if denom <= 0.0:
            break
        u_a_out[t] = float(np.nansum(w_t * ua[:, t]) / denom)
        u_s_out[t] = float(np.nansum(w_t * us[:, t]) / denom)
    valid = ~(np.isnan(u_a_out) | np.isnan(u_s_out))
    t_out = (np.arange(max_len, dtype=float)[valid] + 1.0) * dt
    return t_out, u_a_out[valid], u_s_out[valid]


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


def load_centerline(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not path.is_file():
        return None, None
    cxy = np.loadtxt(path, delimiter=",", skiprows=1)
    if cxy.ndim == 1:
        cxy = cxy.reshape(1, -1)
    return cxy[:, 0], cxy[:, 1]


def load_steps_index(path: Path) -> list[tuple[int, float, str]]:
    rows: list[tuple[int, float, str]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["step"]), float(row["sim_time"]), row["prefix"]))
    return rows


def resolve_step_prefix(log_or_rollout_dir: Path, step: int) -> Path:
    """Resolve a step directory from a temporal log path or rollouts root."""
    if log_or_rollout_dir.is_dir():
        rollout_root = log_or_rollout_dir
    else:
        stem = log_or_rollout_dir.with_suffix("") if log_or_rollout_dir.suffix == ".csv" else log_or_rollout_dir
        rollout_root = Path(str(stem) + "_rollouts")

    index_path = rollout_root / "steps_index.csv"
    if index_path.is_file():
        for s, _t, prefix in load_steps_index(index_path):
            if s == step:
                return Path(prefix)
    return rollout_root / f"step_{step:06d}"


def weights_for_rollout_ids(costs: dict[str, np.ndarray], seg_ids: np.ndarray) -> np.ndarray:
    lookup = {int(i): float(w) for i, w in zip(costs["index"], costs["normalized"])}
    return np.asarray([lookup.get(int(rid), 0.0) for rid in seg_ids], dtype=float)


def weight_to_purple_green(
    weights: np.ndarray,
    *,
    alpha_min: float = 0.07,
    alpha_max: float = 0.98,
    alpha_gamma: float = 2.0,
) -> np.ndarray:
    """Map normalized weights to RGBA colors: purple (low) -> green (high).

    Low-weight rollouts are drawn more transparent so high-weight trajectories stay
    visible when many paths overlap. Use sort_rollouts_for_draw() so heavy lines
    render on top.
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return w.reshape(0, 4)
    w_min = float(np.min(w))
    w_max = float(np.max(w))
    span = max(w_max - w_min, 1e-12)
    t = np.clip((w - w_min) / span, 0.0, 1.0)
    purple = np.array([0.50, 0.00, 0.55, 1.0])
    green = np.array([0.00, 0.78, 0.20, 1.0])
    colors = purple[None, :] + t[:, None] * (green - purple)[None, :]
    alpha_t = t ** alpha_gamma
    colors[:, 3] = alpha_min + alpha_t * (alpha_max - alpha_min)
    return colors


def weight_to_rollout_linewidths(
    weights: np.ndarray,
    *,
    width_min: float = 0.35,
    width_max: float = 1.5,
) -> np.ndarray:
    """Per-segment line widths scaled by relative weight."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return w
    w_min = float(np.min(w))
    w_max = float(np.max(w))
    span = max(w_max - w_min, 1e-12)
    t = np.clip((w - w_min) / span, 0.0, 1.0)
    return width_min + t * (width_max - width_min)


def sort_rollouts_for_draw(
    segments: list[np.ndarray],
    weights: np.ndarray,
    ids: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray] | tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Sort rollouts ascending by weight so high-weight paths draw on top."""
    order = np.argsort(weights)
    sorted_segments = [segments[int(i)] for i in order]
    sorted_weights = weights[order]
    if ids is None:
        return sorted_segments, sorted_weights
    sorted_ids = ids[order]
    return sorted_segments, sorted_weights, sorted_ids


def recompute_weights(raw_cost: np.ndarray, baseline: float, lambda_val: float) -> tuple[np.ndarray, np.ndarray, float]:
    w = np.exp(-(raw_cost - baseline) / max(lambda_val, 1e-12))
    w = np.where(np.isfinite(w), w, 0.0)
    normalizer = float(np.sum(w))
    if normalizer <= 0.0:
        norm = np.zeros_like(w)
    else:
        norm = w / normalizer
    return w, norm, normalizer
