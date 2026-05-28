#!/usr/bin/env python3
"""Live MPPI cost-weight tuner on a stadium tracking log.

Left panel: sliders for PathTrackingCost weights + λ, sim-step picker (off-track episodes from meta).
Right panel: same rollout-analysis view as plot_mppi_rollout_analysis.py.

Requires:
  - Built binary: build/examples/dubins_stadium_mppi_tune_from_log
  - matplotlib, numpy

Usage:
  python3 examples/mppi_tune_ui.py dubins_stadium_path_tracking_log.csv
  python3 examples/mppi_tune_ui.py --binary build/examples/dubins_stadium_mppi_tune_from_log log.csv
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

# Allow running as script from repo root or examples/
_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from mppi_rollout_plot import (
    build_rollout_analysis_figure,
    effective_sample_size,
    finite_raw_costs,
    load_costs,
    pick_rollout_encoding,
    prefix_paths,
    suggest_lambda,
)

DEFAULT_WEIGHTS = {
    "w_pos": 20.0,
    "w_heading_so2": 3.0,
    "w_vel": 5.0,
    "w_lat_accel": 1.0,
    "w_lat_jerk": 0.05,
    "w_steer_dot": 0.0,
    "w_accel": 0.05,
    "w_steer": 0.05,
}
WEIGHT_ORDER = list(DEFAULT_WEIGHTS.keys())
DEFAULT_LAMBDA = 30.0
CLOSED_LOOP_LAMBDA = 3000.0


def repo_root() -> Path:
    return _EXAMPLES.parent


def default_tune_binary() -> Path:
    return repo_root() / "build" / "examples" / "dubins_stadium_mppi_tune_from_log"


def load_tracking_meta(log_csv: Path) -> dict[str, str]:
    meta_path = log_csv.with_name(log_csv.stem + "_meta.csv")
    if not meta_path.is_file():
        return {}
    out: dict[str, str] = {}
    with meta_path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["key"].strip()] = row["value"].strip()
    return out


def deviation_steps_from_meta(meta: dict[str, str]) -> list[int]:
    raw = meta.get("track_departure_steps", "").strip()
    if not raw:
        return []
    steps: list[int] = []
    for part in raw.split(";"):
        part = part.strip()
        if part:
            steps.append(int(part))
    return steps


def tune_output_prefix(log_csv: Path) -> Path:
    return log_csv.with_name(log_csv.stem + "_tune_live")


class MppiTuneApp:
    def __init__(self, log_csv: Path, binary: Path, seed: int) -> None:
        self.log_csv = log_csv.resolve()
        self.binary = binary.resolve()
        self.seed = seed
        self.meta = load_tracking_meta(self.log_csv)
        self.deviation_steps = deviation_steps_from_meta(self.meta)
        self.output_prefix = tune_output_prefix(self.log_csv)
        self._run_after_id: str | None = None
        self._busy = False

        self.root = tk.Tk()
        self.root.title(f"MPPI weight tuner — {self.log_csv.name}")
        self.root.geometry("1500x900")

        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, padding=8)
        paned.add(left, weight=0)

        right = ttk.Frame(paned, padding=4)
        paned.add(right, weight=1)

        self._build_controls(left)
        self._build_plot_panel(right)

        if not self.binary.is_file():
            self.status.set(f"Build tuner: cmake --build build --target dubins_stadium_mppi_tune_from_log")
        else:
            self.status.set("Adjust weights and click Run MPPI (or enable auto-run).")

    def _build_controls(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Log", font=("", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.log_var = tk.StringVar(value=str(self.log_csv))
        ttk.Entry(parent, textvariable=self.log_var, width=36).grid(row=1, column=0, columnspan=2, sticky="ew")

        ttk.Label(parent, text="Sim step (1-based)", font=("", 10, "bold")).grid(row=2, column=0, sticky="w", pady=(12, 0))
        step_default = str(self.deviation_steps[0]) if self.deviation_steps else "226"
        self.step_var = tk.StringVar(value=step_default)
        self.step_combo = ttk.Combobox(
            parent,
            textvariable=self.step_var,
            values=[str(s) for s in self.deviation_steps],
            width=12,
        )
        self.step_combo.grid(row=3, column=0, sticky="w")
        if self.deviation_steps:
            ttk.Label(parent, text="(off-track episodes from meta)", font=("", 8)).grid(row=4, column=0, sticky="w")
        else:
            ttk.Label(parent, text="(no meta; enter step manually)", font=("", 8)).grid(row=4, column=0, sticky="w")

        ttk.Label(parent, text="Cost weights", font=("", 10, "bold")).grid(row=5, column=0, sticky="w", pady=(12, 0))
        self.weight_vars: dict[str, tk.DoubleVar] = {}
        row = 6
        ranges = {
            "w_pos": (0.0, 100.0),
            "w_heading_so2": (0.0, 50.0),
            "w_vel": (0.0, 50.0),
            "w_lat_accel": (0.0, 50.0),
            "w_lat_jerk": (0.0, 100.0),
            "w_steer_dot": (0.0, 200.0),
            "w_accel": (0.0, 10.0),
            "w_steer": (0.0, 10.0),
        }
        for key in WEIGHT_ORDER:
            ttk.Label(parent, text=key).grid(row=row, column=0, sticky="w")
            var = tk.DoubleVar(value=DEFAULT_WEIGHTS[key])
            self.weight_vars[key] = var
            lo, hi = ranges[key]
            scale = ttk.Scale(parent, variable=var, from_=lo, to=hi, orient=tk.HORIZONTAL, length=220)
            scale.grid(row=row, column=1, sticky="ew", padx=(4, 0))
            ent = ttk.Entry(parent, textvariable=var, width=8)
            ent.grid(row=row, column=2, padx=(4, 0))
            scale.configure(command=lambda _v, k=key: self._on_weight_slider(k))
            row += 1

        ttk.Label(parent, text="lambda", font=("", 10, "bold")).grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1
        self.lambda_var = tk.DoubleVar(value=DEFAULT_LAMBDA)
        lam_scale = ttk.Scale(
            parent, variable=self.lambda_var, from_=1.0, to=10000.0, orient=tk.HORIZONTAL, length=220
        )
        lam_scale.grid(row=row, column=1, sticky="ew")
        lam_scale.configure(command=lambda _v: self._schedule_run() if self.auto_run.get() else None)
        ttk.Entry(parent, textvariable=self.lambda_var, width=8).grid(row=row, column=2)
        row += 1

        lam_preset = ttk.Frame(parent)
        lam_preset.grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(lam_preset, text="λ presets:", font=("", 8)).pack(side=tk.LEFT)
        for label, val in (("tune 30", 30.0), ("med 100", 100.0), ("closed-loop 3000", CLOSED_LOOP_LAMBDA)):
            ttk.Button(
                lam_preset,
                text=label,
                width=14,
                command=lambda v=val: self._set_lambda(v),
            ).pack(side=tk.LEFT, padx=2)
        row += 1

        self.lambda_hint = tk.StringVar(value="Lower λ to sharpen weights (closed-loop uses 3000).")
        ttk.Label(parent, textvariable=self.lambda_hint, wraplength=320, font=("", 8)).grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1

        self.highlight_turn = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent,
            text="Highlight rollouts turning with reference",
            variable=self.highlight_turn,
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.auto_run = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Auto-run on change (slow)", variable=self.auto_run).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(12, 0)
        )
        row += 1

        btn_row = ttk.Frame(parent)
        btn_row.grid(row=row, column=0, columnspan=3, pady=(8, 0), sticky="ew")
        ttk.Button(btn_row, text="Run MPPI", command=self.run_mppi).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Reset weights", command=self.reset_weights).pack(side=tk.LEFT, padx=6)
        row += 1

        self.status = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.status, wraplength=320).grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 0))
        parent.columnconfigure(1, weight=1)

    def _build_plot_panel(self, parent: ttk.Frame) -> None:
        import matplotlib

        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.plot_frame = parent
        self.fig = None
        self.canvas = None
        self.toolbar = None
        placeholder = ttk.Label(parent, text="Run MPPI to render rollout analysis here.")
        placeholder.pack(expand=True)

    def _on_weight_slider(self, _key: str) -> None:
        if self.auto_run.get():
            self._schedule_run()

    def _schedule_run(self) -> None:
        if self._run_after_id is not None:
            self.root.after_cancel(self._run_after_id)
        self._run_after_id = self.root.after(600, self._auto_run_dispatch)

    def _auto_run_dispatch(self) -> None:
        self._run_after_id = None
        self.run_mppi()

    def _set_lambda(self, value: float) -> None:
        self.lambda_var.set(value)
        if self.auto_run.get():
            self._schedule_run()

    def reset_weights(self) -> None:
        for k, v in DEFAULT_WEIGHTS.items():
            self.weight_vars[k].set(v)
        self.lambda_var.set(DEFAULT_LAMBDA)

    def weights_csv(self) -> str:
        return ",".join(f"{self.weight_vars[k].get():.6g}" for k in WEIGHT_ORDER)

    def run_mppi(self) -> None:
        if self._busy:
            return
        if not self.binary.is_file():
            messagebox.showerror("Missing binary", f"Not found:\n{self.binary}\n\nBuild dubins_stadium_mppi_tune_from_log.")
            return

        log_path = Path(self.log_var.get()).expanduser()
        if not log_path.is_file():
            messagebox.showerror("Log missing", str(log_path))
            return

        try:
            step = int(self.step_var.get())
        except ValueError:
            messagebox.showerror("Invalid step", "Step must be an integer.")
            return

        self._busy = True
        self.status.set("Running MPPI on GPU…")
        self.root.update_idletasks()

        cmd = [
            str(self.binary),
            "--log",
            str(log_path),
            "--step",
            str(step),
            "--prefix",
            str(self.output_prefix),
            "--weights",
            self.weights_csv(),
            "--lambda",
            f"{self.lambda_var.get():.6g}",
            "--seed",
            str(self.seed),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root()))
        except OSError as e:
            self._busy = False
            messagebox.showerror("Launch failed", str(e))
            self.status.set(str(e))
            return

        self._busy = False
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
            messagebox.showerror("MPPI failed", err)
            self.status.set(err[:200])
            return

        self.refresh_plot()
        try:
            costs = load_costs(prefix_paths(self.output_prefix)["costs"])
            raw = costs["raw_cost"]
            n = len(costs["normalized"])
            _, enc_mode, _, _, _, _ = pick_rollout_encoding(raw, costs["normalized"])
            ess = effective_sample_size(costs["normalized"])
            lam_s = suggest_lambda(raw, n)
            fin = finite_raw_costs(raw)
            spread = float(fin.max() - fin.min()) if fin.size else 0.0
            msg = f"OK — ESS≈{ess:.0f}/{n}  Δcost≈{spread:.3g}  try λ≈{lam_s:.0f}"
            if ess > 0.5 * n:
                msg += " — weights averaging (use λ≈30–100)"
                self.lambda_hint.set(
                    f"Weights are flat at λ={self.lambda_var.get():.0g}. "
                    f"Click 'tune 30' or try λ≈{lam_s:.0f} (Δcost≈{spread:.3g})."
                )
            else:
                self.lambda_hint.set(f"Weight spread OK at λ={self.lambda_var.get():.0g}.")
            self.status.set(msg)
        except OSError:
            self.status.set(f"OK — {self.output_prefix.name}")

    def refresh_plot(self) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

            wtxt = "  ".join(f"{k}={self.weight_vars[k].get():.3g}" for k in ("w_pos", "w_heading_so2", "w_vel"))
            try:
                step = int(self.step_var.get())
            except ValueError:
                step = None
            fig = build_rollout_analysis_figure(
                self.output_prefix,
                suptitle_extra=f"λ={self.lambda_var.get():.3g}  {wtxt}",
                log_csv=Path(self.log_var.get()).expanduser(),
                sim_step=step,
                highlight_ref_turn=self.highlight_turn.get(),
            )
        except FileNotFoundError as e:
            messagebox.showerror("Plot failed", str(e))
            return

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar is not None:
            self.toolbar.destroy()
        for child in self.plot_frame.winfo_children():
            child.destroy()

        self.fig = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log_csv", type=Path, nargs="?", default=Path("dubins_stadium_path_tracking_log.csv"))
    p.add_argument("--binary", type=Path, default=None, help="Path to dubins_stadium_mppi_tune_from_log")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    binary = args.binary if args.binary is not None else default_tune_binary()
    if not args.log_csv.is_file():
        print(f"error: log not found: {args.log_csv}", file=sys.stderr)
        return 1

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("error: pip install matplotlib numpy", file=sys.stderr)
        return 1

    app = MppiTuneApp(args.log_csv, binary, args.seed)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
