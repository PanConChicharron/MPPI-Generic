/**
 * Header-only CSV writers for MPPI rollout / iteration analysis.
 *
 * All writers are templated on the dynamics type so they can be reused across
 * different `*_mppi_rollout_analysis_example.cu` style examples. The dynamics
 * type must expose `DYN_T::DYN_PARAMS_T` with `StateIndex`, `OutputIndex`, and
 * `ControlIndex` enums (the convention used by every dynamics class in this
 * repo, e.g. `DubinsBicycle`).
 *
 * Schemas (in lock-step with `examples/plot_mppi_rollout_analysis.py`):
 *   <prefix>_centerline.csv : x_m,y_m
 *   <prefix>_meta.csv       : key,value
 *   <prefix>_costs.csv      : rollout_index,raw_cost,unnormalized_importance,normalized_weight
 *   <prefix>_combined.csv   : step,t,x,y,yaw,vel_x,steer,u_accel,u_steer
 *   <prefix>_rollouts_xy.csv: rollout_index,step,x,y,yaw,vel_x
 */
#ifndef MPPI_ROLLOUT_CSV_HPP_
#define MPPI_ROLLOUT_CSV_HPP_

#include <Eigen/Dense>
#include <mppi/path/path2d.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace mppi
{
namespace rollout_csv
{
inline void writeCenterline(const mppi::path::Path2D& path, const std::string& prefix, bool log_path = true)
{
  const std::string out = prefix + "_centerline.csv";
  std::ofstream f(out.c_str());
  if (!f)
  {
    return;
  }
  f << "x_m,y_m\n";
  for (const mppi::path::PathAnchor& a : path.anchors())
  {
    f << a.x << "," << a.y << "\n";
  }
  if (log_path)
  {
    std::cout << "Wrote centerline: " << out << "\n";
  }
}

/**
 * Sister to `writeCenterline` for the closed-loop tracking examples that pass
 * a full log path like "foo.csv": writes "foo_centerline.csv" (or appends
 * "_centerline.csv" if the path doesn't end in ".csv").
 */
inline void writeCenterlineForLog(const mppi::path::Path2D& path, const std::string& log_csv_path)
{
  std::string out = log_csv_path;
  const size_t n = out.size();
  if (n > 4U && out.compare(n - 4U, 4U, ".csv") == 0)
  {
    out.insert(n - 4U, "_centerline");
  }
  else
  {
    out += "_centerline.csv";
  }
  std::ofstream f(out.c_str());
  if (!f)
  {
    return;
  }
  f << "x_m,y_m\n";
  for (const mppi::path::PathAnchor& a : path.anchors())
  {
    f << a.x << "," << a.y << "\n";
  }
  std::cout << "Wrote centerline for plot: " << out << "\n";
}

template <class DYN_T>
void writeMeta(const std::string& path, const typename DYN_T::state_array& x, float dt, float lambda, int horizon,
               int num_rollouts, int num_logged_traj, float baseline, float normalizer)
{
  using S = typename DYN_T::DYN_PARAMS_T::StateIndex;
  std::ofstream f(path.c_str());
  if (!f)
  {
    return;
  }
  f << "key,value\n";
  f << "dt," << dt << "\n";
  f << "horizon," << horizon << "\n";
  f << "num_rollouts," << num_rollouts << "\n";
  f << "num_logged_trajectories," << num_logged_traj << "\n";
  f << "lambda," << lambda << "\n";
  f << "baseline," << baseline << "\n";
  f << "normalizer," << normalizer << "\n";
  f << "init_pos_x," << x(static_cast<int>(S::POS_X)) << "\n";
  f << "init_pos_y," << x(static_cast<int>(S::POS_Y)) << "\n";
  f << "init_yaw," << x(static_cast<int>(S::YAW)) << "\n";
  f << "init_vel_x," << x(static_cast<int>(S::VEL_X)) << "\n";
  f << "init_steer," << x(static_cast<int>(S::STEER_ANGLE)) << "\n";
}

inline void writeCosts(const std::string& path, const std::vector<float>& raw_costs,
                       const std::vector<float>& unnormalized_importance,
                       const std::vector<float>& normalized_weights)
{
  std::ofstream f(path.c_str());
  if (!f)
  {
    return;
  }
  f << "rollout_index,raw_cost,unnormalized_importance,normalized_weight\n";
  const int n = static_cast<int>(raw_costs.size());
  for (int i = 0; i < n; ++i)
  {
    f << i << "," << raw_costs[i] << "," << unnormalized_importance[i] << "," << normalized_weights[i] << "\n";
  }
}

/**
 * Replays the optimal control sequence through the dynamics model on the host
 * to log the realized trajectory in CSV form. This is just for plotting — the
 * controls themselves came from MPPI on the GPU.
 */
template <class DYN_T, class CONTROL_TRAJ_T>
void writeCombinedTrajectory(DYN_T& model, const typename DYN_T::state_array& x0, const CONTROL_TRAJ_T& u,
                             const std::string& path, float dt)
{
  using S = typename DYN_T::DYN_PARAMS_T::StateIndex;
  using C = typename DYN_T::DYN_PARAMS_T::ControlIndex;
  std::ofstream f(path.c_str());
  if (!f)
  {
    return;
  }
  f << "step,t,x,y,yaw,vel_x,steer,u_accel,u_steer\n";
  typename DYN_T::state_array x = x0;
  typename DYN_T::state_array x_next = model.getZeroState();
  typename DYN_T::state_array xdot = model.getZeroState();
  typename DYN_T::output_array y = DYN_T::output_array::Zero();
  typename DYN_T::control_array u_step = DYN_T::control_array::Zero();

  f << "0,0," << x(static_cast<int>(S::POS_X)) << "," << x(static_cast<int>(S::POS_Y)) << ","
    << x(static_cast<int>(S::YAW)) << "," << x(static_cast<int>(S::VEL_X)) << ","
    << x(static_cast<int>(S::STEER_ANGLE)) << ",0,0\n";

  const int steps = static_cast<int>(u.cols());
  for (int k = 0; k < steps; ++k)
  {
    u_step = u.col(k);
    model.enforceConstraints(x, u_step);
    model.step(x, x_next, xdot, u_step, y, static_cast<float>(k), dt);
    const float t = static_cast<float>(k + 1) * dt;
    f << (k + 1) << "," << t << "," << x_next(static_cast<int>(S::POS_X)) << "," << x_next(static_cast<int>(S::POS_Y))
      << "," << x_next(static_cast<int>(S::YAW)) << "," << x_next(static_cast<int>(S::VEL_X)) << ","
      << x_next(static_cast<int>(S::STEER_ANGLE)) << "," << u_step(static_cast<int>(C::ACCEL)) << ","
      << u_step(static_cast<int>(C::STEER)) << "\n";
    x = x_next;
  }
}

/**
 * Per-rollout (x, y, yaw, vel_x) trajectories. `outputs[i]` is expected to be
 * an Eigen matrix of shape (OUTPUT_DIM, horizon) for rollout `i`, indexed by
 * `DYN_T::DYN_PARAMS_T::OutputIndex`.
 */
template <class DYN_T, class TrajT>
void writeRolloutTrajectories(const std::string& path, const typename DYN_T::state_array& x0, int horizon,
                              const std::vector<TrajT>& outputs)
{
  using S = typename DYN_T::DYN_PARAMS_T::StateIndex;
  using O = typename DYN_T::DYN_PARAMS_T::OutputIndex;
  std::ofstream f(path.c_str());
  if (!f)
  {
    return;
  }
  f << "rollout_index,step,x,y,yaw,vel_x\n";
  const int n_out = static_cast<int>(outputs.size());
  const int oix = static_cast<int>(O::POS_X);
  const int oiy = static_cast<int>(O::POS_Y);
  const int oiyaw = static_cast<int>(O::YAW);
  const int oiv = static_cast<int>(O::VEL_X);

  for (int r = 0; r < n_out; ++r)
  {
    f << r << ",0," << x0(static_cast<int>(S::POS_X)) << "," << x0(static_cast<int>(S::POS_Y)) << ","
      << x0(static_cast<int>(S::YAW)) << "," << x0(static_cast<int>(S::VEL_X)) << "\n";
    const TrajT& traj = outputs[static_cast<size_t>(r)];
    for (int c = 0; c < horizon; ++c)
    {
      f << r << "," << (c + 1) << "," << traj(oix, c) << "," << traj(oiy, c) << "," << traj(oiyaw, c) << ","
        << traj(oiv, c) << "\n";
    }
  }
}
}  // namespace rollout_csv
}  // namespace mppi

#endif  // MPPI_ROLLOUT_CSV_HPP_
