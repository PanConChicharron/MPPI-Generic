/**
 * @file data_manager.hpp
 * @brief Modular CSV / rollout snapshot dumping for MPPI examples and tools.
 *
 * Bundles closed-loop temporal logs, per-step rollout snapshots (top-N by weight),
 * and single-iteration analysis dumps. Output schemas match:
 *   scripts/mppi/plot_racer_dubins_temporal_mppi.py
 *   scripts/mppi/plot_mppi_rollouts_at_step.py
 *   scripts/mppi/plot_mppi_lambda_retune.py
 *   scripts/mppi/plot_mppi_rollout_analysis.py
 */
#ifndef MPPI_UTILS_DATA_MANAGER_HPP_
#define MPPI_UTILS_DATA_MANAGER_HPP_

#include <mppi/utils/rollout_csv.hpp>
#include <mppi/utils/gpu_err_chk.cuh>

#include <cuda_runtime.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mppi
{
namespace data
{
constexpr int kDefaultTopRollouts = 400;

enum class PathTrackingLogSchema
{
  kRefV,
  kRefVPoseTarget,
};

struct PathTrackingStepLog
{
  float t = 0.0F;
  float pos_x = 0.0F;
  float pos_y = 0.0F;
  float yaw = 0.0F;
  float vel_x = 0.0F;
  float steer_angle = 0.0F;
  float brake_state = 0.0F;
  float u_accel = 0.0F;
  float u_steer = 0.0F;
  float nom_u_accel = 0.0F;
  float nom_u_steer = 0.0F;
  float ref_x = 0.0F;
  float ref_y = 0.0F;
  float ref_yaw = 0.0F;
  float ref_v = 0.0F;
  float ref_v_pose = 0.0F;
  float ref_v_target = 0.0F;
  float arc_s = 0.0F;
  float lat_err = 0.0F;
  float baseline = 0.0F;
};

struct RolloutOutputIndices
{
  int x = 0;
  int y = 1;
  int yaw = 2;
  int vel = 3;

  RolloutOutputIndices() = default;
  RolloutOutputIndices(int x_idx, int y_idx, int yaw_idx, int vel_idx) : x(x_idx), y(y_idx), yaw(yaw_idx), vel(vel_idx)
  {
  }
};

struct RolloutWeightBundle
{
  std::vector<int> rollout_indices;
  std::vector<float> raw_costs;
  std::vector<float> unnormalized_importance;
  std::vector<float> normalized_weights;
  float baseline = 0.0F;
  float normalizer = 0.0F;
};

inline std::string rolloutDirectoryFromLog(const std::string& log_csv_path)
{
  std::string stem = log_csv_path;
  const size_t n = stem.size();
  if (n > 4U && stem.compare(n - 4U, 4U, ".csv") == 0)
  {
    stem.erase(n - 4U);
  }
  return stem + "_rollouts";
}

inline std::string stepDirectoryName(int step)
{
  std::ostringstream oss;
  oss << "step_" << std::setw(6) << std::setfill('0') << step;
  return oss.str();
}

inline bool ensureDirectory(const std::string& path)
{
  struct stat st;
  if (stat(path.c_str(), &st) == 0)
  {
    return S_ISDIR(st.st_mode);
  }
  return mkdir(path.c_str(), 0755) == 0;
}

template <class CTRL_T>
inline void extractWeightsFromController(const CTRL_T& controller, float lambda, RolloutWeightBundle& out)
{
  const auto& weights_eig = controller.getSampledCostSeq();
  const float baseline = static_cast<float>(controller.getBaselineCost());
  const float normalizer = static_cast<float>(controller.getNormalizerCost());
  const int num_rollouts = static_cast<int>(weights_eig.size());

  out.baseline = baseline;
  out.normalizer = normalizer;
  out.rollout_indices.resize(static_cast<size_t>(num_rollouts));
  out.raw_costs.resize(static_cast<size_t>(num_rollouts));
  out.unnormalized_importance.resize(static_cast<size_t>(num_rollouts));
  out.normalized_weights.resize(static_cast<size_t>(num_rollouts));

  for (int i = 0; i < num_rollouts; ++i)
  {
    const float w = static_cast<float>(weights_eig(i));
    out.rollout_indices[static_cast<size_t>(i)] = i;
    out.unnormalized_importance[static_cast<size_t>(i)] = w;
    out.normalized_weights[static_cast<size_t>(i)] = (normalizer > 0.0F) ? w / normalizer : 0.0F;
    out.raw_costs[static_cast<size_t>(i)] =
        (w > 0.0F) ? (baseline - lambda * std::log(w)) : (baseline + 1.0e30F);
  }
}

inline void selectTopRolloutsByWeight(RolloutWeightBundle& bundle, int top_n)
{
  const int n = static_cast<int>(bundle.normalized_weights.size());
  if (top_n <= 0 || top_n >= n)
  {
    return;
  }

  std::vector<int> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + top_n, order.end(), [&](int a, int b) {
    return bundle.normalized_weights[static_cast<size_t>(a)] > bundle.normalized_weights[static_cast<size_t>(b)];
  });
  order.resize(static_cast<size_t>(top_n));

  RolloutWeightBundle top;
  top.baseline = bundle.baseline;
  top.normalizer = bundle.normalizer;
  top.rollout_indices.resize(static_cast<size_t>(top_n));
  top.raw_costs.resize(static_cast<size_t>(top_n));
  top.unnormalized_importance.resize(static_cast<size_t>(top_n));
  top.normalized_weights.resize(static_cast<size_t>(top_n));
  for (int i = 0; i < top_n; ++i)
  {
    const int src = order[static_cast<size_t>(i)];
    top.rollout_indices[static_cast<size_t>(i)] = bundle.rollout_indices[static_cast<size_t>(src)];
    top.raw_costs[static_cast<size_t>(i)] = bundle.raw_costs[static_cast<size_t>(src)];
    top.unnormalized_importance[static_cast<size_t>(i)] = bundle.unnormalized_importance[static_cast<size_t>(src)];
    top.normalized_weights[static_cast<size_t>(i)] = bundle.normalized_weights[static_cast<size_t>(src)];
  }
  bundle = std::move(top);
}

template <class DYN_T, class SAMPLER_T>
void copySamplerControlsToHost(SAMPLER_T& sampler, int horizon, int num_rollouts, std::vector<float>& host_controls)
{
  host_controls.assign(static_cast<size_t>(num_rollouts) * static_cast<size_t>(horizon) *
                           static_cast<size_t>(DYN_T::CONTROL_DIM),
                       0.0F);
  float* device_controls = sampler.getControlSample(0, 0, 0);
  HANDLE_ERROR(cudaMemcpy(host_controls.data(), device_controls, host_controls.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

template <class DYN_T, class SAMPLER_T, class OutputTrajT>
void hostReplayRollouts(DYN_T& model, SAMPLER_T& sampler, const typename DYN_T::state_array& x0, int horizon, float dt,
                        int num_rollouts, const std::vector<int>& rollout_indices, std::vector<OutputTrajT>& outputs,
                        std::vector<float>* host_controls_out = nullptr)
{
  const int k = static_cast<int>(rollout_indices.size());
  outputs.assign(static_cast<size_t>(k), OutputTrajT::Zero());

  std::vector<float> host_controls;
  copySamplerControlsToHost<DYN_T>(sampler, horizon, num_rollouts, host_controls);
  if (host_controls_out != nullptr)
  {
    *host_controls_out = host_controls;
  }

  typename DYN_T::state_array x_local;
  typename DYN_T::state_array x_next = model.getZeroState();
  typename DYN_T::state_array xdot = model.getZeroState();
  typename DYN_T::output_array y_t = DYN_T::output_array::Zero();
  typename DYN_T::control_array u_local = DYN_T::control_array::Zero();

  for (int out_idx = 0; out_idx < k; ++out_idx)
  {
    const int rollout = rollout_indices[static_cast<size_t>(out_idx)];
    x_local = x0;
    for (int t = 0; t < horizon; ++t)
    {
      const float* src = host_controls.data() +
                       (static_cast<size_t>(rollout) * static_cast<size_t>(horizon) + static_cast<size_t>(t)) *
                           static_cast<size_t>(DYN_T::CONTROL_DIM);
      for (int d = 0; d < DYN_T::CONTROL_DIM; ++d)
      {
        u_local(d) = src[d];
      }
      model.enforceConstraints(x_local, u_local);
      model.step(x_local, x_next, xdot, u_local, y_t, static_cast<float>(t), dt);
      outputs[static_cast<size_t>(out_idx)].col(t) = y_t;
      x_local = x_next;
    }
  }
}

template <class DYN_T, class SAMPLER_T, class OutputTrajT>
void hostReplayAllRollouts(DYN_T& model, SAMPLER_T& sampler, const typename DYN_T::state_array& x0, int horizon,
                           float dt, int num_rollouts, std::vector<OutputTrajT>& outputs)
{
  std::vector<int> all(static_cast<size_t>(num_rollouts));
  std::iota(all.begin(), all.end(), 0);
  hostReplayRollouts<DYN_T, SAMPLER_T, OutputTrajT>(model, sampler, x0, horizon, dt, num_rollouts, all, outputs);
}

/**
 * Central dumping helper for MPPI examples.
 *
 * @tparam DYN_T Dynamics type (must expose POS_X/Y/YAW/VEL_X/STEER_ANGLE state indices).
 */
template <class DYN_T>
class MppiDataManager
{
public:
  bool beginRun(const std::string& log_csv_path, const mppi::path::Path2D& path,
                PathTrackingLogSchema schema = PathTrackingLogSchema::kRefVPoseTarget)
  {
    log_path_ = log_csv_path;
    rollout_dir_ = rolloutDirectoryFromLog(log_csv_path);
    schema_ = schema;

    mppi::rollout_csv::writeCenterlineForLog(path, log_csv_path);
    if (!ensureDirectory(rollout_dir_))
    {
      std::cerr << "Could not create rollout directory: " << rollout_dir_ << "\n";
      return false;
    }

    rollout_index_path_ = rollout_dir_ + "/steps_index.csv";
    {
      std::ofstream idx(rollout_index_path_.c_str());
      if (!idx)
      {
        std::cerr << "Could not open rollout index: " << rollout_index_path_ << "\n";
        return false;
      }
      idx << "step,sim_time,prefix\n";
    }

    temporal_log_.open(log_csv_path.c_str());
    if (!temporal_log_)
    {
      std::cerr << "Could not open temporal log: " << log_csv_path << "\n";
      return false;
    }
    writeTemporalHeader();
    temporal_log_ << std::scientific;
    return true;
  }

  bool beginAnalysisRun(const std::string& prefix, const mppi::path::Path2D& path)
  {
    analysis_prefix_ = prefix;
    mppi::rollout_csv::writeCenterline(path, prefix);
    return true;
  }

  void setRoadBoundaryLimits(float left, float right)
  {
    boundary_left_ = left;
    boundary_right_ = right;
  }

  void logPathTrackingStep(const PathTrackingStepLog& rec)
  {
    if (!temporal_log_)
    {
      return;
    }
    temporal_log_ << rec.t << "," << rec.pos_x << "," << rec.pos_y << "," << rec.yaw << "," << rec.vel_x << ","
                  << rec.steer_angle << "," << rec.brake_state << "," << rec.u_accel << "," << rec.u_steer << ","
                  << rec.nom_u_accel << "," << rec.nom_u_steer << "," << rec.ref_x << "," << rec.ref_y << ","
                  << rec.ref_yaw << ",";
    if (schema_ == PathTrackingLogSchema::kRefV)
    {
      temporal_log_ << rec.ref_v << ",";
    }
    else
    {
      temporal_log_ << rec.ref_v_pose << "," << rec.ref_v_target << ",";
    }
    temporal_log_ << rec.arc_s << "," << rec.lat_err << "," << rec.baseline << "\n";
  }

  template <class CTRL_T, class SAMPLER_T, class ControlTrajT>
  void dumpRolloutSnapshot(int step, float sim_time, const typename DYN_T::state_array& x, CTRL_T& controller,
                           DYN_T& model, SAMPLER_T& sampler, int horizon, float lambda, float dt,
                           const ControlTrajT& u_opt, const RolloutOutputIndices& out_idx,
                           int top_n = kDefaultTopRollouts)
  {
    using OutputTrajT = typename CTRL_T::output_trajectory;
    RolloutWeightBundle weights;
    extractWeightsFromController(controller, lambda, weights);

    const std::string step_dir = rollout_dir_ + "/" + stepDirectoryName(step);
    if (!ensureDirectory(step_dir))
    {
      return;
    }

    const int num_rollouts = static_cast<int>(controller.getSampledCostSeq().size());
    rollout_csv::writeMeta<DYN_T>(step_dir + "/meta.csv", x, dt, lambda, horizon, num_rollouts, num_rollouts,
                                  weights.baseline, weights.normalizer, step, sim_time, boundary_left_,
                                  boundary_right_);
    rollout_csv::writeCosts(step_dir + "/costs.csv", weights.raw_costs, weights.unnormalized_importance,
                            weights.normalized_weights);
    rollout_csv::writeCombinedTrajectory<DYN_T>(model, x, u_opt, step_dir + "/combined.csv", dt);

    RolloutWeightBundle top_weights = weights;
    selectTopRolloutsByWeight(top_weights, top_n);
    std::vector<OutputTrajT> top_outputs;
    std::vector<float> host_controls;
    hostReplayRollouts<DYN_T, SAMPLER_T, OutputTrajT>(model, sampler, x, horizon, dt, num_rollouts,
                                                    top_weights.rollout_indices, top_outputs, &host_controls);
    rollout_csv::writeRolloutTrajectories<DYN_T, OutputTrajT>(step_dir + "/rollouts_xy.csv", x, horizon, top_outputs,
                                                              top_weights.rollout_indices, out_idx.x, out_idx.y,
                                                              out_idx.yaw, out_idx.vel);
    rollout_csv::writeRolloutControls<DYN_T>(step_dir + "/rollouts_controls.csv", host_controls, horizon, num_rollouts,
                                             top_weights.rollout_indices);

    std::ofstream idx(rollout_index_path_.c_str(), std::ios::app);
    if (idx)
    {
      idx << step << "," << sim_time << "," << step_dir << "\n";
    }
  }

  template <class CTRL_T, class SAMPLER_T, class ControlTrajT>
  void dumpSingleIterationFromController(const typename DYN_T::state_array& x, CTRL_T& controller, DYN_T& model,
                                         SAMPLER_T& sampler, int horizon, float lambda, float dt,
                                         const ControlTrajT& u_opt, const RolloutOutputIndices& out_idx,
                                         int top_n = kDefaultTopRollouts)
  {
    using OutputTrajT = typename CTRL_T::output_trajectory;
    RolloutWeightBundle weights;
    extractWeightsFromController(controller, lambda, weights);
    const int num_rollouts = static_cast<int>(controller.getSampledCostSeq().size());

    rollout_csv::writeMeta<DYN_T>(analysis_prefix_ + "_meta.csv", x, dt, lambda, horizon, num_rollouts, num_rollouts,
                                  weights.baseline, weights.normalizer, -1, -1.0F, boundary_left_, boundary_right_);
    rollout_csv::writeCosts(analysis_prefix_ + "_costs.csv", weights.raw_costs, weights.unnormalized_importance,
                            weights.normalized_weights);
    rollout_csv::writeCombinedTrajectory<DYN_T>(model, x, u_opt, analysis_prefix_ + "_combined.csv", dt);

    RolloutWeightBundle top_weights = weights;
    selectTopRolloutsByWeight(top_weights, top_n);
    std::vector<OutputTrajT> top_outputs;
    std::vector<float> host_controls;
    hostReplayRollouts<DYN_T, SAMPLER_T, OutputTrajT>(model, sampler, x, horizon, dt, num_rollouts,
                                                    top_weights.rollout_indices, top_outputs, &host_controls);
    rollout_csv::writeRolloutTrajectories<DYN_T, OutputTrajT>(analysis_prefix_ + "_rollouts_xy.csv", x, horizon,
                                                              top_outputs, top_weights.rollout_indices, out_idx.x,
                                                              out_idx.y, out_idx.yaw, out_idx.vel);
    rollout_csv::writeRolloutControls<DYN_T>(analysis_prefix_ + "_rollouts_controls.csv", host_controls, horizon,
                                             num_rollouts, top_weights.rollout_indices);
  }

  template <class SAMPLER_T, class OutputTrajT, class ControlTrajT>
  void dumpSingleIterationHostReplay(const typename DYN_T::state_array& x, DYN_T& model, SAMPLER_T& sampler,
                                     int horizon, float lambda, float dt, int num_rollouts,
                                     const ControlTrajT& u_opt, const std::vector<float>& raw_costs, float baseline,
                                     const RolloutOutputIndices& out_idx, int top_n = kDefaultTopRollouts)
  {
    std::vector<float> unnormalized_importance(static_cast<size_t>(num_rollouts), 0.0F);
    for (int i = 0; i < num_rollouts; ++i)
    {
      unnormalized_importance[static_cast<size_t>(i)] = std::exp(-(raw_costs[static_cast<size_t>(i)] - baseline) / lambda);
    }
    const float normalizer = std::accumulate(unnormalized_importance.begin(), unnormalized_importance.end(), 0.0F);
    std::vector<float> normalized_weights(static_cast<size_t>(num_rollouts), 0.0F);
    for (int i = 0; i < num_rollouts; ++i)
    {
      normalized_weights[static_cast<size_t>(i)] =
          (normalizer > 0.0F) ? unnormalized_importance[static_cast<size_t>(i)] / normalizer : 0.0F;
    }

    rollout_csv::writeMeta<DYN_T>(analysis_prefix_ + "_meta.csv", x, dt, lambda, horizon, num_rollouts, num_rollouts,
                                  baseline, normalizer, -1, -1.0F, boundary_left_, boundary_right_);
    rollout_csv::writeCosts(analysis_prefix_ + "_costs.csv", raw_costs, unnormalized_importance, normalized_weights);
    rollout_csv::writeCombinedTrajectory<DYN_T>(model, x, u_opt, analysis_prefix_ + "_combined.csv", dt);

    RolloutWeightBundle top;
    top.baseline = baseline;
    top.normalizer = normalizer;
    top.rollout_indices.resize(static_cast<size_t>(num_rollouts));
    top.raw_costs = raw_costs;
    top.unnormalized_importance = unnormalized_importance;
    top.normalized_weights = normalized_weights;
    selectTopRolloutsByWeight(top, top_n);

    std::vector<OutputTrajT> top_outputs;
    std::vector<float> host_controls;
    hostReplayRollouts<DYN_T, SAMPLER_T, OutputTrajT>(model, sampler, x, horizon, dt, num_rollouts, top.rollout_indices,
                                                      top_outputs, &host_controls);
    rollout_csv::writeRolloutTrajectories<DYN_T, OutputTrajT>(analysis_prefix_ + "_rollouts_xy.csv", x, horizon,
                                                              top_outputs, top.rollout_indices, out_idx.x, out_idx.y,
                                                              out_idx.yaw, out_idx.vel);
    rollout_csv::writeRolloutControls<DYN_T>(analysis_prefix_ + "_rollouts_controls.csv", host_controls, horizon,
                                             num_rollouts, top.rollout_indices);
  }

  void close()
  {
    if (temporal_log_.is_open())
    {
      temporal_log_.close();
    }
  }

  const std::string& logPath() const
  {
    return log_path_;
  }
  const std::string& rolloutDirectory() const
  {
    return rollout_dir_;
  }

private:
  void writeTemporalHeader()
  {
    temporal_log_ << "t,pos_x,pos_y,yaw,vel_x,steer_angle,brake_state,u_accel,u_steer,nom_u_accel,nom_u_steer,"
                     "ref_x,ref_y,ref_yaw,";
    if (schema_ == PathTrackingLogSchema::kRefV)
    {
      temporal_log_ << "ref_v,";
    }
    else
    {
      temporal_log_ << "ref_v_pose,ref_v_target,";
    }
    temporal_log_ << "arc_s,lat_err,baseline\n";
  }

  std::string log_path_;
  std::string rollout_dir_;
  std::string rollout_index_path_;
  std::string analysis_prefix_;
  PathTrackingLogSchema schema_ = PathTrackingLogSchema::kRefVPoseTarget;
  float boundary_left_ = -1.0F;
  float boundary_right_ = -1.0F;
  std::ofstream temporal_log_;
};

}  // namespace data
}  // namespace mppi

#endif  // MPPI_UTILS_DATA_MANAGER_HPP_
