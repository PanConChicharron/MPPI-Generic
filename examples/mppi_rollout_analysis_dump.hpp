/**
 * Dump one MPPI optimization iteration to CSV (for plot_mppi_rollout_analysis.py).
 *
 * Call immediately after computeControl(), before the state is stepped forward.
 */
#ifndef MPPI_ROLLOUT_ANALYSIS_DUMP_HPP_
#define MPPI_ROLLOUT_ANALYSIS_DUMP_HPP_

#include "mppi_rollout_csv.hpp"

#include <mppi/path/path2d.hpp>
#include <mppi/path/path_reference_generator.hpp>
#include <mppi/utils/gpu_err_chk.cuh>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace mppi
{
namespace rollout_csv
{

inline std::string analysisPrefixForLogStep(const std::string& log_csv_path, const int step)
{
  std::string stem = log_csv_path;
  const size_t n = stem.size();
  if (n > 4U && stem.compare(n - 4U, 4U, ".csv") == 0)
  {
    stem.erase(n - 4U);
  }
  std::ostringstream oss;
  oss << stem << "_mppi_step_" << std::setw(5) << std::setfill('0') << step;
  return oss.str();
}

template <class DYN_T, class CONTROLLER_T, class SAMPLER_T, class CONTROL_TRAJ_T, class OUTPUT_TRAJ_T>
void dumpSingleMppiIteration(DYN_T& model, CONTROLLER_T& controller, SAMPLER_T& sampler,
                             const typename DYN_T::state_array& x, const std::string& prefix, const float dt,
                             const float lambda, const int horizon, const int num_rollouts,
                             const mppi::path::Path2D* path_for_centerline = nullptr, const int sim_step = -1,
                             const float sim_t = -1.0F, const float off_track_distance_m = -1.0F,
                             const std::vector<mppi::path::PathReferenceSample>* ref_horizon = nullptr)
{
  using OutputTraj = OUTPUT_TRAJ_T;

  if (path_for_centerline != nullptr)
  {
    writeCenterline(*path_for_centerline, prefix, false);
  }

  const auto& weights_eig = controller.getSampledCostSeq();
  const auto& raw_eig = controller.getRawRolloutCosts();
  const float baseline = static_cast<float>(controller.getBaselineCost());
  const float normalizer = static_cast<float>(controller.getNormalizerCost());

  std::vector<float> raw_costs(static_cast<size_t>(num_rollouts), 0.0F);
  std::vector<float> unnormalized_importance(static_cast<size_t>(num_rollouts), 0.0F);
  std::vector<float> normalized_weights(static_cast<size_t>(num_rollouts), 0.0F);
  const bool have_gpu_raw = raw_eig.size() == num_rollouts;
  for (int i = 0; i < num_rollouts; ++i)
  {
    const float w = weights_eig(i);
    unnormalized_importance[static_cast<size_t>(i)] = w;
    normalized_weights[static_cast<size_t>(i)] = (normalizer > 0.0F) ? w / normalizer : 0.0F;
    if (have_gpu_raw)
    {
      raw_costs[static_cast<size_t>(i)] = raw_eig(i);
    }
    else
    {
      raw_costs[static_cast<size_t>(i)] =
          (w > 0.0F) ? (baseline - lambda * std::log(w)) : (baseline + 1.0e30F);
    }
  }

  float raw_min = raw_costs[0];
  float raw_max = raw_costs[0];
  for (int i = 1; i < num_rollouts; ++i)
  {
    raw_min = std::min(raw_min, raw_costs[static_cast<size_t>(i)]);
    raw_max = std::max(raw_max, raw_costs[static_cast<size_t>(i)]);
  }

  std::vector<float> host_controls(static_cast<size_t>(num_rollouts) * static_cast<size_t>(horizon) *
                                   static_cast<size_t>(DYN_T::CONTROL_DIM));
  {
    float* device_controls = sampler.getControlSample(0, 0, 0);
    HANDLE_ERROR(cudaMemcpy(host_controls.data(), device_controls, host_controls.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
  }

  std::vector<OutputTraj> sampled_outputs(static_cast<size_t>(num_rollouts), OutputTraj::Zero());
  {
    typename DYN_T::state_array x_local;
    typename DYN_T::state_array x_next = model.getZeroState();
    typename DYN_T::state_array xdot = model.getZeroState();
    typename DYN_T::output_array y_t = DYN_T::output_array::Zero();
    typename DYN_T::control_array u_local = DYN_T::control_array::Zero();
    for (int i = 0; i < num_rollouts; ++i)
    {
      x_local = x;
      for (int t = 0; t < horizon; ++t)
      {
        const float* src = host_controls.data() + (i * horizon + t) * DYN_T::CONTROL_DIM;
        for (int d = 0; d < DYN_T::CONTROL_DIM; ++d)
        {
          u_local(d) = src[d];
        }
        model.enforceConstraints(x_local, u_local);
        model.step(x_local, x_next, xdot, u_local, y_t, static_cast<float>(t), dt);
        sampled_outputs[static_cast<size_t>(i)].col(t) = y_t;
        x_local = x_next;
      }
    }
  }

  const CONTROL_TRAJ_T u_opt = controller.getControlSeq();

  writeMeta<DYN_T>(prefix + "_meta.csv", x, dt, lambda, horizon, num_rollouts, num_rollouts, baseline, normalizer);
  {
    std::ofstream meta_extra((prefix + "_meta.csv").c_str(), std::ios::app);
    if (meta_extra)
    {
      meta_extra << "raw_cost_min," << raw_min << "\n";
      meta_extra << "raw_cost_max," << raw_max << "\n";
      meta_extra << "raw_cost_spread," << (raw_max - raw_min) << "\n";
      if (sim_step >= 0)
      {
        meta_extra << "sim_step," << sim_step << "\n";
        meta_extra << "sim_t," << sim_t << "\n";
        meta_extra << "off_track_distance_m," << off_track_distance_m << "\n";
      }
    }
  }
  writeCosts(prefix + "_costs.csv", raw_costs, unnormalized_importance, normalized_weights);
  writeCombinedTrajectory<DYN_T>(model, x, u_opt, prefix + "_combined.csv", dt);
  writeRolloutTrajectories<DYN_T>(prefix + "_rollouts_xy.csv", x, horizon, sampled_outputs);
  if (ref_horizon != nullptr && !ref_horizon->empty())
  {
    writeReferenceHorizon(prefix + "_reference.csv", *ref_horizon);
  }

  std::cout << "MPPI rollout dump @ sim step " << sim_step << " (t=" << sim_t << " s, dist=" << off_track_distance_m
            << " m): " << prefix << "\n";
}

}  // namespace rollout_csv
}  // namespace mppi

#endif  // MPPI_ROLLOUT_ANALYSIS_DUMP_HPP_
