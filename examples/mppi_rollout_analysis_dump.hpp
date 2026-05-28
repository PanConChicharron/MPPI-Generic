/**
 * Dump one MPPI optimization iteration to CSV (for plot_mppi_rollout_analysis.py).
 *
 * Call immediately after computeControl(), before the state is stepped forward.
 */
#ifndef MPPI_ROLLOUT_ANALYSIS_DUMP_HPP_
#define MPPI_ROLLOUT_ANALYSIS_DUMP_HPP_

#include "mppi_rollout_csv.hpp"

#include <mppi/cost_functions/path_tracking/path_tracking_cost.cuh>
#include <mppi/path/path2d.hpp>
#include <mppi/path/path_reference_generator.hpp>
#include <mppi/utils/gpu_err_chk.cuh>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

template <class DYN_T, class COST_T, class CONTROLLER_T, class SAMPLER_T, class CONTROL_TRAJ_T, class OUTPUT_TRAJ_T>
void dumpSingleMppiIteration(DYN_T& model, COST_T& cost, CONTROLLER_T& controller, SAMPLER_T& sampler,
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

  std::vector<float> path_tracking_cost_host(static_cast<size_t>(num_rollouts), 0.0F);
  {
    int* crash = nullptr;
    for (int i = 0; i < num_rollouts; ++i)
    {
      float path_sum = 0.0F;
      for (int t = 0; t < horizon; ++t)
      {
        const typename DYN_T::output_array y_t = sampled_outputs[static_cast<size_t>(i)].col(t);
        typename DYN_T::control_array u_t = DYN_T::control_array::Zero();
        const float* src = host_controls.data() + (i * horizon + t) * DYN_T::CONTROL_DIM;
        for (int d = 0; d < DYN_T::CONTROL_DIM; ++d)
        {
          u_t(d) = src[d];
        }
        path_sum += cost.computeRunningCost(y_t, u_t, t, crash);
      }
      path_tracking_cost_host[static_cast<size_t>(i)] = path_sum / static_cast<float>(horizon);
    }
  }

  float path_host_min = path_tracking_cost_host[0];
  float path_host_max = path_tracking_cost_host[0];
  for (int i = 1; i < num_rollouts; ++i)
  {
    path_host_min = std::min(path_host_min, path_tracking_cost_host[static_cast<size_t>(i)]);
    path_host_max = std::max(path_host_max, path_tracking_cost_host[static_cast<size_t>(i)]);
  }

  // Diagnostic: compare costs computed on GPU rollout outputs vs host replay outputs.
  // We force sampled trajectories to include all rollouts, then map each sampled trajectory
  // back to rollout index by nearest control sequence in L2.
  {
    const float prev_sample_perc = controller.getPercentageSampledControlTrajectories();
    controller.setPercentageSampledControlTrajectories(1.0F);
    controller.calculateSampledStateTrajectories();
    controller.setPercentageSampledControlTrajectories(prev_sample_perc);

    const std::vector<OutputTraj> sampled_outputs_gpu = controller.getSampledOutputTrajectories();
    const int num_sampled = static_cast<int>(sampled_outputs_gpu.size());
    const size_t per_rollout_ctrl_count = static_cast<size_t>(horizon) * static_cast<size_t>(DYN_T::CONTROL_DIM);
    const size_t per_rollout_ctrl_bytes = per_rollout_ctrl_count * sizeof(float);

    std::vector<float> vis_controls(static_cast<size_t>(num_sampled) * per_rollout_ctrl_count, 0.0F);
    for (int j = 0; j < num_sampled; ++j)
    {
      float* vis_ptr = sampler.getVisControlSample(j, 0, 0);
      HANDLE_ERROR(cudaMemcpy(vis_controls.data() + static_cast<size_t>(j) * per_rollout_ctrl_count, vis_ptr,
                              per_rollout_ctrl_bytes, cudaMemcpyDeviceToHost));
    }

    std::ofstream dbg((prefix + "_gpu_output_cost_debug.csv").c_str());
    if (dbg)
    {
      dbg << "sampled_index,matched_rollout,control_match_rmse,path_cost_from_gpu_output,path_cost_from_host_replay,"
             "raw_cost,normalized_weight\n";
      const int gpu_eval_horizon = std::max(1, horizon - 1);  // sampled_trajectories_ keeps first (H-1) outputs.
      int* crash = nullptr;
      for (int j = 0; j < num_sampled; ++j)
      {
        int best_rollout = -1;
        double best_l2 = std::numeric_limits<double>::infinity();
        const float* vis_ctrl = vis_controls.data() + static_cast<size_t>(j) * per_rollout_ctrl_count;
        for (int i = 0; i < num_rollouts; ++i)
        {
          const float* host_ctrl = host_controls.data() + static_cast<size_t>(i) * per_rollout_ctrl_count;
          double l2 = 0.0;
          for (size_t k = 0; k < per_rollout_ctrl_count; ++k)
          {
            const double d = static_cast<double>(vis_ctrl[k]) - static_cast<double>(host_ctrl[k]);
            l2 += d * d;
          }
          if (l2 < best_l2)
          {
            best_l2 = l2;
            best_rollout = i;
          }
        }

        const double rmse = std::sqrt(best_l2 / static_cast<double>(per_rollout_ctrl_count));
        // sampled index 0 is the nominal trajectory, so it should not map to a rollout sample.
        if (best_rollout < 0 || rmse > 1.0E-5)
        {
          dbg << j << ",-1," << rmse << ",nan,nan,nan,nan\n";
          continue;
        }

        float path_gpu_sum = 0.0F;
        for (int t = 0; t < gpu_eval_horizon; ++t)
        {
          const typename DYN_T::output_array y_t = sampled_outputs_gpu[static_cast<size_t>(j)].col(t);
          typename DYN_T::control_array u_t = DYN_T::control_array::Zero();
          const float* src = host_controls.data() +
                             (static_cast<size_t>(best_rollout) * static_cast<size_t>(horizon) + static_cast<size_t>(t)) *
                                 static_cast<size_t>(DYN_T::CONTROL_DIM);
          for (int d = 0; d < DYN_T::CONTROL_DIM; ++d)
          {
            u_t(d) = src[d];
          }
          path_gpu_sum += cost.computeRunningCost(y_t, u_t, t, crash);
        }
        const float path_gpu_mean = path_gpu_sum / static_cast<float>(gpu_eval_horizon);

        dbg << j << "," << best_rollout << "," << rmse << "," << path_gpu_mean << ","
            << path_tracking_cost_host[static_cast<size_t>(best_rollout)] << ","
            << raw_costs[static_cast<size_t>(best_rollout)] << ","
            << normalized_weights[static_cast<size_t>(best_rollout)] << "\n";
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
      meta_extra << "path_tracking_cost_host_min," << path_host_min << "\n";
      meta_extra << "path_tracking_cost_host_max," << path_host_max << "\n";
      meta_extra << "path_tracking_cost_host_spread," << (path_host_max - path_host_min) << "\n";
      if (sim_step >= 0)
      {
        meta_extra << "sim_step," << sim_step << "\n";
        meta_extra << "sim_t," << sim_t << "\n";
        meta_extra << "off_track_distance_m," << off_track_distance_m << "\n";
      }
    }
  }
  writeCosts(prefix + "_costs.csv", raw_costs, unnormalized_importance, normalized_weights, path_tracking_cost_host);
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
