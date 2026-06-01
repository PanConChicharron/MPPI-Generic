/**
 * Racer Dubins stadium path tracking — log one MPPI iteration (rollouts, costs, weights).
 *
 * Build: cmake --build build --target racer_dubins_stadium_mppi_rollout_analysis_example
 * Run:   ./build/examples/racer_dubins_stadium_mppi_rollout_analysis_example [output_prefix]
 * Plot:  python3 examples/plot_mppi_rollout_analysis.py racer_dubins_stadium_mppi_rollout_analysis
 *
 * Same single-iteration structure as dubins_stadium_mppi_rollout_analysis_example, but uses
 * RacerDubins + RacerCost to match racer_dubins_stadium_path_tracking_example.cu.
 *
 * Per-rollout costs/weights come from the controller after one GPU MPPI iteration:
 *   weights[i] = controller.getSampledCostSeq()[i]
 *   baseline   = controller.getBaselineCost()
 *   raw_cost[i] = baseline - lambda * log(weights[i])
 *
 * Rollout (x, y) trajectories are host-replayed from GPU noise samples (output_d_ is not
 * readable from the host in this codebase).
 */
#include "mppi_rollout_csv.hpp"

#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/cost_functions/racer/racer_cost.cuh>
#include <mppi/cost_functions/racer/racer_cost_bridge.hpp>
#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>
#include <mppi/feedback_controllers/zero_feedback.cuh>
#include <mppi/path/path2d.hpp>
#include <mppi/path/path_reference_generator.hpp>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

namespace
{
constexpr int kMppiHorizon = 50;
constexpr int kRefHorizon = kMppiHorizon;
constexpr float kDt = 0.1F;
constexpr int kNumRollouts = 32 * 1024;
constexpr float kTargetSpeed = 5.0F;
constexpr float kVMax = 5.0F;

constexpr float kStraightLength = 40.0F;
constexpr float kTurnRadius = 10.0F;
constexpr int kSamplesPerArc = 48;
constexpr float kInitArcLength = kStraightLength - 2.0F;
constexpr float kLambda = 1000.0F;

using DYN = RacerDubins;
using COST = RacerCost<kRefHorizon>;
using FB = ZeroFeedback<DYN, kMppiHorizon>;
using SAMPLER = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
using Mppi = VanillaMPPIController<DYN, COST, FB, kMppiHorizon, kNumRollouts, SAMPLER>;

void writeRacerCombinedTrajectory(DYN& model, const DYN::state_array& x0, const Mppi::control_trajectory& u,
                                  const std::string& path, const float dt)
{
  std::ofstream f(path.c_str());
  if (!f)
  {
    return;
  }
  f << "step,t,x,y,yaw,vel_x,steer,u_accel,u_steer\n";
  DYN::state_array x = x0;
  DYN::state_array x_next = model.getZeroState();
  DYN::state_array xdot = model.getZeroState();
  DYN::output_array y = DYN::output_array::Zero();
  DYN::control_array u_step = DYN::control_array::Zero();

  f << "0,0," << x(static_cast<int>(RacerDubinsParams::StateIndex::POS_X)) << ","
    << x(static_cast<int>(RacerDubinsParams::StateIndex::POS_Y)) << ","
    << x(static_cast<int>(RacerDubinsParams::StateIndex::YAW)) << ","
    << x(static_cast<int>(RacerDubinsParams::StateIndex::VEL_X)) << ","
    << x(static_cast<int>(RacerDubinsParams::StateIndex::STEER_ANGLE)) << ",0,0\n";

  const int steps = static_cast<int>(u.cols());
  for (int k = 0; k < steps; ++k)
  {
    u_step = u.col(k);
    model.enforceConstraints(x, u_step);
    model.step(x, x_next, xdot, u_step, y, static_cast<float>(k), dt);
    const float t = static_cast<float>(k + 1) * dt;
    f << (k + 1) << "," << t << "," << x_next(static_cast<int>(RacerDubinsParams::StateIndex::POS_X)) << ","
      << x_next(static_cast<int>(RacerDubinsParams::StateIndex::POS_Y)) << ","
      << x_next(static_cast<int>(RacerDubinsParams::StateIndex::YAW)) << ","
      << x_next(static_cast<int>(RacerDubinsParams::StateIndex::VEL_X)) << ","
      << x_next(static_cast<int>(RacerDubinsParams::StateIndex::STEER_ANGLE)) << ","
      << u_step(static_cast<int>(RacerDubinsParams::ControlIndex::THROTTLE_BRAKE)) << ","
      << u_step(static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD)) << "\n";
    x = x_next;
  }
}

void writeRacerRolloutTrajectories(const std::string& path,
                                   const std::vector<std::vector<DYN::state_array>>& rollout_states)
{
  std::ofstream f(path.c_str());
  if (!f)
  {
    return;
  }
  f << "rollout_index,step,x,y,yaw,vel_x\n";
  const int n_out = static_cast<int>(rollout_states.size());
  for (int r = 0; r < n_out; ++r)
  {
    const auto& traj = rollout_states[static_cast<size_t>(r)];
    for (size_t step = 0; step < traj.size(); ++step)
    {
      const DYN::state_array& s = traj[step];
      f << r << "," << step << "," << s(static_cast<int>(RacerDubinsParams::StateIndex::POS_X)) << ","
        << s(static_cast<int>(RacerDubinsParams::StateIndex::POS_Y)) << ","
        << s(static_cast<int>(RacerDubinsParams::StateIndex::YAW)) << ","
        << s(static_cast<int>(RacerDubinsParams::StateIndex::VEL_X)) << "\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv)
{
  std::string prefix = "racer_dubins_stadium_mppi_rollout_analysis";
  for (int a = 1; a < argc; ++a)
  {
    if (argv[a][0] != '-')
    {
      prefix = argv[a];
    }
  }

  const mppi::path::Path2D path = mppi::path::Path2D::stadium(kStraightLength, kTurnRadius, kSamplesPerArc);
  mppi::rollout_csv::writeCenterline(path, prefix);

  mppi::path::PathReferenceGenerator ref_gen(kDt);
  ref_gen.setSpeedCap(kVMax);
  ref_gen.setTargetSpeed(kTargetSpeed);

  DYN model;
  RacerDubinsParams dyn;
  dyn.wheel_base = 0.3f;
  model.setParams(dyn);
  std::array<float2, DYN::CONTROL_DIM> u_rng{};
  u_rng[static_cast<int>(RacerDubinsParams::ControlIndex::THROTTLE_BRAKE)] = { -1.0f, 1.0f };
  u_rng[static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD)] = { -1.0f, 1.0f };
  model.setControlRanges(u_rng);

  COST cost;
  cost.GPUSetup();

  RacerCostParams<kRefHorizon> cost_params;
  cost_params.desired_speed = kTargetSpeed;
  cost_params.wheel_base = dyn.wheel_base;
  cost_params.steer_angle_scale = dyn.steer_angle_scale;
  cost.setParams(cost_params);

  SAMPLER::SAMPLING_PARAMS_T sp{};
  sp.std_dev[static_cast<int>(RacerDubinsParams::ControlIndex::THROTTLE_BRAKE)] = 0.2F;
  sp.std_dev[static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD)] = 0.3F;
  sp.sum_strides = std::max(32, (kNumRollouts + 1023) / 1024);
  SAMPLER sampler(sp);

  FB feedback(&model, kDt);
  Mppi::control_trajectory u_nom = Mppi::control_trajectory::Zero();
  Mppi controller(&model, &cost, &feedback, &sampler, kDt, 1, kLambda, 0.0F, kMppiHorizon, u_nom);
  {
    auto cp = controller.getParams();
    cp.dynamics_rollout_dim_ = dim3(32, 2, 1);
    cp.cost_rollout_dim_ = dim3(32, 2, 1);
    cp.seed_ = 42U;
    controller.setParams(cp);
  }
  model.GPUSetup();

  DYN::state_array x = model.getZeroState();
  const mppi::path::Pose2D p0 = path.poseAt(kInitArcLength);
  x(static_cast<int>(RacerDubinsParams::StateIndex::POS_X)) = p0.x;
  x(static_cast<int>(RacerDubinsParams::StateIndex::POS_Y)) = p0.y;
  x(static_cast<int>(RacerDubinsParams::StateIndex::YAW)) = p0.yaw;
  x(static_cast<int>(RacerDubinsParams::StateIndex::VEL_X)) = kTargetSpeed;

  const std::vector<mppi::path::PathReferenceSample> ref = ref_gen.generate(
      path, kInitArcLength, kRefHorizon, x(static_cast<int>(RacerDubinsParams::StateIndex::POS_X)),
      x(static_cast<int>(RacerDubinsParams::StateIndex::POS_Y)),
      x(static_cast<int>(RacerDubinsParams::StateIndex::YAW)),
      x(static_cast<int>(RacerDubinsParams::StateIndex::VEL_X)));
  mppi::cost::fillRacerCostFromPathReference<kRefHorizon>(cost, ref);
  controller.updateImportanceSampler(u_nom);

  std::cout << "Racer Dubins stadium MPPI rollout analysis  straight=" << kStraightLength << " m  R=" << kTurnRadius
            << " m  init_s=" << kInitArcLength << " m  v=" << kTargetSpeed << " m/s  rollouts=" << kNumRollouts
            << "  horizon=" << kMppiHorizon << "  lambda=" << kLambda << "\n";

  controller.computeControl(x, 1);
  cudaStreamSynchronize(controller.stream_);

  const auto& weights_eig = controller.getSampledCostSeq();
  const float baseline = static_cast<float>(controller.getBaselineCost());
  const float normalizer = static_cast<float>(controller.getNormalizerCost());

  const int num_logged = kNumRollouts;
  std::vector<float> raw_costs(num_logged, 0.0F);
  std::vector<float> unnormalized_importance(num_logged, 0.0F);
  std::vector<float> normalized_weights(num_logged, 0.0F);
  for (int i = 0; i < num_logged; ++i)
  {
    const float w = weights_eig(i);
    unnormalized_importance[i] = w;
    normalized_weights[i] = (normalizer > 0.0F) ? w / normalizer : 0.0F;
    raw_costs[i] = (w > 0.0F) ? (baseline - kLambda * std::log(w)) : (baseline + 1.0e30F);
  }

  const int H = kMppiHorizon;
  std::vector<float> host_controls(static_cast<size_t>(num_logged) * H * DYN::CONTROL_DIM);
  {
    float* device_controls = sampler.getControlSample(0, 0, 0);
    HANDLE_ERROR(cudaMemcpy(host_controls.data(), device_controls, host_controls.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
  }

  std::vector<std::vector<DYN::state_array>> rollout_states(static_cast<size_t>(num_logged));
  {
    DYN::state_array x_local;
    DYN::state_array x_next = model.getZeroState();
    DYN::state_array xdot = model.getZeroState();
    DYN::output_array y_t = DYN::output_array::Zero();
    DYN::control_array u_local = DYN::control_array::Zero();
    for (int i = 0; i < num_logged; ++i)
    {
      rollout_states[static_cast<size_t>(i)].resize(static_cast<size_t>(H + 1));
      x_local = x;
      rollout_states[static_cast<size_t>(i)][0] = x_local;
      for (int t = 0; t < H; ++t)
      {
        const float* src = host_controls.data() + (i * H + t) * DYN::CONTROL_DIM;
        for (int d = 0; d < DYN::CONTROL_DIM; ++d)
        {
          u_local(d) = src[d];
        }
        model.enforceConstraints(x_local, u_local);
        model.step(x_local, x_next, xdot, u_local, y_t, static_cast<float>(t), kDt);
        rollout_states[static_cast<size_t>(i)][static_cast<size_t>(t + 1)] = x_next;
        x_local = x_next;
      }
    }
  }

  const Mppi::control_trajectory u_opt = controller.getControlSeq();

  mppi::rollout_csv::writeMeta<DYN>(prefix + "_meta.csv", x, kDt, kLambda, kMppiHorizon, kNumRollouts, num_logged,
                                    baseline, normalizer);
  mppi::rollout_csv::writeCosts(prefix + "_costs.csv", raw_costs, unnormalized_importance, normalized_weights);
  writeRacerCombinedTrajectory(model, x, u_opt, prefix + "_combined.csv", kDt);
  writeRacerRolloutTrajectories(prefix + "_rollouts_xy.csv", rollout_states);

  const auto min_it = std::min_element(raw_costs.begin(), raw_costs.end());
  const auto max_it = std::max_element(raw_costs.begin(), raw_costs.end());
  const int best_idx = static_cast<int>(std::distance(raw_costs.begin(), min_it));
  float sum_w_sq = 0.0F;
  for (int i = 0; i < num_logged; ++i)
  {
    sum_w_sq += normalized_weights[i] * normalized_weights[i];
  }
  const float ess = (sum_w_sq > 0.0F) ? (1.0F / sum_w_sq) : 0.0F;

  std::cout << "One MPPI iteration done.\n";
  std::cout << "  baseline=" << baseline << "  normalizer=" << normalizer << "  ESS=" << ess << "/" << num_logged
            << " (" << (100.0F * ess / static_cast<float>(num_logged)) << "%)\n";
  std::cout << "  raw_cost spread [" << *min_it << ", " << *max_it << "]  delta=" << (*max_it - *min_it) << "\n";
  std::cout << "  best rollout index=" << best_idx << "  raw_cost=" << raw_costs[best_idx]
            << "  weight=" << normalized_weights[best_idx] << "\n";
  if (ess > 0.9F * num_logged)
  {
    std::cout << "  warning: ESS ~ N -> weights nearly uniform (lambda=" << kLambda
              << " may be too large relative to cost spread).\n";
  }
  if ((*max_it - *min_it) < 100.0F)
  {
    std::cout << "  warning: raw cost spread < 100 -> rollouts look similar in cost (check RacerCost track term).\n";
  }
  std::cout << "Wrote " << prefix << "_meta.csv, _costs.csv, _combined.csv, _rollouts_xy.csv\n";
  std::cout << "Plot: python3 examples/plot_mppi_rollout_analysis.py " << prefix << "\n";
  cost.freeCudaMem();
  return 0;
}
