/**
 * Dubins bicycle + MPPI tracking on a closed stadium (racetrack) path.
 *
 * Build: cmake --build build --target dubins_stadium_path_tracking_example
 * Run:   ./build/examples/dubins_stadium_path_tracking_example [--straight 40] [--radius 10] [log.csv]
 * Plot:  python3 examples/plot_racer_dubins_temporal_mppi.py dubins_stadium_path_tracking_log.csv
 */
#include "mppi_rollout_csv.hpp"

#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/cost_functions/path_tracking/path_tracking_cost.cuh>
#include <mppi/dynamics/dubins_bicycle/dubins_bicycle.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/path/path_projection.hpp>
#include <mppi/path/path_reference_generator.hpp>
#include <mppi/path/path_tracking_bridge.hpp>
#include <mppi/path/path2d.hpp>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace
{
/** MPPI optimization (int, not float). */
const int kMppiNumIters = 1;
// lambda is the MPPI temperature in w_i = exp(-(c_i - baseline) / lambda). It's much higher
// here than in the circle/straight examples (λ=100) because the cost weights are heavier
// (lat_accel=5, lat_jerk=20, steer_dot=50 add comfort terms not present in the simpler
// examples), so per-rollout costs and their spread are 10–100x larger. See
// dubins_circle_mppi_rollout_analysis_example.cu for the diagnostic methodology (cost spread →
// effective sample size → temperature). Override at runtime with --lambda.
const float kMppiLambdaDefault = 100.0F;
const float kMppiAlpha = 0.0F;
const float kDt = 0.1F;
/** Softer feedforward (default bridge: k_lat=0.5, k_heading=1.5). */
const float kNomLatSteerGain = 0.15F;
const float kNomHeadingSteerGain = 0.1F;
const int kMppiHorizon = 50;
const int kRefHorizon = kMppiHorizon + 8;
// Same as dubins_stadium_mppi_rollout_analysis_example.cu so iteration 1 of this closed-loop run
// uses an identical MPPI instance (same number of rollouts -> same noise draw given the same
// seed). Beyond iteration 1 the closed-loop receding horizon and slid control take over, but the
// first call's rollouts/costs/weights match the analysis exactly.
const int kNumRollouts = 32 * 1024;
const float kVMax = 3.0F;

const float kStraightLength = 40.0F;
const float kTurnRadius = 10.0F;
const int kSamplesPerArc = 48;

// Place the vehicle 2 m before the first right-turn corner entry (s = straight_length is the
// bend) at the steady target speed, so the very first MPPI horizon spans the curvature step
// instead of 40 m of trivial straight-line tracking. Matches the analysis example's setup.
const float kInitArcOffsetBeforeBend = 2.0F;
const float kInitLateralOffset = 0.1F;
const float kInitSpeed = 2.5F;
const float kSimLaps = 2.5F;

using DYN = DubinsBicycle;
using COST = PathTrackingCost<kRefHorizon>;
using FB = DDPFeedback<DYN, kMppiHorizon>;
using SAMPLER = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
using Mppi = VanillaMPPIController<DYN, COST, FB, kMppiHorizon, kNumRollouts, SAMPLER>;

int simStepsForLaps(const mppi::path::Path2D& path, const float laps)
{
  const float lap_time = path.length() / kVMax;
  return static_cast<int>(std::ceil(laps * lap_time / kDt));
}
}  // namespace

int main(int argc, char** argv)
{
  std::string log_path = "dubins_stadium_path_tracking_log.csv";
  float straight_length = kStraightLength;
  float turn_radius = kTurnRadius;
  int samples_per_arc = kSamplesPerArc;
  // Tuning knobs (override on CLI for parameter sweeps without rebuilding).
  float mppi_lambda = kMppiLambdaDefault;
  float w_steer_dot = 50.0F;
  float w_lat_jerk = 20.0F;
  float w_pos = 0.5F;
  float w_steer = 0.5F;
  // --single-iter: run computeControl exactly once and apply the full kMppiHorizon-step optimal
  // control sequence open-loop (no replanning, no slide). Output is then byte-identical to
  // dubins_stadium_mppi_rollout_analysis_example for direct apples-to-apples comparison; the
  // ringing visible in the default closed-loop run does not appear here because there is no
  // receding-horizon coupling to the bicycle's first-order steer dynamics.
  bool single_iter = false;

  for (int a = 1; a < argc; ++a)
  {
    const std::string arg = argv[a];
    if (arg == "--straight" && a + 1 < argc)
    {
      straight_length = std::stof(argv[++a]);
      continue;
    }
    if (arg == "--radius" && a + 1 < argc)
    {
      turn_radius = std::stof(argv[++a]);
      continue;
    }
    if (arg == "--arc-samples" && a + 1 < argc)
    {
      samples_per_arc = std::stoi(argv[++a]);
      continue;
    }
    if (arg == "--lambda" && a + 1 < argc)
    {
      mppi_lambda = std::stof(argv[++a]);
      continue;
    }
    if (arg == "--w-steer-dot" && a + 1 < argc)
    {
      w_steer_dot = std::stof(argv[++a]);
      continue;
    }
    if (arg == "--w-lat-jerk" && a + 1 < argc)
    {
      w_lat_jerk = std::stof(argv[++a]);
      continue;
    }
    if (arg == "--w-pos" && a + 1 < argc)
    {
      w_pos = std::stof(argv[++a]);
      continue;
    }
    if (arg == "--w-steer" && a + 1 < argc)
    {
      w_steer = std::stof(argv[++a]);
      continue;
    }
    if (arg == "--single-iter")
    {
      single_iter = true;
      continue;
    }
    if (arg == "--help" || arg == "-h")
    {
      std::cout << "Usage: dubins_stadium_path_tracking_example [--straight L] [--radius R] [--arc-samples N] "
                   "[--lambda L] [--w-steer-dot W] [--w-lat-jerk W] [--w-pos W] [--w-steer W] "
                   "[--single-iter] [log.csv]\n";
      return 0;
    }
    if (arg[0] != '-')
    {
      log_path = arg;
    }
  }

  if (straight_length < 0.0F || turn_radius < 1.0E-3F)
  {
    std::cerr << "Stadium straight length must be >= 0 and radius must be positive.\n";
    return 1;
  }

  const mppi::path::Path2D path = mppi::path::Path2D::stadium(straight_length, turn_radius, samples_per_arc);
  const int kSimSteps = simStepsForLaps(path, kSimLaps);
  mppi::rollout_csv::writeCenterlineForLog(path, log_path);

  mppi::path::PathReferenceGenerator ref_gen(kDt);
  ref_gen.setSpeedCap(kVMax);

  DYN model;
  DubinsBicycleParams dyn;
  /** Exploration noise — large steer std_dev causes control chatter. */
  const float noise_std_accel = (dyn.max_accel - dyn.min_accel) / 2.0F;
  const float noise_std_steer = dyn.max_steer_angle / 2.0F;

  model.setParams(dyn);
  std::array<float2, DYN::CONTROL_DIM> u_rng{};
  u_rng[static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)] = { dyn.min_accel, dyn.max_accel };
  u_rng[static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)] = { -dyn.max_steer_angle, dyn.max_steer_angle };
  model.setControlRanges(u_rng);

  COST cost;
  PathTrackingCostParams<kRefHorizon> cost_params;
  // Tracking + comfort + actuation (also copied to Gaussian IS via control_cost_coeff).
  mppi::path::fillPathTrackingCostWeights<kRefHorizon>(cost_params, 2.0F, 1.0F, 5.0F, 5.0F, 20.0F, 50.0F, 0.50F, 0.5F);
  mppi::path::fillPathTrackingBicycleGeometry<kRefHorizon>(cost_params, dyn);
  cost.setParams(cost_params);

  SAMPLER::SAMPLING_PARAMS_T sp{};
  sp.std_dev[static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)] = noise_std_accel;
  sp.std_dev[static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)] = noise_std_steer;
  sp.control_cost_coeff[0] = cost_params.control_cost_coeff[0];
  sp.control_cost_coeff[1] = cost_params.control_cost_coeff[1];
  sp.sum_strides = std::max(32, (kNumRollouts + 1023) / 1024);
  SAMPLER sampler(sp);

  FB feedback(&model, kDt);
  Mppi::control_trajectory u_nom = Mppi::control_trajectory::Zero();
  Mppi controller(&model, &cost, &feedback, &sampler, kDt, kMppiNumIters, mppi_lambda, kMppiAlpha, kMppiHorizon,
                  u_nom);
  {
    auto cp = controller.getParams();
    cp.dynamics_rollout_dim_ = dim3(32, 2, 1);
    cp.cost_rollout_dim_ = dim3(32, 2, 1);
    cp.seed_ = 42U;
    // Default slide_control_scale is 0: extrapolated horizon columns go to zero_control each
    // slideControlSequence(1), so without nominal blend the warm start collapses to zero.
    cp.slide_control_scale_.setOnes();
    controller.setParams(cp);
  }
  controller.setKernelChoice(kernelType::USE_SPLIT_KERNELS);
  model.GPUSetup();
  cost.GPUSetup();

  const float init_arc_length = std::max(0.0F, straight_length - kInitArcOffsetBeforeBend);
  const std::vector<mppi::path::PathReferenceSample> ref_init = ref_gen.generate(path, init_arc_length, kRefHorizon);
  mppi::path::fillCostFromPathReference<kRefHorizon>(cost_params, ref_init, &path, &dyn);
  cost.setParams(cost_params);

  DYN::state_array x = model.getZeroState();
  const mppi::path::Pose2D p0 = path.poseAt(init_arc_length);
  float init_x = p0.x;
  float init_y = p0.y;
  mppi::path::applyInitialLateralOffset(path, init_arc_length, kInitLateralOffset, init_x, init_y);
  x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_X)) = init_x;
  x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_Y)) = init_y;
  x(static_cast<int>(DubinsBicycleParams::StateIndex::YAW)) = p0.yaw;
  x(static_cast<int>(DubinsBicycleParams::StateIndex::VEL_X)) = kInitSpeed;

  // Pure-curvature feedforward seed: with kNomLatSteerGain = kNomHeadingSteerGain = 0, this
  // becomes u_nom_steer = atan(L * kappa(s_lookahead)) — no state feedback. Same setup as
  // dubins_stadium_mppi_rollout_analysis_example, which we've verified produces a clean
  // lag-aware MPPI plan from this seed.
  mppi::path::fillNominalControlFromReference(u_nom, x, ref_init, dyn, kDt, &path, kNomLatSteerGain,
                                              kNomHeadingSteerGain);
  controller.updateImportanceSampler(u_nom);

  DYN::state_array x_next = model.getZeroState();
  DYN::state_array xdot = model.getZeroState();
  DYN::output_array y = DYN::output_array::Zero();

  float s_prog = init_arc_length;
  float max_lat = 0.0F;

  std::ofstream log(log_path.c_str());
  if (!log)
  {
    std::cerr << "Could not open log: " << log_path << "\n";
    return 1;
  }
  log << "t,pos_x,pos_y,yaw,vel_x,steer_angle,brake_state,u_accel,u_steer,nom_u_accel,nom_u_steer,ref_x,ref_y,ref_yaw,"
         "ref_v,arc_s,lat_err,nominal_cost,mppi_applied_cost,baseline\n";
  log << std::scientific;

  // single_iter: skip the closed-loop receding horizon entirely. Run computeControl exactly
  // once from the initial state (with the reference, cost, and u_nom already set up above) and
  // then play out the full kMppiHorizon-step optimal control sequence open-loop. Output is then
  // identical to dubins_stadium_mppi_rollout_analysis_example for the same initial conditions.
  Mppi::control_trajectory u_opt_horizon = Mppi::control_trajectory::Zero();
  std::vector<mppi::path::PathReferenceSample> ref_initial;
  float baseline_initial = 0.0F;
  if (single_iter)
  {
    controller.computeControl(x, 1);
    u_opt_horizon = controller.getControlSeq();
    baseline_initial = static_cast<float>(controller.getBaselineCost());
    ref_initial = ref_gen.generate(path, s_prog, kRefHorizon);
  }

  const int effective_sim_steps = single_iter ? kMppiHorizon : kSimSteps;

  std::cout << "Dubins stadium path tracking  straight=" << straight_length << " m  R=" << turn_radius
            << " m  path_length=" << path.length() << " m  sim_steps=" << effective_sim_steps
            << (single_iter ? " (single-iter, open-loop replay)" : (" (~" + std::to_string(kSimLaps) + " laps)"))
            << "  rollouts=" << kNumRollouts << "  mppi_iters=" << kMppiNumIters
            << "  lambda=" << mppi_lambda << "  w_pos=" << w_pos << "  w_lat_jerk=" << w_lat_jerk
            << "  w_steer_dot=" << w_steer_dot << "  w_steer=" << w_steer << "\n";
  std::cout << "Logging to " << log_path << "\n";

  const auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < effective_sim_steps; ++k)
  {
    const mppi::path::PathProjection proj = mppi::path::projectPoseOntoPath(
        path, x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_X)),
        x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_Y)), s_prog);
    s_prog = proj.arc_length_s;
    max_lat = std::max(max_lat, std::fabs(proj.signed_lateral_error));

    std::vector<mppi::path::PathReferenceSample> ref;
    if (!single_iter)
    {
      ref = ref_gen.generate(path, s_prog, kRefHorizon);
      mppi::path::fillCostFromPathReference<kRefHorizon>(cost_params, ref, &path, &dyn);
      cost.setParams(cost_params);

      // Refresh u_nom each step so the curvature feedforward at the new s_prog is the IS mean.
      // With kNomLatSteerGain = kNomHeadingSteerGain = 0, u_nom is purely atan(L * kappa) — a
      // smooth, state-independent signal that cannot ring (no closed-loop P-feedback). MPPI then
      // refines it via samples + cost. (Pure warm-start without u_nom collapses the steer to
      // ~0 because the slid plan's tail repeats and MPPI is sample-starved at v=2.5 m/s on
      // R=10 m turns; see commit history for the empirical experiment.)
      mppi::path::fillNominalControlFromReference(u_nom, x, ref, dyn, kDt, &path, kNomLatSteerGain,
                                                  kNomHeadingSteerGain);
      controller.updateImportanceSampler(u_nom);
      controller.computeControl(x, 1);
    }

    // Single-iter: replay column k of the iter-1 plan. Closed-loop: latest computeControl's u[0].
    DYN::control_array control = single_iter ? DYN::control_array(u_opt_horizon.col(k))
                                              : DYN::control_array(controller.getControlSeq().col(0));
    model.enforceConstraints(x, control);
    model.step(x, x_next, xdot, control, y, static_cast<float>(k), kDt);

    int crash = 0;
    const int u_nom_col = single_iter ? std::min(k, kMppiHorizon - 1) : 0;
    const DYN::control_array u_nom_step = u_nom.col(u_nom_col);
    const float nominal_cost = cost.computeRunningCost(y, u_nom_step, 0, &crash);
    crash = 0;
    const float mppi_applied_cost = cost.computeRunningCost(y, control, 0, &crash);
    // baseline = minimum total cost among all GPU rollouts (single_iter: cached from iter 1).
    const float baseline = single_iter ? baseline_initial : static_cast<float>(controller.getBaselineCost());

    // ref.front() in closed-loop is the lookahead at the current state. In single_iter we log the
    // iter-1 reference window's k-th sample so the log columns track the trajectory the optimal
    // plan was scoring against at horizon step k.
    const mppi::path::PathReferenceSample& r0 =
        single_iter ? ref_initial[static_cast<size_t>(std::min(k, static_cast<int>(ref_initial.size()) - 1))]
                     : ref.front();
    const float t_end = static_cast<float>(k + 1) * kDt;
    log << t_end << "," << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::POS_X)) << ","
        << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::POS_Y)) << ","
        << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::YAW)) << ","
        << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::VEL_X)) << ","
        << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::STEER_ANGLE)) << ",0,"
        << control(static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)) << ","
        << control(static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)) << ","
        << u_nom_step(static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)) << ","
        << u_nom_step(static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)) << "," << r0.x << "," << r0.y
        << "," << r0.yaw << "," << r0.v << "," << s_prog << "," << proj.signed_lateral_error << "," << nominal_cost
        << "," << mppi_applied_cost << "," << baseline << "\n";
    x = x_next;
    if (!single_iter)
    {
      controller.slideControlSequence(1);
    }

    if (k % 50 == 0)
    {
      std::cout << "t=" << t_end << "  s=" << s_prog << "  lat=" << proj.signed_lateral_error
                << "  v=" << x(static_cast<int>(DubinsBicycleParams::StateIndex::VEL_X)) 
                << "(x, y, yaw, vel, steer) = " 
                << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::POS_X)) << ","
                << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::POS_Y)) << ","
                << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::YAW)) << ","
                << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::VEL_X)) << ","
                << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::STEER_ANGLE)) << ", "
                << "(a, steer) = " 
                << control(static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)) << ","
                << control(static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)) << "," 
                << "baseline = " << baseline << "\n";
    }
  }

  log.close();
  const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "Done. max |lateral|=" << max_lat << " m  elapsed=" << elapsed << " s"  << " time per step=" << elapsed / kSimSteps << " s\n";
  std::cout << "Plot: python3 examples/plot_racer_dubins_temporal_mppi.py " << log_path << "\n";
  return 0;
}
