/**
 * RacerDubins + MPPI: straight-line reference, stop at goal (see filename).
 * Time-indexed output reference (temporal path tracking).
 *
 * This is meant to mirror the *role* of a receding-horizon path tracker such as
 * acados `PathTrackingMPCTemporal` + Autoware's temporal MPT: a reference
 * (x, y, yaw, speed) along time and MPPI optimizing throttle/steer. It is not
 * a drop-in for Autoware; integrate this pattern behind your own ROS 2 / Trajectory
 * message adapter.
 *
 * Visualization: run the binary (optional path to CSV), then
 *   python3 examples/plot_racer_dubins_temporal_mppi.py <log.csv>
 * Produces PNGs: <stem>_viz.png (path + states + inputs) and <stem>_viz_baseline.png.
 * Requires: Python 3 with numpy and matplotlib.
 * Prints: wall time for the control+simulate loop (s and ms/step) to stdout.
 */
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>
#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
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
// Match Autoware-style temporal discretization (see path_tracking_mpc_temporal.py: e.g. N=80, Tf=8 -> dt=0.1)
constexpr int kMppiHorizon = 50;       // receding-horizon steps in the controller
constexpr int kSimSteps = 2000;
// Precomputed reference rows: must cover the whole sim (each row = one dt step in reference time) + headroom
constexpr int kRefHorizon = kSimSteps + 256;
constexpr float kDt = 0.1F;
constexpr int kNumRollouts = 1024*32;

// Path / speed profile (kVRef, kGoalX) must be **achievable** by the longitudinal ODE, or the controller will
// “fight” the model (never reaching 3 m/s) and do odd lateral motion. Default RacerDubins c_0, c_t[0], c_v[0] give
// v_ss ≈ (c_t[0]+c_0)/c_v[0] ≈ 1.6 m/s at u=+1 — we override in applyDemoRacerDubinsParams() below.
constexpr float kVRef = 3.0F;
constexpr float kGoalX = 120.0F;  // [m]
// Linear ramp of ref v from kVRef → 0 over T_taper [s] before the stop line (smofer than a step; distance = ½ kVRef T)
constexpr float kDecelTaperS = 2.5F;

// Output channel indices (see RacerDubinsParams::OutputIndex in racer_dubins.cuh)
constexpr int o_pos_x = static_cast<int>(RacerDubinsParams::OutputIndex::BASELINK_POS_I_X);
constexpr int o_pos_y = static_cast<int>(RacerDubinsParams::OutputIndex::BASELINK_POS_I_Y);
constexpr int o_yaw = static_cast<int>(RacerDubinsParams::OutputIndex::YAW);
constexpr int o_v = static_cast<int>(RacerDubinsParams::OutputIndex::TOTAL_VELOCITY);

// Roughly similar in spirit to diag Q / R in path_tracking_mpc_temporal.py (tuned for MPPI, not 1:1)
void fillTrackingWeights(QuadraticCostTrajectoryParams<RacerDubins, kRefHorizon>& p)
{
  for (int i = 0; i < RacerDubins::OUTPUT_DIM; ++i)
  {
    p.s_coeffs[i] = 0.0F;
  }
  p.s_coeffs[o_pos_x] = 1.0F;
  p.s_coeffs[o_pos_y] = 1.0F;
  p.s_coeffs[o_yaw] = 0.1F;   // heading; helps against orbiting
  p.s_coeffs[o_v] = 1.5F;  // must dominate when v_ref=0, else throttle stays high

  // Stiffer on throttle so the solver is willing to release gas / use brake when v_ref decays
  p.control_cost_coeff[static_cast<int>(RacerDubinsParams::ControlIndex::THROTTLE_BRAKE)] = 0.12F;
  p.control_cost_coeff[static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD)] = 0.2F;
}

/** Set engine/drag so that v_ss ≈ kVRef at u_throttle=+1, v_ss=0 at u=0 in the unbraked ODE. */
void applyDemoRacerDubinsParams(RacerDubinsParams& p)
{
  p.c_0 = 0.0F;  // default 4.9 makes v_ss ≈1.6 m/s at u=+1; remove that offset for this example
  p.c_v[0] = 1.0F;
  p.c_t[0] = p.c_v[0] * kVRef;  // 3.0 N·s/kg style balance → v_ss = c_t[0]*u / c_v[0] = kVRef at u=1
  p.c_t[1] = p.c_t[0];
  p.c_t[2] = p.c_t[0];
  p.c_v[1] = p.c_v[0];
  p.c_v[2] = p.c_v[0];
}

void fillReferenceTrajectory(QuadraticCostTrajectoryParams<RacerDubins, kRefHorizon>& p)
{
  // Trapezoid in v_ref: constant kVRef, then linear ramp 0..T_taper, then hold (x=goal, v=0).
  // Distance over ramp = ½ kVRef * T_taper. Cruise ends at time T_cruise s at position kVRef * T_cruise.
  const float T_taper = kDecelTaperS;
  const float T_cruise = std::max(0.0F, (kGoalX - 0.5F * kVRef * T_taper) / kVRef);
  const float T_hold = T_cruise + T_taper;
  for (int t = 0; t < kRefHorizon; ++t)
  {
    for (int j = 0; j < RacerDubins::OUTPUT_DIM; ++j)
    {
      p.s_goal[t * RacerDubins::OUTPUT_DIM + j] = 0.0F;
    }
    const float time_s = t * kDt;
    float x_ref;
    float v_ref;
    if (time_s < T_cruise)
    {
      v_ref = kVRef;
      x_ref = kVRef * time_s;
    }
    else if (time_s < T_hold)
    {
      const float tau = time_s - T_cruise;  // [0, T_taper)
      v_ref = kVRef * (1.0F - tau / T_taper);
      // ∫_0^tau kVRef(1 - s/T_taper) ds = kVRef * (tau - tau^2 / (2*T_taper))
      const float x_c_end = kVRef * T_cruise;
      x_ref = x_c_end + kVRef * (tau - 0.5F * tau * tau / T_taper);
    }
    else
    {
      v_ref = 0.0F;
      x_ref = kGoalX;
    }
    p.s_goal[t * RacerDubins::OUTPUT_DIM + o_pos_x] = x_ref;
    p.s_goal[t * RacerDubins::OUTPUT_DIM + o_v] = v_ref;
    p.s_goal[t * RacerDubins::OUTPUT_DIM + o_pos_y] = 0.0F;
    p.s_goal[t * RacerDubins::OUTPUT_DIM + o_yaw] = 0.0F;
  }
}
}  // namespace

using DYN = RacerDubins;
using FB = DDPFeedback<DYN, kMppiHorizon>;
using COST = QuadraticCostTrajectory<DYN, kRefHorizon>;
using SAMPLER = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
using Mppi = VanillaMPPIController<DYN, COST, FB, kMppiHorizon, kNumRollouts, SAMPLER>;

void printUsage(const char* prog)
{
  std::cout << "Usage: " << prog << " [log.csv]\n"
            << "  Simulates RacerDubins + MPPI, writes a CSV, then plot with:\n"
            << "  python3 examples/plot_racer_dubins_temporal_mppi.py <log.csv>\n"
            << "  (default log path: racer_dubins_mppi_log.csv in current working directory)\n";
}

int main(int argc, char** argv)
{
  std::string log_path = "racer_dubins_mppi_log.csv";
  for (int a = 1; a < argc; ++a)
  {
    const std::string arg = argv[a];
    if (arg == "-h" || arg == "--help")
    {
      printUsage(argv[0]);
      return 0;
    }
    if (arg[0] != '-')
    {
      log_path = arg;
    }
  }

  DYN model;
  RacerDubinsParams dyn_params;
  applyDemoRacerDubinsParams(dyn_params);
  model.setParams(dyn_params);
  // Reasonable actuation limits for a planning demo (tune for your platform)
  std::array<float2, DYN::CONTROL_DIM> u_rng{};
  u_rng[static_cast<int>(RacerDubinsParams::ControlIndex::THROTTLE_BRAKE)] = { -0.2F, 1.0F };
  u_rng[static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD)] = { -0.3F, 0.3F };
  model.setControlRanges(u_rng);

  COST cost;
  QuadraticCostTrajectoryParams<RacerDubins, kRefHorizon> cost_params;
  fillReferenceTrajectory(cost_params);
  fillTrackingWeights(cost_params);
  cost.setParams(cost_params);

  SAMPLER::SAMPLING_PARAMS_T sampler_params;
  for (int c = 0; c < DYN::CONTROL_DIM; ++c)
  {
    sampler_params.std_dev[c] = 0.16F;
    sampler_params.control_cost_coeff[c] = cost_params.control_cost_coeff[c];
  }
  SAMPLER sampler(sampler_params);

  FB feedback(&model, kDt);
  Mppi controller(&model, &cost, &feedback, &sampler, kDt, 1, 0.28F, 0.0F);
  {
    auto p = controller.getParams();
    p.dynamics_rollout_dim_ = dim3(32, 2, 1);
    p.cost_rollout_dim_ = dim3(32, 2, 1);
    controller.setParams(p);
  }

  model.GPUSetup();
  cost.GPUSetup();

  // State: VEL_X, YAW, POS_X, POS_Y, STEER_ANGLE, BRAKE_STATE, STEER_ANGLE_RATE — start near origin, slightly offset in y
  DYN::state_array x = model.getZeroState();
  x(0) = 1.5F;  // VEL_X
  x(1) = 0.0F;  // YAW
  x(2) = 0.0F;  // POS_X
  x(3) = 0.15F;  // POS_Y
  x(4) = 0.0F;  // STEER_ANGLE
  x(5) = 0.0F;  // BRAKE_STATE
  x(6) = 0.0F;  // STEER_ANGLE_RATE

  DYN::state_array x_next = model.getZeroState();
  DYN::state_array xdot = model.getZeroState();
  DYN::output_array y = DYN::output_array::Zero();  // rollouts use full outputs on device; host step only fills a prefix

  std::ofstream log(log_path.c_str());
  if (!log)
  {
    std::cerr << "Could not open log file: " << log_path << std::endl;
    return 1;
  }
  log << "t,pos_x,pos_y,yaw,vel_x,steer_angle,brake_state,u_throttle,u_steer,ref_x,ref_y,ref_yaw,ref_v,baseline\n";
  log << std::scientific;

  std::cout << "RacerDubins + temporal MPPI (ref: straight " << kVRef << " m/s, dt=" << kDt << ")\n";
  std::cout << "Logging to " << log_path << " (for matplotlib plots)\n";

  // Initial sample at t=0 (before any integration)
  {
    const int r0 = 0;
    const int ri0 = r0 * RacerDubins::OUTPUT_DIM;
    log << 0.0F << "," << x(2) << "," << x(3) << "," << x(1) << "," << x(0) << "," << x(4) << "," << x(5) << ",0,0,"
        << cost_params.s_goal[ri0 + o_pos_x] << "," << cost_params.s_goal[ri0 + o_pos_y] << ","
        << cost_params.s_goal[ri0 + o_yaw] << "," << cost_params.s_goal[ri0 + o_v] << ",0"
        << "\n";
  }

  const auto time_loop_start = std::chrono::steady_clock::now();
  for (int k = 0; k < kSimSteps; ++k)
  {
    cost_params.setCurrentTime(k);
    cost.setParams(cost_params);

    controller.computeControl(x, 1);
    const float baseline = static_cast<float>(controller.getBaselineCost());

    DYN::control_array u = controller.getControlSeq().col(0);
    model.enforceConstraints(x, u);
    model.step(x, x_next, xdot, u, y, k, kDt);
    const float t_end = (k + 1) * kDt;
    // Post-step state x_next; reference at the same wall time index (k+1) for comparison plots
    const int rrow = std::min(k + 1, kRefHorizon - 1);
    const int ri = rrow * RacerDubins::OUTPUT_DIM;
    const float ref_x = cost_params.s_goal[ri + o_pos_x];
    const float ref_y = cost_params.s_goal[ri + o_pos_y];
    const float ref_yaw = cost_params.s_goal[ri + o_yaw];
    const float ref_v = cost_params.s_goal[ri + o_v];
    // State: VEL_X(0), YAW(1), POS_X(2), POS_Y(3), STEER_ANGLE(4), BRAKE_STATE(5)
    log << t_end << "," << x_next(2) << "," << x_next(3) << "," << x_next(1) << "," << x_next(0) << "," << x_next(4)
        << "," << x_next(5) << "," << u(0) << "," << u(1) << "," << ref_x << "," << ref_y << "," << ref_yaw << ","
        << ref_v << "," << baseline << "\n";
    x = x_next;

    if (k % 20 == 0)
    {
      std::cout << "t=" << std::fixed << std::setprecision(2) << t_end << "  pos(" << x(2) << ", " << x(3) << ")"
                << " yaw=" << x(1) << " v_x=" << x(0) << "  u=(" << u(0) << ", " << u(1) << ")  x_ref=" << ref_x
                << std::endl;
    }
    controller.slideControlSequence(1);
  }
  const auto time_loop_end = std::chrono::steady_clock::now();
  const double elapsed_s =
      std::chrono::duration_cast<std::chrono::duration<double>>(time_loop_end - time_loop_start).count();
  const double ms_per_step = 1000.0 * elapsed_s / static_cast<double>(kSimSteps);

  log.close();
  std::cout << "Wrote " << log_path << "\n";
  std::cout << "Elapsed (MPPI + simulate, " << kSimSteps << " steps): " << std::fixed << std::setprecision(3)
            << elapsed_s << " s  (" << std::setprecision(2) << ms_per_step << " ms/step)\n";
  std::cout << "Plot: python3 examples/plot_racer_dubins_temporal_mppi.py " << log_path << std::endl;
  return 0;
}
