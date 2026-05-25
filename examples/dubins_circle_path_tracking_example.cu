/**
 * Dubins bicycle + MPPI tracking on a closed circular path.
 *
 * Build: cmake --build build --target dubins_circle_path_tracking_example
 * Run:   ./build/examples/dubins_circle_path_tracking_example [log.csv]
 * Plot:  python3 examples/plot_racer_dubins_temporal_mppi.py dubins_circle_path_tracking_log.csv
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
constexpr int kMppiHorizon = 50;
constexpr int kRefHorizon = kMppiHorizon + 8;
constexpr float kDt = 0.1F;
constexpr int kNumRollouts = 32 * 1024;
constexpr float kVMax = 3.0F;

constexpr float kCircleRadius = 20.0F;
constexpr float kCircleCenterX = 0.0F;
constexpr float kCircleCenterY = 0.0F;
constexpr float kCircleTheta0 = 0.0F;
/** Plot/export polyline density only; tracking uses exact circle geometry. */
constexpr int kCirclePlotSamples = 512;

constexpr float kInitLateralOffset = 0.1F;
constexpr float kInitSpeed = 1.5F;
constexpr float kSimLaps = 2.5F;

using DYN = DubinsBicycle;
using COST = PathTrackingCost<kRefHorizon>;
using FB = DDPFeedback<DYN, kMppiHorizon>;
using SAMPLER = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
using Mppi = VanillaMPPIController<DYN, COST, FB, kMppiHorizon, kNumRollouts, SAMPLER>;

void clipControl(const DubinsBicycleParams& p, DYN::control_array& u)
{
  u(static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)) =
      std::max(p.min_accel, std::min(u(static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)), p.max_accel));
  u(static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)) =
      std::max(-p.max_steer_angle, std::min(u(static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)), p.max_steer_angle));
}

int simStepsForLaps(const mppi::path::Path2D& path, const float laps)
{
  const float lap_time = path.length() / kVMax;
  return static_cast<int>(std::ceil(laps * lap_time / kDt));
}
}  // namespace

int main(int argc, char** argv)
{
  std::string log_path = "dubins_circle_path_tracking_log.csv";
  for (int a = 1; a < argc; ++a)
  {
    if (argv[a][0] != '-')
    {
      log_path = argv[a];
    }
  }

  const mppi::path::Path2D path =
      mppi::path::Path2D::circle(kCircleCenterX, kCircleCenterY, kCircleRadius, kCircleTheta0, kCirclePlotSamples);
  const int kSimSteps = simStepsForLaps(path, kSimLaps);
  mppi::rollout_csv::writeCenterlineForLog(path, log_path);

  mppi::path::PathReferenceGenerator ref_gen(kDt);
  ref_gen.setSpeedCap(kVMax);

  DYN model;
  DubinsBicycleParams dyn;
  model.setParams(dyn);
  std::array<float2, DYN::CONTROL_DIM> u_rng{};
  u_rng[static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)] = { dyn.min_accel, dyn.max_accel };
  u_rng[static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)] = { -dyn.max_steer_angle, dyn.max_steer_angle };
  model.setControlRanges(u_rng);

  COST cost;
  PathTrackingCostParams<kRefHorizon> cost_params;
  mppi::path::fillPathTrackingCostWeights<kRefHorizon>(cost_params, 10.0F, 1.0F, 5.0F, 1.0F, 0.05F);
  mppi::path::fillPathTrackingBicycleGeometry<kRefHorizon>(cost_params, dyn);
  cost.setParams(cost_params);

  SAMPLER::SAMPLING_PARAMS_T sp{};
  sp.std_dev[static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)] = 0.15F;
  sp.std_dev[static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)] = 0.04F;
  sp.control_cost_coeff[0] = cost_params.control_cost_coeff[0];
  sp.control_cost_coeff[1] = cost_params.control_cost_coeff[1];
  sp.sum_strides = std::max(32, (kNumRollouts + 1023) / 1024);
  SAMPLER sampler(sp);

  FB feedback(&model, kDt);
  Mppi::control_trajectory u_nom = Mppi::control_trajectory::Zero();
  // lambda is the MPPI temperature in w_i = exp(-(c_i - baseline) / lambda).
  // With these cost weights the per-step running cost runs at ~150 and per-rollout totals are
  // ~8000, so the cost spread between baseline and "good" rollouts is on the order of 100s.
  // lambda << that → argmin behavior (only the single best rollout contributes); lambda >> that →
  // weights collapse to uniform and the optimal control regresses to the noise mean. 100 lands
  // in the healthy "soft mixture" range. See dubins_circle_mppi_rollout_analysis_example.cu for
  // the diagnostic methodology.
  constexpr float kMppiLambda = 100.0F;
  Mppi controller(&model, &cost, &feedback, &sampler, kDt, 1, kMppiLambda, 0.0F, kMppiHorizon, u_nom);
  {
    auto cp = controller.getParams();
    cp.dynamics_rollout_dim_ = dim3(32, 2, 1);
    cp.cost_rollout_dim_ = dim3(32, 2, 1);
    cp.seed_ = 42U;
    controller.setParams(cp);
  }
  // PathTrackingCost has a large device params blob; combined rollout kernel misaligns shared memory.
  controller.setKernelChoice(kernelType::USE_SPLIT_KERNELS);
  model.GPUSetup();
  cost.GPUSetup();

  const std::vector<mppi::path::PathReferenceSample> ref_init = ref_gen.generate(path, 0.0F, kRefHorizon);
  mppi::path::fillCostFromPathReference<kRefHorizon>(cost_params, ref_init, &path, &dyn);
  cost.setParams(cost_params);

  DYN::state_array x = model.getZeroState();
  const mppi::path::Pose2D p0 = path.poseAt(0.0F);
  float init_x = p0.x;
  float init_y = p0.y;
  mppi::path::applyInitialLateralOffset(path, 0.0F, kInitLateralOffset, init_x, init_y);
  x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_X)) = init_x;
  x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_Y)) = init_y;
  x(static_cast<int>(DubinsBicycleParams::StateIndex::YAW)) = p0.yaw;
  x(static_cast<int>(DubinsBicycleParams::StateIndex::VEL_X)) = kInitSpeed;

  mppi::path::fillNominalControlFromReference(u_nom, x, ref_init, dyn, kDt, &path);
  controller.updateImportanceSampler(u_nom);

  DYN::state_array x_next = model.getZeroState();
  DYN::state_array xdot = model.getZeroState();
  DYN::output_array y = DYN::output_array::Zero();

  float s_prog = 0.0F;
  float max_lat = 0.0F;

  std::ofstream log(log_path.c_str());
  if (!log)
  {
    std::cerr << "Could not open log: " << log_path << "\n";
    return 1;
  }
  log << "t,pos_x,pos_y,yaw,vel_x,steer_angle,brake_state,u_accel,u_steer,nom_u_accel,nom_u_steer,ref_x,ref_y,ref_yaw,"
         "ref_v,arc_s,lat_err,baseline\n";
  log << std::scientific;

  std::cout << "Dubins circle path tracking  R=" << kCircleRadius << " m  path_length=" << path.length()
            << " m  sim_steps=" << kSimSteps << " (~" << kSimLaps << " laps)\n";
  std::cout << "Logging to " << log_path << "\n";

  const auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < kSimSteps; ++k)
  {
    const mppi::path::PathProjection proj = mppi::path::projectPoseOntoPath(
        path, x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_X)),
        x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_Y)), s_prog);
    s_prog = proj.arc_length_s;
    max_lat = std::max(max_lat, std::fabs(proj.signed_lateral_error));

    const std::vector<mppi::path::PathReferenceSample> ref = ref_gen.generate(path, s_prog, kRefHorizon);
    mppi::path::fillCostFromPathReference<kRefHorizon>(cost_params, ref, &path, &dyn);
    cost.setParams(cost_params);

    mppi::path::fillNominalControlFromReference(u_nom, x, ref, dyn, kDt, &path);
    controller.updateImportanceSampler(u_nom);
    controller.computeControl(x, 1);
    const DYN::control_array u_apply = controller.getControlSeq().col(0);
    DYN::control_array u = u_apply;
    clipControl(dyn, u);
    const float baseline = static_cast<float>(controller.getBaselineCost());
    model.enforceConstraints(x, u);
    model.step(x, x_next, xdot, u, y, static_cast<float>(k), kDt);

    const mppi::path::PathReferenceSample& r0 = ref.front();
    const float t_end = static_cast<float>(k + 1) * kDt;
    log << t_end << "," << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::POS_X)) << ","
        << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::POS_Y)) << ","
        << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::YAW)) << ","
        << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::VEL_X)) << ","
        << x_next(static_cast<int>(DubinsBicycleParams::StateIndex::STEER_ANGLE)) << ",0,"
        << u(static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)) << ","
        << u(static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)) << ","
        << u_apply(static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)) << ","
        << u_apply(static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)) << "," << r0.x << "," << r0.y << ","
        << r0.yaw << "," << r0.v << "," << s_prog << "," << proj.signed_lateral_error << "," << baseline << "\n";

    x = x_next;
    controller.slideControlSequence(1);

    if (k % 50 == 0)
    {
      std::cout << "t=" << t_end << "  s=" << s_prog << "  lat=" << proj.signed_lateral_error
                << "  v=" << x(static_cast<int>(DubinsBicycleParams::StateIndex::VEL_X)) << "\n";
    }
  }

  log.close();
  const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "Done. max |lateral|=" << max_lat << " m  elapsed=" << elapsed << " s " << " time per step=" << elapsed / kSimSteps << " s\n";
  std::cout << "Plot: python3 examples/plot_racer_dubins_temporal_mppi.py " << log_path << "\n";
  return 0;
}
