/**
 * @file first_order_dubins_two_lane_double_park_example.cu
 * @brief Two-lane straight road: ego in the left lane goes around a stopped vehicle by
 *        nudging into the right lane while avoiding faster traffic there from behind.
 *
 * Build: cmake --build build --target first_order_dubins_two_lane_double_park_example
 * Run:   ./build/examples/first_order_dubins_two_lane_double_park_example [seed] [log.csv]
 */

#include "mppi_rollout_csv.hpp"

#include <mppi/cost_functions/dubins/first_order_dubins_bicycle_cost.cuh>
#include <mppi/cost_functions/dubins/first_order_dubins_bicycle_cost_bridge.hpp>
#include <mppi/cost_functions/moving_car_obstacles.hpp>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/dynamics/dubins/first_order_dubins_bicycle.cuh>
#include <mppi/feedback_controllers/zero_feedback.cuh>
#include <mppi/path/path_projection.hpp>
#include <mppi/path/path_reference_generator.hpp>
#include <mppi/path/path2d.hpp>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

#include "path_tracking_viz.hpp"
#include "step_timing.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace
{
  constexpr int kMppiHorizon = 50;
  constexpr int kRefHorizon = kMppiHorizon;
  constexpr float kDt = 0.1F;
  constexpr int kNumRollouts = 32 * 1024;
  constexpr float kTargetSpeed = 3.0F;
  constexpr float kVMax = 5.0F;
  constexpr float kLambda = 1500.0F;

  constexpr float kLeftLaneX = mppi::cost::TwoLaneRoadLayout::kLeftLaneX;
  constexpr float kRightLaneX = mppi::cost::TwoLaneRoadLayout::kRightLaneX;
  constexpr float kLaneHalfWidth = mppi::cost::TwoLaneRoadLayout::kLaneHalfWidth;
  /** Lateral slack from left-lane center so ego may nudge into the right lane. */
  constexpr float kPathBoundary = 2.25F;
  constexpr float kRoadYStart = mppi::cost::TwoLaneRoadLayout::kRoadYStart;
  constexpr float kRoadYEnd = mppi::cost::TwoLaneRoadLayout::kRoadYEnd;
  constexpr float kInitArcLength = 1.5F;

  using DYN = FirstOrderDubinsBicycle;
  using COST = FirstOrderDubinsBicycleCost<kRefHorizon>;
  using FB = ZeroFeedback<DYN, kMppiHorizon>;
  using SAMPLER = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
  using Mppi = VanillaMPPIController<DYN, COST, FB, kMppiHorizon, kNumRollouts, SAMPLER>;

  mppi::path::Path2D makeLeftLanePath()
  {
    return mppi::path::Path2D::straightLine(kLeftLaneX, kRoadYStart, kLeftLaneX, kRoadYEnd, 36);
  }

  int simulationSteps(const mppi::path::Path2D& path)
  {
    const float duration = path.length() / kTargetSpeed + 6.0F;
    return static_cast<int>(std::ceil(duration / kDt));
  }
}  // namespace

int main(int argc, char** argv)
{
  unsigned int seed = 42;
  if (argc > 1)
  {
    seed = std::stoul(argv[1]);
  }
  (void)seed;
  std::cout << "Using random seed: " << seed << std::endl;

  const std::string video_path = "first_order_dubins_two_lane_double_park.mp4";
  std::string log_path = "first_order_dubins_two_lane_double_park_log.csv";
  if (argc > 2)
  {
    log_path = argv[2];
  }

  const mppi::path::Path2D path = makeLeftLanePath();
  mppi::rollout_csv::writeCenterlineForLog(path, log_path);

  std::vector<mppi::cost::MovingCarObstacle> obstacles = mppi::cost::twoLaneDoubleParkAndRearApproach();
  std::cout << "Ego left lane x=" << kLeftLaneX << ", stopped @ (" << obstacles[0].x0 << ", " << obstacles[0].y0
            << "), right-lane traffic @ x=" << obstacles[1].x0 << " y=" << obstacles[1].y0
            << " vy=" << obstacles[1].vy << " m/s\n";

  mppi::path::PathReferenceGenerator ref_gen(kDt);
  ref_gen.setSpeedCap(kVMax);
  ref_gen.setTargetSpeed(kTargetSpeed);
  ref_gen.setMergeHorizonSteps(10);

  const int num_sim_steps = simulationSteps(path);
  float arcLength = kInitArcLength;

  DYN model;
  FirstOrderDubinsBicycleParams dyn;
  model.setParams(dyn);

  COST cost;
  cost.GPUSetup();

  FirstOrderDubinsBicycleCostParams<kRefHorizon> cost_params;
  cost_params.desired_speed = kTargetSpeed;
  cost_params.boundary_threshold = kPathBoundary;
  mppi::cost::fillFirstOrderDubinsBicycleCostGeometry<kRefHorizon>(cost_params, dyn);
  constexpr float kEgoLength = 0.55F * 1.5F;
  constexpr float kEgoWidth = 0.28F * 1.5F;
  mppi::cost::setFirstOrderDubinsBicycleCostEgoFootprint<kRefHorizon>(cost_params, dyn.wheel_base, kEgoLength,
                                                                    kEgoWidth);
  cost.setParams(cost_params);

  const float kMaxSteer = dyn.max_steer_angle;
  std::array<float2, DYN::CONTROL_DIM> u_rng{};
  u_rng[static_cast<int>(FirstOrderDubinsBicycleParams::ControlIndex::ACCELERATION_CMD)] = { dyn.min_accel,
                                                                                             dyn.max_accel };
  u_rng[static_cast<int>(FirstOrderDubinsBicycleParams::ControlIndex::STEER_CMD)] = { -kMaxSteer, kMaxSteer };
  model.setControlRanges(u_rng);

  SAMPLER::SAMPLING_PARAMS_T sp{};
  sp.std_dev[static_cast<int>(FirstOrderDubinsBicycleParams::ControlIndex::ACCELERATION_CMD)] = 0.35F;
  sp.std_dev[static_cast<int>(FirstOrderDubinsBicycleParams::ControlIndex::STEER_CMD)] = 0.06F;
  sp.sum_strides = std::max(32, (kNumRollouts + 1023) / 1024);
  SAMPLER sampler(sp);

  FB feedback(&model, kDt);
  Mppi::control_trajectory u_nom = Mppi::control_trajectory::Zero();
  Mppi::control_trajectory u_opt = u_nom;
  Mppi controller(&model, &cost, &feedback, &sampler, kDt, 1, kLambda, 0.0F, kMppiHorizon, u_nom);
  {
    auto cp = controller.getParams();
    cp.dynamics_rollout_dim_ = dim3(32, 2, 1);
    cp.cost_rollout_dim_ = dim3(32, 2, 1);
    cp.seed_ = 1U;
    controller.setParams(cp);
    controller.setPercentageSampledControlTrajectories(128.0F / static_cast<float>(kNumRollouts));
  }
  model.GPUSetup();

  DYN::state_array x = model.getZeroState();
  const mppi::path::Pose2D p0 = path.poseAt(kInitArcLength);
  x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::POS_X)) = p0.x;
  x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::POS_Y)) = p0.y;
  x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::YAW)) = p0.yaw;
  x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::VEL_X)) = kTargetSpeed;

  std::vector<float> obs_traj_x;
  std::vector<float> obs_traj_y;
  std::vector<float> obs_traj_yaw;
  std::vector<float> obs_half_length;
  std::vector<float> obs_half_width;

  cv::Mat base_frame = mppi::viz::makeWhiteFrame(1024, 1024);
  mppi::viz::drawStraightCorridor(base_frame, kLeftLaneX, kRoadYStart, kLeftLaneX, kRoadYEnd, kLaneHalfWidth);
  mppi::viz::drawStraightCorridor(base_frame, kRightLaneX, kRoadYStart, kRightLaneX, kRoadYEnd, kLaneHalfWidth);
  mppi::viz::drawRoadBoundaries(base_frame, path, kLaneHalfWidth);
  mppi::viz::drawCenterline(base_frame, path);

  const mppi::viz::TimeSeriesPlotLayout& plot_layout = mppi::viz::defaultTimeSeriesPlotLayout();
  const cv::Size composite_size =
      mppi::viz::compositeFrameSize(base_frame.cols, base_frame.rows, plot_layout);
  cv::VideoWriter video(video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), static_cast<int>(1.0F / kDt),
                        composite_size);

  cv::namedWindow("MPPI Two-Lane Double Park", cv::WINDOW_NORMAL);

  mppi::viz::RunningTimeSeries signal_history;

  std::ofstream log(log_path.c_str());
  if (!log)
  {
    std::cerr << "Could not open log: " << log_path << "\n";
    return 1;
  }
  log << "t,pos_x,pos_y,yaw,vel_x,steer_angle,brake_state,u_accel,u_steer,nom_u_accel,nom_u_steer,"
         "ref_x,ref_y,ref_yaw,ref_v_pose,ref_v_target,arc_s,lat_err,baseline\n";
  log << std::scientific;

  mppi::timing::StepTimingCollector step_timing;
  step_timing.reserve(static_cast<size_t>(num_sim_steps));

  const int state_x_idx = static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::POS_X);
  const int state_y_idx = static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::POS_Y);
  const int output_x_idx = static_cast<int>(FirstOrderDubinsBicycleParams::OutputIndex::BASELINK_POS_I_X);
  const int output_y_idx = static_cast<int>(FirstOrderDubinsBicycleParams::OutputIndex::BASELINK_POS_I_Y);

  const cv::Scalar kParkedFill(70, 70, 70);
  const cv::Scalar kParkedOutline(30, 30, 30);
  const cv::Scalar kTrafficFill(40, 80, 200);
  const cv::Scalar kTrafficOutline(20, 40, 120);

  for (int k = 0; k < num_sim_steps; ++k)
  {
    step_timing.beginStep();

    const float sim_time = static_cast<float>(k) * kDt;

    const std::vector<mppi::path::PathReferenceSample> ref = ref_gen.generate(
        path, arcLength, kRefHorizon, x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::POS_X)),
        x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::POS_Y)),
        x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::YAW)),
        x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::VEL_X)));
    mppi::cost::fillFirstOrderDubinsBicycleCostFromPathReference<kRefHorizon>(cost, ref);

    mppi::cost::buildObstacleTrajectoryBuffers(obstacles, sim_time, kDt, kRefHorizon, obs_traj_x, obs_traj_y,
                                               obs_traj_yaw, obs_half_length, obs_half_width);
    mppi::cost::fillFirstOrderDubinsBicycleCostObstacleTrajectories<kRefHorizon>(
        cost, obs_traj_x.data(), obs_traj_y.data(), obs_traj_yaw.data(), obs_half_length.data(), obs_half_width.data(),
        static_cast<int>(obstacles.size()), kRefHorizon);

    if (k > 0)
    {
      u_nom.leftCols(kMppiHorizon - 1) = u_opt.rightCols(kMppiHorizon - 1);
      u_nom.rightCols(1) = u_opt.rightCols(1);
    }

    controller.updateImportanceSampler(u_nom);
    const DYN::control_array u_nom_step = u_nom.col(0);

    controller.computeControl(x, 1);
    cudaStreamSynchronize(controller.stream_);
    controller.calculateSampledStateTrajectories();

    step_timing.endMppi();

    const Mppi::control_trajectory u_opt_traj = controller.getControlSeq();
    u_opt = u_opt_traj;

    const auto state_trajectory = controller.getActualStateSeq();
    const auto sampled_trajectories = controller.getSampledOutputTrajectories();
    const auto sampled_cost_trajs = controller.getSampledCostTrajectories();

    std::vector<float> rollout_costs(sampled_cost_trajs.size());
    for (size_t i = 0; i < sampled_cost_trajs.size(); ++i)
    {
      rollout_costs[i] = sampled_cost_trajs[i].sum();
    }

    const std::vector<mppi::cost::ParkedCarObstacle> obs_viz =
        mppi::cost::movingCarPosesAt(obstacles, sim_time);

    auto frame = base_frame.clone();
    mppi::viz::drawReferencePath(frame, ref);
    mppi::viz::drawSampledTrajectories(frame, sampled_trajectories, output_x_idx, output_y_idx, kMppiHorizon,
                                       rollout_costs);
    mppi::viz::drawTrajectory(frame, state_trajectory, state_x_idx, state_y_idx);
    if (!obs_viz.empty())
    {
      const std::vector<mppi::cost::ParkedCarObstacle> parked_only(1, obs_viz[0]);
      mppi::viz::drawParkedCars(frame, parked_only, kParkedFill, kParkedOutline);
    }
    if (obs_viz.size() > 1)
    {
      const std::vector<mppi::cost::ParkedCarObstacle> traffic_only(1, obs_viz[1]);
      mppi::viz::drawParkedCars(frame, traffic_only, kTrafficFill, kTrafficOutline);
    }
    mppi::viz::drawEgoVehicleAtRearAxle(frame, x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::POS_X)),
                                        x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::POS_Y)),
                                        x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::YAW)), kEgoLength,
                                        kEgoWidth, cost_params.ego_axle_to_box_center);

    if (cv::waitKey(1) == 27)
    {
      step_timing.endStepEarlyExit();
      break;
    }

    DYN::state_array x_next = model.getZeroState();
    DYN::state_array xdot = model.getZeroState();
    DYN::output_array y = DYN::output_array::Zero();

    DYN::control_array u_apply = u_opt_traj.col(0);
    model.enforceConstraints(x, u_apply);
    model.step(x, x_next, xdot, u_apply, y, static_cast<float>(k), kDt);

    x = x_next;

    const float t_end = static_cast<float>(k + 1) * kDt;
    const float accel_cmd = u_apply(static_cast<int>(FirstOrderDubinsBicycleParams::ControlIndex::ACCELERATION_CMD));
    const float steer_cmd = u_apply(static_cast<int>(FirstOrderDubinsBicycleParams::ControlIndex::STEER_CMD));
    const float vel_x = x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::VEL_X));
    const float steer_state = x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::STEER_ANGLE));
    signal_history.push(t_end, vel_x, accel_cmd, steer_cmd, steer_state);

    const cv::Mat composite = mppi::viz::composeFrameWithTimeSeriesPlots(
        frame, signal_history, kTargetSpeed, kVMax, dyn.min_accel, dyn.max_accel, kMaxSteer, plot_layout);
    video.write(composite);
    mppi::viz::showCompositeFrame("MPPI Two-Lane Double Park", composite, plot_layout);
    step_timing.endViz();

    const mppi::path::PathProjection proj = mppi::path::projectPoseOntoPath(
        path, x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::POS_X)),
        x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::POS_Y)), arcLength);
    arcLength = proj.arc_length_s;

    const mppi::path::PathReferenceSample& r0 = ref.front();
    const float ref_v_target = ref_gen.speedAt(path, proj.arc_length_s);
    log << t_end << "," << x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::POS_X)) << ","
        << x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::POS_Y)) << ","
        << x(static_cast<int>(FirstOrderDubinsBicycleParams::StateIndex::YAW)) << ","
        << vel_x << ","
        << steer_state << ",0," << accel_cmd << ","
        << steer_cmd << ","
        << u_nom_step(static_cast<int>(FirstOrderDubinsBicycleParams::ControlIndex::ACCELERATION_CMD)) << ","
        << u_nom_step(static_cast<int>(FirstOrderDubinsBicycleParams::ControlIndex::STEER_CMD)) << "," << r0.x << ","
        << r0.y << "," << r0.yaw << "," << r0.v << "," << ref_v_target << "," << proj.arc_length_s << ","
        << proj.signed_lateral_error << "," << static_cast<float>(controller.getBaselineCost()) << "\n";

    step_timing.endStep();
  }

  log.close();
  step_timing.printReport();
  cost.freeCudaMem();
  std::cout << "Wrote " << video_path << " and " << log_path << "\n";
  return 0;
}
