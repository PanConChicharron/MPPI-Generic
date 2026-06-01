/**
 * @file racer_dubins_stadium_path_tracking_example.cu
 * @brief Example of MPPI-based path tracking and obstacle avoidance for a Racer Dubins model on a stadium track.
 *
 * Build: cmake --build build --target racer_dubins_stadium_path_tracking_example
 * Run:   ./build/examples/racer_dubins_stadium_path_tracking_example [seed] [log.csv]
 * Plot:  python3 examples/plot_racer_dubins_temporal_mppi.py racer_dubins_stadium_path_tracking_log.csv
 */

#include "mppi_rollout_csv.hpp"

#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>
#include <mppi/cost_functions/racer/racer_cost.cuh>
#include <mppi/cost_functions/racer/racer_cost_bridge.hpp>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/feedback_controllers/zero_feedback.cuh>
#include <mppi/path/path_projection.hpp>
#include <mppi/path/path_reference_generator.hpp>
#include <mppi/path/path2d.hpp>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

#include "path_tracking_viz.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>

namespace
{
  // --- Simulation Parameters ---
  constexpr int kMppiHorizon = 50;
  constexpr int kRefHorizon = kMppiHorizon;
  constexpr float kDt = 0.1F;
  constexpr int kNumRollouts = 32*1024;
  constexpr float kTargetSpeed = 5.0F;
  constexpr float kVMax = 5.0F;
  constexpr size_t kSimLaps = 5;
  
  constexpr float kStraightLength = 40.0F;
  constexpr float kTurnRadius = 10.0F;
  constexpr int kSamplesPerArc = 48;
  
  constexpr float kInitArcLength = kStraightLength - 2.0F;
  
  constexpr float kLambda = 100.0F;

  // --- MPPI Controller Setup ---
  using DYN = RacerDubins;
  using COST = RacerCost<kRefHorizon>;
  using FB = ZeroFeedback<DYN, kMppiHorizon>;
  using SAMPLER = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
  using Mppi = VanillaMPPIController<DYN, COST, FB, kMppiHorizon, kNumRollouts, SAMPLER>;

  /**
   * @brief Calculates simulation steps needed for a given number of laps.
   */
  int simStepsForLaps(const mppi::path::Path2D& path, const float laps)
  {
    const float lap_time = path.length() / kVMax;
    return static_cast<int>(std::ceil(laps * lap_time / kDt));
  }
}  // namespace

/**
 * @brief Main simulation loop for the Racer Dubins path tracking example.
 *
 * @param argc Argument count.
 * @param argv Argument vector (optional: seed for RNG).
 * @return int Exit code.
 */
int main(int argc, char** argv)
{
    // 1. Initialize random seed for reproducibility
    unsigned int seed = 42;
    if (argc > 1) {
        seed = std::stoul(argv[1]);
    }
    std::cout << "Using random seed: " << seed << std::endl;
    std::string video_path = "racer_dubins_stadium_path_tracking.mp4";
    std::string log_path = "racer_dubins_stadium_path_tracking_log.csv";
    if (argc > 2) {
        log_path = argv[2];
    }

    /* Environment */
    // 2. Generate track path (stadium shape)
    const mppi::path::Path2D path = mppi::path::Path2D::stadium(kStraightLength, kTurnRadius, kSamplesPerArc);
    mppi::rollout_csv::writeCenterlineForLog(path, log_path);

    std::vector<mppi::cost::RacerCostObstacle> obstacles;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist_s(0, path.length());
    std::uniform_real_distribution<float> dist_side(-1.0, 1.0);
    std::uniform_real_distribution<float> dist_r(2.0, 4.5);
    
    for (int i = 0; i < 15; ++i) {
        float s = dist_s(gen);
        float side = (dist_side(gen) > 0 ? 1.0 : -1.0) * 2.5; 
        float r = dist_r(gen);
        auto p = path.poseAt(s);
        float tx, ty;
        path.tangentAt(s, tx, ty);
        // obstacles.emplace_back(p.x - side * ty, p.y + side * tx, r);
    }

    mppi::path::PathReferenceGenerator ref_gen(kDt);
    ref_gen.setSpeedCap(kVMax);

    const size_t num_sim_steps = simStepsForLaps(path, kSimLaps);

    float arcLength = kInitArcLength;
    const std::vector<mppi::path::PathReferenceSample> ref_init =
        ref_gen.generate(path, arcLength, kRefHorizon);

    /* Model parameters */
    // 5. Setup model and sampling distributions
    DYN model;
    RacerDubinsParams dyn;
    dyn.wheel_base = 0.3f;
    model.setParams(dyn);

    COST cost;
    cost.GPUSetup();

    RacerCostParams<kRefHorizon> cost_params;
    cost_params.desired_speed = kTargetSpeed;
    // Keep curvature/jerk comfort terms consistent with the active vehicle model.
    cost_params.wheel_base = dyn.wheel_base;
    cost_params.steer_angle_scale = dyn.steer_angle_scale;
    cost.setParams(cost_params);
    mppi::cost::fillRacerCostFromPathReference<kRefHorizon>(cost, ref_init);
    mppi::cost::fillRacerCostObstacles<kRefHorizon>(cost, obstacles);

    std::array<float2, DYN::CONTROL_DIM> u_rng{};
    u_rng[static_cast<int>(RacerDubinsParams::ControlIndex::THROTTLE_BRAKE)] = { -1.0f, 1.0f };
    u_rng[static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD)] = { -1.0f, 1.0f };
    model.setControlRanges(u_rng);

    SAMPLER::SAMPLING_PARAMS_T sp{};
    sp.std_dev[static_cast<int>(RacerDubinsParams::ControlIndex::THROTTLE_BRAKE)] = 0.2F;
    sp.std_dev[static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD)] = 0.3F;
    sp.sum_strides = std::max(32, (kNumRollouts + 1023) / 1024);
    SAMPLER sampler(sp);

    // 6. Setup MPPI Controller
    FB feedback(&model, kDt);
    Mppi::control_trajectory u_nom = Mppi::control_trajectory::Zero();
    Mppi controller(&model, &cost, &feedback, &sampler, kDt, 1, kLambda, 0.0F, kMppiHorizon, u_nom);
    {
      auto cp = controller.getParams();
      cp.dynamics_rollout_dim_ = dim3(32, 2, 1);
      cp.cost_rollout_dim_ = dim3(32, 2, 1);
      cp.seed_ = 1U;
      controller.setParams(cp);
      // 128 sampled rollouts (128 % 32 == 0); 0.01F yields 40 and breaks visualizeKernel
      controller.setPercentageSampledControlTrajectories(128.0F / static_cast<float>(kNumRollouts));
    }
    model.GPUSetup();

    // 7. Initialize simulation state
    DYN::state_array x = model.getZeroState();
    const mppi::path::Pose2D p0 = path.poseAt(kInitArcLength);
    x(static_cast<int>(RacerDubinsParams::StateIndex::POS_X)) = p0.x;
    x(static_cast<int>(RacerDubinsParams::StateIndex::POS_Y)) = p0.y;
    x(static_cast<int>(RacerDubinsParams::StateIndex::YAW)) = p0.yaw;
    x(static_cast<int>(RacerDubinsParams::StateIndex::VEL_X)) = kTargetSpeed;

    // 8. Prepare visualization
    cv::Mat base_frame = mppi::viz::makeWhiteFrame(1024, 1024);
    mppi::viz::drawCenterline(base_frame, path);
    for (const auto& obs : obstacles) {
        cv::circle(base_frame, mppi::viz::worldToPixel(obs.ox, obs.oy, 1024, 1024), obs.r * 15.0f, cv::Scalar(0, 0, 255), -1);
    }

    cv::VideoWriter video(video_path, 
                      cv::VideoWriter::fourcc('m','p','4','v'), 
                      static_cast<int>(1.0F/kDt), base_frame.size());

    cv::namedWindow("MPPI Tracking", cv::WINDOW_NORMAL);
    cv::resizeWindow("MPPI Tracking", base_frame.cols, base_frame.rows);

    std::ofstream log(log_path.c_str());
    if (!log) {
      std::cerr << "Could not open log: " << log_path << "\n";
      return 1;
    }
    // Same schema as dubins_circle_path_tracking_example (u_accel/u_steer columns hold throttle/steer).
    log << "t,pos_x,pos_y,yaw,vel_x,steer_angle,brake_state,u_accel,u_steer,nom_u_accel,nom_u_steer,"
           "ref_x,ref_y,ref_yaw,ref_v,arc_s,lat_err,baseline\n";
    log << std::scientific;

    // 9. Main simulation loop
    for (size_t k = 0; k < num_sim_steps; ++k) {
      const std::vector<mppi::path::PathReferenceSample> ref = ref_gen.generate(path, arcLength, kRefHorizon);
      mppi::cost::fillRacerCostFromPathReference<kRefHorizon>(cost, ref);
      
      // Update importance sampling based on current nominal control
      controller.updateImportanceSampler(u_nom);
      const DYN::control_array u_nom_step = u_nom.col(0);

      // Compute control sequence
      controller.computeControl(x, 1);
      cudaStreamSynchronize(controller.stream_);
      controller.calculateSampledStateTrajectories();
      
      Mppi::control_trajectory u_opt = controller.getControlSeq();

      /* Video frame generation */
      const auto state_trajectory = controller.getActualStateSeq();
      const auto sampled_trajectories = controller.getSampledOutputTrajectories();
      const auto sampled_cost_trajs = controller.getSampledCostTrajectories();

      std::vector<float> rollout_costs(sampled_cost_trajs.size());
      for (size_t i = 0; i < sampled_cost_trajs.size(); ++i)
      {
        rollout_costs[i] = sampled_cost_trajs[i].sum();
      }

      const int state_x_idx = static_cast<int>(RacerDubinsParams::StateIndex::POS_X);
      const int state_y_idx = static_cast<int>(RacerDubinsParams::StateIndex::POS_Y);
      const int output_x_idx = static_cast<int>(RacerDubinsParams::OutputIndex::BASELINK_POS_I_X);
      const int output_y_idx = static_cast<int>(RacerDubinsParams::OutputIndex::BASELINK_POS_I_Y);

      auto frame = base_frame.clone();

      mppi::viz::drawReferencePath(frame, ref);
      mppi::viz::drawSampledTrajectories(frame, sampled_trajectories, output_x_idx, output_y_idx, kMppiHorizon,
                                         rollout_costs);
      mppi::viz::drawTrajectory(frame, state_trajectory, state_x_idx, state_y_idx);
      video.write(frame);

      cv::imshow("MPPI Tracking", frame);
      if (cv::waitKey(1) == 27) break; // Exit on ESC

      // Step simulation model
      DYN::state_array x_next = model.getZeroState();
      DYN::state_array xdot = model.getZeroState();
      DYN::output_array y = DYN::output_array::Zero();

      model.enforceConstraints(x, u_opt.col(0));
      model.step(x, x_next, xdot, u_opt.col(0), y, static_cast<float>(k), kDt);

      // Shift nominal control trajectory for the next step
      u_nom.leftCols(kMppiHorizon - 1) = u_opt.rightCols(kMppiHorizon - 1);
      u_nom.rightCols(1) = u_opt.rightCols(1); 

      x = x_next;

      // Project state onto path to update progress
      const mppi::path::PathProjection proj = mppi::path::projectPoseOntoPath(path, x(static_cast<int>(RacerDubinsParams::StateIndex::POS_X)), x(static_cast<int>(RacerDubinsParams::StateIndex::POS_Y)), arcLength);
      arcLength = proj.arc_length_s;

      const mppi::path::PathReferenceSample& r0 = ref.front();
      const float t_end = static_cast<float>(k + 1) * kDt;
      log << t_end << ","
          << x(static_cast<int>(RacerDubinsParams::StateIndex::POS_X)) << ","
          << x(static_cast<int>(RacerDubinsParams::StateIndex::POS_Y)) << ","
          << x(static_cast<int>(RacerDubinsParams::StateIndex::YAW)) << ","
          << x(static_cast<int>(RacerDubinsParams::StateIndex::VEL_X)) << ","
          << x(static_cast<int>(RacerDubinsParams::StateIndex::STEER_ANGLE)) << ","
          << x(static_cast<int>(RacerDubinsParams::StateIndex::BRAKE_STATE)) << ","
          << u_opt.col(0)(static_cast<int>(RacerDubinsParams::ControlIndex::THROTTLE_BRAKE)) << ","
          << u_opt.col(0)(static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD)) << ","
          << u_nom_step(static_cast<int>(RacerDubinsParams::ControlIndex::THROTTLE_BRAKE)) << ","
          << u_nom_step(static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD)) << ","
          << r0.x << "," << r0.y << "," << r0.yaw << "," << r0.v << ","
          << proj.arc_length_s << "," << proj.signed_lateral_error << ","
          << static_cast<float>(controller.getBaselineCost()) << "\n";
    }

    log.close();
    cost.freeCudaMem();
    return 0;
}
