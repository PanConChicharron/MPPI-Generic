/**
 * @file racer_dubins_stadium_path_tracking_example.cu
 * @brief Example of MPPI-based path tracking and obstacle avoidance for a Racer Dubins model on a stadium track.
 */

#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>
#include <mppi/cost_functions/racer/racer_cost_map.cuh>
#include <mppi/cost_functions/racer/racer_costmap_builder.hpp>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/feedback_controllers/path_tracker_feedback.cuh>
#include <mppi/path/path_projection.hpp>
#include <mppi/path/path_reference_generator.hpp>
#include <mppi/path/path2d.hpp>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

#include <mppi/viz/path_tracking_viz.hpp>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <random>

namespace
{
  // --- Simulation Parameters ---
  constexpr int kMppiHorizon = 50;
  constexpr int kRefHorizon = kMppiHorizon;
  constexpr float kDt = 0.1F;
  constexpr int kNumRollouts = 4*1024;
  constexpr float kTargetSpeed = 2.5F;
  constexpr float kVMax = 3.0F;
  constexpr size_t kSimLaps = 1;
  
  constexpr float kStraightLength = 40.0F;
  constexpr float kTurnRadius = 10.0F;
  constexpr int kSamplesPerArc = 48;
  
  constexpr float kInitArcLength = kStraightLength - 2.0F;
  constexpr float kInitLateralOffset = 0.1F;
  
  constexpr float kLambda = 100.0F;

  // --- MPPI Controller Setup ---
  using DYN = RacerDubins;
  using COST = RacerCostMap;
  using FB = PathTrackerFeedback<DYN, kMppiHorizon>;
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

    /* Environment */
    // 2. Generate track path (stadium shape)
    const mppi::path::Path2D path = mppi::path::Path2D::stadium(kStraightLength, kTurnRadius, kSamplesPerArc);
    
    float x_min = -40, x_max = 60, y_min = -30, y_max = 30;
    float ppm = 10.0f;
    int width = (x_max - x_min) * ppm;
    int height = (y_max - y_min) * ppm;
    
    cv::Mat costmap_img;

    std::vector<mppi::cost::RacerCostmapObstacle> obstacles;
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
    std::vector<float4> cost_map_gpu_data;
    mppi::cost::updateRacerCostmap(costmap_img, ref_init, obstacles, width, height, ppm, x_min, y_min,
                                   cost_map_gpu_data);

    COST cost;
    cost.GPUSetup();
    cost.costmapToTexture(width, height, cost_map_gpu_data.data());
    cost.setCpuCostmap(costmap_img);

    RacerCostMapParams cost_params;
    cost_params.desired_speed = kTargetSpeed;
    cost.setParams(cost_params);
    cost.setWorldToCostmapBounds(x_min, x_max, y_min, y_max);

    /* Model parameters */
    // 5. Setup model and sampling distributions
    DYN model;
    RacerDubinsParams dyn;
    dyn.wheel_base = 0.3f;
    model.setParams(dyn);
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
    mppi::viz::drawRoadBoundaries(base_frame, path, 0.8F);
    mppi::viz::drawCenterline(base_frame, path);
    // Draw obstacles on base frame
    for (const auto& obs : obstacles) {
        cv::circle(base_frame, mppi::viz::worldToPixel(obs.ox, obs.oy, 1024, 1024), obs.r * 15.0f, cv::Scalar(0, 0, 255), -1);
    }

    cv::VideoWriter video(video_path, 
                      cv::VideoWriter::fourcc('m','p','4','v'), 
                      static_cast<int>(1.0F/kDt), base_frame.size());
    cv::VideoWriter costmap_video("racer_dubins_stadium_costmap.mp4", 
                               cv::VideoWriter::fourcc('m','p','4','v'), 
                               static_cast<int>(1.0F/kDt), costmap_img.size());

    cv::namedWindow("MPPI Tracking", cv::WINDOW_NORMAL);
    cv::resizeWindow("MPPI Tracking", base_frame.cols, base_frame.rows);

    // 9. Main simulation loop
    for (size_t k = 0; k < num_sim_steps; ++k) {
      const std::vector<mppi::path::PathReferenceSample> ref = ref_gen.generate(path, arcLength, kRefHorizon);
      
      // Update the costmap
      mppi::cost::updateRacerCostmap(costmap_img, ref, obstacles, width, height, ppm, x_min, y_min,
                                     cost_map_gpu_data);
      
      // Save costmap frame
      cv::Mat costmap_vis;
      costmap_img.convertTo(costmap_vis, CV_8UC1, 255.0);
      cv::cvtColor(costmap_vis, costmap_vis, cv::COLOR_GRAY2BGR);
      costmap_video.write(costmap_vis);

      cost.updateCostmapTexture(cost_map_gpu_data.data());
      cost.setCpuCostmap(costmap_img);

      const Mppi::state_trajectory goal_traj =
          mppi::feedback::goalTrajectoryFromPathReference<DYN, kMppiHorizon>(ref);
      feedback.updateReference(path, ref);
      feedback.applyFeedforwardToNominal(u_nom, x, goal_traj);

      // Update importance sampling based on current nominal control
      controller.updateImportanceSampler(u_nom);

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
    }

    cost.freeCudaMem();
    return 0;
}