/**
 * @file racer_dubins_stadium_path_tracking_example.cu
 * @brief Example of MPPI-based path tracking and obstacle avoidance for a Racer Dubins model on a stadium track.
 */

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

    /* Environment */
    // 2. Generate track path (stadium shape)
    const mppi::path::Path2D path = mppi::path::Path2D::stadium(kStraightLength, kTurnRadius, kSamplesPerArc);

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

    COST cost;
    cost.GPUSetup();

    RacerCostParams<kRefHorizon> cost_params;
    cost_params.desired_speed = kTargetSpeed;
    cost.setParams(cost_params);
    mppi::cost::fillRacerCostFromPathReference<kRefHorizon>(cost, ref_init);
    mppi::cost::fillRacerCostObstacles<kRefHorizon>(cost, obstacles);

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
      controller.setPercentageSampledControlTrajectories(0.01F);
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
    cv::Mat base_frame = cv::Mat::zeros(1024, 1024, CV_8UC3);
    mppi::viz::drawCenterline(base_frame, path);
    for (const auto& obs : obstacles) {
        cv::circle(base_frame, mppi::viz::worldToPixel(obs.ox, obs.oy, 1024, 1024), obs.r * 15.0f, cv::Scalar(0, 0, 255), -1);
    }

    cv::VideoWriter video(video_path, 
                      cv::VideoWriter::fourcc('m','p','4','v'), 
                      static_cast<int>(1.0F/kDt), base_frame.size());

    cv::namedWindow("MPPI Tracking", cv::WINDOW_NORMAL);
    cv::resizeWindow("MPPI Tracking", base_frame.cols, base_frame.rows);

    // 9. Main simulation loop
    for (size_t k = 0; k < num_sim_steps; ++k) {
      const std::vector<mppi::path::PathReferenceSample> ref = ref_gen.generate(path, arcLength, kRefHorizon);
      mppi::cost::fillRacerCostFromPathReference<kRefHorizon>(cost, ref);
      
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
      
      const int x_idx = static_cast<int>(RacerDubinsParams::StateIndex::POS_X);
      const int y_idx = static_cast<int>(RacerDubinsParams::StateIndex::POS_Y);

      auto frame = base_frame.clone();
      
      mppi::viz::drawReferencePath(frame, ref);
      mppi::viz::drawSampledTrajectories(frame, sampled_trajectories, x_idx, y_idx);
      mppi::viz::drawTrajectory(frame, state_trajectory, x_idx, y_idx);
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
