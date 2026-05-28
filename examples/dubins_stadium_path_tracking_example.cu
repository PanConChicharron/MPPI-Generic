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

#include <opencv2/opencv.hpp>

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
  constexpr int kNumRollouts = 4*1024;
  constexpr float kTargetSpeed = 2.5F;
  constexpr float kVMax = 3.0F;
  constexpr size_t kSimLaps = 1;
  
  constexpr float kStraightLength = 40.0F;
  constexpr float kTurnRadius = 10.0F;
  constexpr int kSamplesPerArc = 48;
  
  // s = kStraightLength is exactly the right-turn corner entry of the stadium. Start 2 m before it
  // so MPPI sees the curvature step around step 8 of the horizon - the bend lives in the middle of
  // the prediction window where the rollouts have spread enough to be informative.
  constexpr float kInitArcLength = kStraightLength - 2.0F;
  constexpr float kInitLateralOffset = 0.1F;
  
  // Match the closed-loop stadium tracking example so this analysis reflects the same controller.
  constexpr float kNoiseStdAccel = 0.15F;
  constexpr float kNoiseStdSteer = 0.12F;
  constexpr float kNomLatSteerGain = 0.0F;
  constexpr float kNomHeadingSteerGain = 0.0F;
  constexpr float kLambda = 30.0F;
  
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

  cv::Point2f worldToPixel(float x, float y, int img_w, int img_h)
  {
    const float scale = 15.0F;
    const float u = static_cast<float>(img_w) / 2.0F + x * scale;
    const float v = static_cast<float>(img_h) / 2.0F - y * scale;
    return cv::Point2f(u, v);
  }

  void draw_centerline(cv::Mat& img, const mppi::path::Path2D& path)
  {
    cv::Mat overlay = img.clone();
    const auto& anchors = path.anchors();
    for (size_t i = 0; i < anchors.size() - 1; ++i)
    {
      cv::line(overlay, worldToPixel(anchors[i].x, anchors[i].y, img.cols, img.rows),
               worldToPixel(anchors[i + 1].x, anchors[i + 1].y, img.cols, img.rows),
               cv::Scalar(128, 128, 128), 2);
    }
    cv::addWeighted(overlay, 0.5, img, 0.5, 0, img);
  }

  void draw_reference_path(cv::Mat& img, const std::vector<mppi::path::PathReferenceSample>& ref)
  {
    if (ref.size() < 2) return;
    cv::Mat overlay = img.clone();
    for (size_t i = 0; i < ref.size() - 1; ++i)
    {
      cv::line(overlay, worldToPixel(ref[i].x, ref[i].y, img.cols, img.rows),
               worldToPixel(ref[i + 1].x, ref[i + 1].y, img.cols, img.rows),
               cv::Scalar(255, 0, 0), 2); // Blue
    }
    cv::addWeighted(overlay, 0.5, img, 0.5, 0, img);
  }

  void draw_trajectory(cv::Mat& img, const Mppi::state_trajectory& traj)
  {
    cv::Mat overlay = img.clone();
    const int x_idx = static_cast<int>(DubinsBicycleParams::StateIndex::POS_X);
    const int y_idx = static_cast<int>(DubinsBicycleParams::StateIndex::POS_Y);
    for (int i = 0; i < traj.cols() - 1; ++i)
    {
      cv::line(overlay, worldToPixel(traj(x_idx, i), traj(y_idx, i), img.cols, img.rows),
               worldToPixel(traj(x_idx, i + 1), traj(y_idx, i + 1), img.cols, img.rows),
               cv::Scalar(0, 255, 0), 2); // Green
    }
    cv::addWeighted(overlay, 0.75, img, 0.25, 0, img);
  }

  void draw_sampled_trajectories(cv::Mat& img, const std::vector<Mppi::output_trajectory>& sampled_trajectories)
  {
    if (sampled_trajectories.empty()) return;
    cv::Mat overlay = img.clone();
    const int x_idx = static_cast<int>(DubinsBicycleParams::OutputIndex::POS_X);
    const int y_idx = static_cast<int>(DubinsBicycleParams::OutputIndex::POS_Y);
    std::cout << "plotting " << sampled_trajectories.size() << " sampled trajectories\n";
    for (const auto& traj : sampled_trajectories)
    {
      // NOTE: the first point corresponds to t=1
      for (int i = 0; i + 1 < traj.cols(); ++i)
      {
        cv::line(overlay, worldToPixel(traj(x_idx, i), traj(y_idx, i), img.cols, img.rows),
                 worldToPixel(traj(x_idx, i + 1), traj(y_idx, i + 1), img.cols, img.rows),
                 cv::Scalar(180, 180, 180), 1); // Light Gray
      }
    }
    cv::addWeighted(overlay, 0.4, img, 0.6, 0, img);
  }
}  // namespace

int main(int argc, char** argv)
{
    std::string log_path = "dubins_stadium_path_tracking_log.csv";
    std::string video_path = "dubins_stadium_path_tracking_log.mp4"; // path to the video file to generate with OpenCV

    const mppi::path::Path2D path = mppi::path::Path2D::stadium(kStraightLength, kTurnRadius, kSamplesPerArc);
    const size_t num_sim_steps = simStepsForLaps(path, kSimLaps);
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
    // Order: w_pos, w_heading_so2, w_vel, w_lat_accel, w_lat_jerk, w_steer_dot, w_accel, w_steer.
    // These match the closed-loop dubins_stadium_path_tracking_example defaults.
    mppi::path::fillPathTrackingCostWeights<kRefHorizon>(cost_params, 2.0F, 1.0F, 5.0F, 5.0F, 20.0F, 50.0F, 5.0F, 0.5F);
    mppi::path::fillPathTrackingBicycleGeometry<kRefHorizon>(cost_params, dyn);
    cost.setParams(cost_params);

    SAMPLER::SAMPLING_PARAMS_T sp{};
    sp.std_dev[static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)] = kNoiseStdAccel;
    sp.std_dev[static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)] = kNoiseStdSteer;
    sp.control_cost_coeff[0] = cost_params.control_cost_coeff[0];
    sp.control_cost_coeff[1] = cost_params.control_cost_coeff[1];
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
      controller.setPercentageSampledControlTrajectories(1.0F);
    }
    // PathTrackingCost has a large device params blob; the combined rollout kernel mis-aligns
    // shared memory (illegal memory access). Use split kernels for the optimization itself.
    controller.setKernelChoice(kernelType::USE_SPLIT_KERNELS);
    model.GPUSetup();
    cost.GPUSetup();

    DYN::state_array x = model.getZeroState();
    const mppi::path::Pose2D p0 = path.poseAt(kInitArcLength);
    float init_x = p0.x;
    float init_y = p0.y;
    mppi::path::applyInitialLateralOffset(path, kInitArcLength, kInitLateralOffset, init_x, init_y);
    x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_X)) = init_x;
    x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_Y)) = init_y;
    x(static_cast<int>(DubinsBicycleParams::StateIndex::YAW)) = p0.yaw;
    x(static_cast<int>(DubinsBicycleParams::StateIndex::VEL_X)) = kTargetSpeed;

    std::ofstream log(log_path.c_str());
    if (!log) {
      std::cerr << "Could not open log: " << log_path << "\n";
      return 1;
    }
    log << "t,pos_x,pos_y,yaw,vel_x,steer_angle,brake_state,u_accel,u_steer,nom_u_accel,nom_u_steer,"
           "ref_x,ref_y,ref_yaw,ref_v,arc_s,lat_err,baseline\n";
    log << std::scientific;

    float arcLength = kInitArcLength;

    cv::Mat base_frame = cv::Mat::zeros(1024, 1024, CV_8UC3);
    // prepare the base frame with the static elements
    draw_centerline(base_frame, path);
    cv::VideoWriter video(video_path, 
                      cv::VideoWriter::fourcc('m','p','4','v'), 
                      static_cast<int>(1.0F/kDt), base_frame.size());

    cv::namedWindow("MPPI Tracking", cv::WINDOW_NORMAL);

    for (size_t k = 0; k < num_sim_steps; ++k) {
      const std::vector<mppi::path::PathReferenceSample> ref = ref_gen.generate(path, arcLength, kRefHorizon);
      mppi::path::fillCostFromPathReference<kRefHorizon>(cost_params, ref, &path, &dyn);
      cost.setParams(cost_params);
      mppi::path::fillNominalControlFromReference(u_nom, x, ref, dyn, kDt, &path, kNomLatSteerGain, kNomHeadingSteerGain);
      controller.updateImportanceSampler(u_nom);
      const DYN::control_array u_nom_step = u_nom.col(0);

      controller.computeControl(x, 1);
      cudaStreamSynchronize(controller.stream_);
      controller.calculateSampledStateTrajectories();
      
      Mppi::control_trajectory u_opt = controller.getControlSeq();

      /* Video frame generation */
      const auto state_trajectory = controller.getActualStateSeq();
      const auto sampled_trajectories = controller.getSampledOutputTrajectories();
      auto frame = base_frame.clone();
      draw_reference_path(frame, ref);
      // draw_sampled_trajectories(frame, sampled_trajectories); // TODO: this is broken right now
      draw_trajectory(frame, state_trajectory);
      video.write(frame);

      cv::imshow("MPPI Tracking", frame);
      if (cv::waitKey(1) == 27) break; // Exit on ESC

      DYN::state_array x_next = model.getZeroState();
      DYN::state_array xdot = model.getZeroState();
      DYN::output_array y = DYN::output_array::Zero();

      model.enforceConstraints(x, u_opt.col(0));
      model.step(x, x_next, xdot, u_opt.col(0), y, static_cast<float>(k), kDt);

      x = x_next;

      const mppi::path::PathProjection proj = mppi::path::projectPoseOntoPath(path, x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_X)), x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_Y)), arcLength);
      arcLength = proj.arc_length_s;

      const mppi::path::PathReferenceSample& r0 = ref.front();
      const float t_end = static_cast<float>(k + 1) * kDt;
      log << t_end << ","
          << x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_X)) << ","
          << x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_Y)) << ","
          << x(static_cast<int>(DubinsBicycleParams::StateIndex::YAW)) << ","
          << x(static_cast<int>(DubinsBicycleParams::StateIndex::VEL_X)) << ","
          << x(static_cast<int>(DubinsBicycleParams::StateIndex::STEER_ANGLE)) << ",0,"
          << u_opt.col(0)(static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)) << ","
          << u_opt.col(0)(static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)) << ","
          << u_nom_step(static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)) << ","
          << u_nom_step(static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)) << ","
          << r0.x << "," << r0.y << "," << r0.yaw << "," << r0.v << ","
          << proj.arc_length_s << "," << proj.signed_lateral_error << ","
          << static_cast<float>(controller.getBaselineCost()) << "\n";
    }

    log.close();
    return 0;
  }