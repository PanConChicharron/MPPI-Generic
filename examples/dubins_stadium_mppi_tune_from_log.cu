/**
 * Re-run one MPPI iteration from a closed-loop tracking log with custom cost weights.
 *
 * Build: cmake --build build --target dubins_stadium_mppi_tune_from_log
 * Run:   ./build/examples/dubins_stadium_mppi_tune_from_log --log foo.csv --step 226 \\
 *          --prefix foo_tune_live --weights 20,3,5,1,0.05,0,0.05,0.05 --lambda 3000
 *
 * Step is 1-based (matches track_departure_step in meta). State is taken from the log row
 * immediately before that step's MPPI solve (row index step-2).
 */
#include "mppi_rollout_analysis_dump.hpp"

#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/cost_functions/path_tracking/path_tracking_cost.cuh>
#include <mppi/dynamics/dubins_bicycle/dubins_bicycle.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/path/path_projection.hpp>
#include <mppi/path/path_reference_generator.hpp>
#include <mppi/path/path_tracking_bridge.hpp>
#include <mppi/path/path2d.hpp>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
constexpr int kMppiHorizon = 50;
constexpr int kRefHorizon = kMppiHorizon + 8;
constexpr float kDt = 0.1F;
constexpr int kNumRollouts = 4 * 1024;
constexpr float kVMax = 3.0F;
constexpr float kStraightLength = 40.0F;
constexpr float kTurnRadius = 10.0F;
constexpr int kSamplesPerArc = 48;
constexpr float kNoiseStdAccel = 0.15F;
constexpr float kNoiseStdSteer = 0.12F;
constexpr float kNomLatSteerGain = 0.0F;
constexpr float kNomHeadingSteerGain = 0.0F;
constexpr float kLambdaDefault = 3000.0F;

using DYN = DubinsBicycle;
using COST = PathTrackingCost<kRefHorizon>;
using FB = DDPFeedback<DYN, kMppiHorizon>;
using SAMPLER = mppi::sampling_distributions::GaussianDistribution<DYN::DYN_PARAMS_T>;
using Mppi = VanillaMPPIController<DYN, COST, FB, kMppiHorizon, kNumRollouts, SAMPLER>;

struct CostWeights
{
  float w_pos = 20.0F;
  float w_heading_so2 = 3.0F;
  float w_vel = 5.0F;
  float w_lat_accel = 1.0F;
  float w_lat_jerk = 0.05F;
  float w_steer_dot = 0.0F;
  float w_accel = 0.05F;
  float w_steer = 0.05F;
};

struct LogRow
{
  float t = 0.0F;
  float pos_x = 0.0F;
  float pos_y = 0.0F;
  float yaw = 0.0F;
  float vel_x = 0.0F;
  float steer_angle = 0.0F;
  float arc_s = 0.0F;
};

bool parseFloatsCsv(const std::string& s, std::vector<float>& out)
{
  out.clear();
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
  {
    try
    {
      out.push_back(std::stof(item));
    }
    catch (...)
    {
      return false;
    }
  }
  return !out.empty();
}

bool loadLogRows(const std::string& path, std::vector<LogRow>& rows)
{
  std::ifstream f(path.c_str());
  if (!f)
  {
    return false;
  }
  std::string line;
  if (!std::getline(f, line))
  {
    return false;
  }
  rows.clear();
  while (std::getline(f, line))
  {
    if (line.empty())
    {
      continue;
    }
    std::stringstream ss(line);
    std::string cell;
    std::vector<float> vals;
    while (std::getline(ss, cell, ','))
    {
      try
      {
        vals.push_back(std::stof(cell));
      }
      catch (...)
      {
        vals.clear();
        break;
      }
    }
    if (vals.size() < 16U)
    {
      continue;
    }
    LogRow r;
    r.t = vals[0];
    r.pos_x = vals[1];
    r.pos_y = vals[2];
    r.yaw = vals[3];
    r.vel_x = vals[4];
    r.steer_angle = vals[5];
    r.arc_s = vals[15];
    rows.push_back(r);
  }
  return !rows.empty();
}

void appendMetaKeys(const std::string& meta_path, const CostWeights& w, const float lambda)
{
  std::ofstream meta(meta_path.c_str(), std::ios::app);
  if (!meta)
  {
    return;
  }
  meta << "w_pos," << w.w_pos << "\n";
  meta << "w_heading_so2," << w.w_heading_so2 << "\n";
  meta << "w_vel," << w.w_vel << "\n";
  meta << "w_lat_accel," << w.w_lat_accel << "\n";
  meta << "w_lat_jerk," << w.w_lat_jerk << "\n";
  meta << "w_steer_dot," << w.w_steer_dot << "\n";
  meta << "w_accel," << w.w_accel << "\n";
  meta << "w_steer," << w.w_steer << "\n";
  meta << "lambda," << lambda << "\n";
}

}  // namespace

int main(int argc, char** argv)
{
  std::string log_path;
  std::string prefix = "dubins_stadium_tune_live";
  int step_1based = 226;
  std::string weights_str;
  float lambda = kLambdaDefault;
  unsigned int seed = 42U;

  for (int a = 1; a < argc; ++a)
  {
    const std::string arg = argv[a];
    if (arg == "--log" && a + 1 < argc)
    {
      log_path = argv[++a];
    }
    else if (arg == "--prefix" && a + 1 < argc)
    {
      prefix = argv[++a];
    }
    else if (arg == "--step" && a + 1 < argc)
    {
      step_1based = std::stoi(argv[++a]);
    }
    else if (arg == "--weights" && a + 1 < argc)
    {
      weights_str = argv[++a];
    }
    else if (arg == "--lambda" && a + 1 < argc)
    {
      lambda = std::stof(argv[++a]);
    }
    else if (arg == "--seed" && a + 1 < argc)
    {
      seed = static_cast<unsigned int>(std::stoul(argv[++a]));
    }
    else if (arg == "--help" || arg == "-h")
    {
      std::cout << "Usage: dubins_stadium_mppi_tune_from_log --log PATH --step N [--prefix P] "
                   "[--weights w_pos,w_heading,w_vel,w_lat_a,w_lat_j,w_steer_dot,w_accel,w_steer] "
                   "[--lambda L] [--seed S]\n";
      return 0;
    }
  }

  if (log_path.empty())
  {
    std::cerr << "error: --log PATH required\n";
    return 1;
  }
  if (step_1based < 1)
  {
    std::cerr << "error: --step must be >= 1\n";
    return 1;
  }

  CostWeights cw{};
  if (!weights_str.empty())
  {
    std::vector<float> w;
    if (!parseFloatsCsv(weights_str, w) || w.size() != 8U)
    {
      std::cerr << "error: --weights needs 8 comma-separated floats\n";
      return 1;
    }
    cw.w_pos = w[0];
    cw.w_heading_so2 = w[1];
    cw.w_vel = w[2];
    cw.w_lat_accel = w[3];
    cw.w_lat_jerk = w[4];
    cw.w_steer_dot = w[5];
    cw.w_accel = w[6];
    cw.w_steer = w[7];
  }

  std::vector<LogRow> rows;
  if (!loadLogRows(log_path, rows))
  {
    std::cerr << "error: could not read log " << log_path << "\n";
    return 1;
  }

  const int row_index = step_1based - 2;
  if (row_index < 0 || row_index >= static_cast<int>(rows.size()))
  {
    std::cerr << "error: step " << step_1based << " out of range for log (" << rows.size() << " rows)\n";
    return 1;
  }
  const LogRow& row = rows[static_cast<size_t>(row_index)];

  const mppi::path::Path2D path = mppi::path::Path2D::stadium(kStraightLength, kTurnRadius, kSamplesPerArc);
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
  mppi::path::fillPathTrackingCostWeights<kRefHorizon>(cost_params, cw.w_pos, cw.w_heading_so2, cw.w_vel, cw.w_lat_accel,
                                                       cw.w_lat_jerk, cw.w_steer_dot, cw.w_accel, cw.w_steer);
  mppi::path::fillPathTrackingBicycleGeometry<kRefHorizon>(cost_params, dyn);

  SAMPLER::SAMPLING_PARAMS_T sp{};
  sp.std_dev[static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)] = kNoiseStdAccel;
  sp.std_dev[static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)] = kNoiseStdSteer;
  sp.control_cost_coeff[0] = cost_params.control_cost_coeff[0];
  sp.control_cost_coeff[1] = cost_params.control_cost_coeff[1];
  sp.sum_strides = std::max(32, (kNumRollouts + 1023) / 1024);
  SAMPLER sampler(sp);

  FB feedback(&model, kDt);
  Mppi::control_trajectory u_nom = Mppi::control_trajectory::Zero();
  Mppi controller(&model, &cost, &feedback, &sampler, kDt, 1, lambda, 0.0F, kMppiHorizon, u_nom);
  controller.setPercentageSampledControlTrajectories(1.0F);
  {
    auto cp = controller.getParams();
    cp.dynamics_rollout_dim_ = dim3(32, 2, 1);
    cp.cost_rollout_dim_ = dim3(kMppiHorizon, 1, 1);
    cp.seed_ = seed;
    controller.setParams(cp);
  }
  controller.setKernelChoice(kernelType::USE_SPLIT_KERNELS);
  model.GPUSetup();
  cost.GPUSetup();

  DYN::state_array x = model.getZeroState();
  x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_X)) = row.pos_x;
  x(static_cast<int>(DubinsBicycleParams::StateIndex::POS_Y)) = row.pos_y;
  x(static_cast<int>(DubinsBicycleParams::StateIndex::YAW)) = row.yaw;
  x(static_cast<int>(DubinsBicycleParams::StateIndex::VEL_X)) = row.vel_x;
  x(static_cast<int>(DubinsBicycleParams::StateIndex::STEER_ANGLE)) = row.steer_angle;

  float arc_s = row.arc_s;
  const mppi::path::PathProjection proj =
      mppi::path::projectPoseOntoPath(path, row.pos_x, row.pos_y, arc_s);
  arc_s = proj.arc_length_s;
  const float along_v = mppi::path::alongPathSpeedFromState(path, arc_s, x, &proj.signed_lateral_error);
  const std::vector<mppi::path::PathReferenceSample> ref =
      ref_gen.generate(path, arc_s, kRefHorizon, along_v);
  mppi::path::fillCostFromPathReference<kRefHorizon>(cost_params, ref, &path, &dyn);
  cost.setParams(cost_params);
  mppi::path::fillNominalControlFromReference(u_nom, x, ref, dyn, kDt, &path, kNomLatSteerGain, kNomHeadingSteerGain);
  controller.updateImportanceSampler(u_nom);

  controller.computeControl(x, 1);

  const float t_solve = static_cast<float>(step_1based - 1) * kDt;
  mppi::rollout_csv::dumpSingleMppiIteration<DYN, COST, Mppi, SAMPLER, Mppi::control_trajectory, Mppi::output_trajectory>(
      model, cost, controller, sampler, x, prefix, kDt, lambda, kMppiHorizon, kNumRollouts, &path, step_1based, t_solve,
      proj.distance, &ref);
  appendMetaKeys(prefix + "_meta.csv", cw, lambda);

  std::cout << "Tuned MPPI dump: step=" << step_1based << " t=" << t_solve << " prefix=" << prefix << "\n";
  return 0;
}
