/**
 * RacerDubins + MPPI: closed 2D road loop — stylized Nordschleife-like centerline (long straights,
 * esses, fast / mid / tight sections). Not a geographic replica; Catmull–Rom through control
 * points, dense polyline, s mod perimeter, constant v_ref. The track is a **closed** polyline: the
 *  last sample is snapped to the first (S/F) before arc length is accumulated. Also writes
 *  <log_stem>_centerline.csv (x_m,y_m) for matplotlib so the plan view can show a full closed loop
 *  (red dashed s(t) in the log is only a partial path if T*v_ref < lap length).
 *
 * If CMake fails on cuda/barrier and sm_70, reconfigure with
 *   -DMPPI_USE_CUDA_BARRIERS=OFF
 * or restrict architectures, e.g.  -DMPPI_CUDA_ARCH_LIST=86
 */
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>
#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace
{
constexpr int kMppiHorizon = 50;
constexpr int kSimSteps = 20000;
constexpr int kRefHorizon = kSimSteps + 256;
constexpr float kDt = 0.1F;
// Host rollout costs are Eigen::VectorXf; main limit is GPU memory, not a ~32k fixed-size Eigen cap.
constexpr int kNumRollouts = 1024 * 32;
constexpr float kVRef = 3.0F;

// Polyline for arc-length: vertices V0..V_{M-1} in order, then a closing edge V_{M-1} -> V0
constexpr int kMaxPoly = 800;
constexpr int kNumCtrl = 20;  // all distinct, CCW; do not repeat P0 at end of list (use mod in Catmull)
constexpr int kSamplesPerSpan = 16;  // more samples = smaller gap between last point and S/F

int g_nPoly = 0;  // number of vertices
float g_px[kMaxPoly] = {};
float g_py[kMaxPoly] = {};
// g_s_at_vertex[i] = arc length from S/F to vertex i. After build, g_px[M-1] == g_px[0] (start/finish).
float g_s_at_vertex[kMaxPoly] = {};
float g_perimeter = 1.0F;

__host__ inline void wrapAngle(float& a)
{
  const float two_pi = 2.0F * static_cast<float>(M_PI);
  while (a > static_cast<float>(M_PI))
  {
    a -= two_pi;
  }
  while (a < -static_cast<float>(M_PI))
  {
    a += two_pi;
  }
}

/** Uniform Catmull–Rom, t in [0,1) */
__host__ void catmullRom2d(float p0x, float p0y, float p1x, float p1y, float p2x, float p2y, float p3x, float p3y,
                            float t, float& outx, float& outy)
{
  const float t2 = t * t;
  const float t3 = t2 * t;
  outx = 0.5F * (2.0F * p1x + (-p0x + p2x) * t + (2.0F * p0x - 5.0F * p1x + 4.0F * p2x - p3x) * t2
                 + (-p0x + 3.0F * p1x - 3.0F * p2x + p3x) * t3);
  outy = 0.5F * (2.0F * p1y + (-p0y + p2y) * t + (2.0F * p0y - 5.0F * p1y + 4.0F * p2y - p3y) * t2
                 + (-p0y + 3.0F * p1y - 3.0F * p2y + p3y) * t3);
}

/** Wiggly closed "circuit" in the spirit of a long, flowing lap (Nordschleife-inspired, not a map).
 *  20 **distinct** CCW control points; periodic Catmull; **never** duplicate the first at the last index. */
__host__ void getControlPoint(int i, float& ox, float& oy)
{
  i = (i % kNumCtrl + kNumCtrl) % kNumCtrl;
  // Ellipse + harmonics: guaranteed simple closed control polygon, distinct points for all i.
  const float t = 2.0F * static_cast<float>(M_PI) * (static_cast<float>(i) / static_cast<float>(kNumCtrl)) - 0.4F;
  const float cxc = 280.0F;
  const float cyc = 255.0F;
  const float a0 = 228.0F;
  const float b0 = 188.0F;
  const float w = 1.0F + 0.14F * cosf(3.0F * t) + 0.1F * cosf(7.0F * t) + 0.11F * sinf(11.0F * t);
  // Slight "long straight" stretch in one leg (Döttinger-like)
  const float e = 1.0F + 0.38F * fmaxf(0.0F, cosf(2.3F * t));
  ox = cxc + a0 * w * e * cosf(t);
  oy = cyc + b0 * w * sinf(t);
}

__host__ void buildStylizedNordschleifeTrack()
{
  g_nPoly = 0;
  for (int seg = 0; seg < kNumCtrl; ++seg)
  {
    float p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y;
    getControlPoint(seg - 1, p0x, p0y);
    getControlPoint(seg, p1x, p1y);
    getControlPoint(seg + 1, p2x, p2y);
    getControlPoint(seg + 2, p3x, p3y);
    for (int k = 0; k < kSamplesPerSpan; ++k)
    {
      if (seg > 0 && k == 0)
      {
        continue;
      }
      if (g_nPoly >= kMaxPoly)
      {
        seg = kNumCtrl;
        break;
      }
      const float t = static_cast<float>(k) / static_cast<float>(kSamplesPerSpan);
      float ox, oy;
      catmullRom2d(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t, ox, oy);
      g_px[g_nPoly] = ox;
      g_py[g_nPoly] = oy;
      g_nPoly++;
    }
  }
  if (g_nPoly < 2)
  {
    g_nPoly = 0;
    g_perimeter = 1.0F;
    return;
  }
  // Geometrically close: last sample = S/F. No long chord: perimeter is the polyline + final edge to S/F.
  g_px[g_nPoly - 1] = g_px[0];
  g_py[g_nPoly - 1] = g_py[0];
  g_s_at_vertex[0] = 0.0F;
  for (int i = 1; i < g_nPoly; ++i)
  {
    const float dx = g_px[i] - g_px[i - 1];
    const float dy = g_py[i] - g_py[i - 1];
    g_s_at_vertex[i] = g_s_at_vertex[i - 1] + sqrtf(dx * dx + dy * dy);
  }
  g_perimeter = g_s_at_vertex[g_nPoly - 1];
  if (g_perimeter < 1.0F)
  {
    g_perimeter = 1.0F;
  }
}

__host__ float loopSEffective(float s)
{
  s = s - floorf(s / g_perimeter) * g_perimeter;
  if (s < 0.0F)
  {
    s += g_perimeter;
  }
  if (g_nPoly < 2)
  {
    return 0.0F;
  }
  return s;
}

__host__ void trackAtS(float s, float& x, float& y, float& psi)
{
  const float se = loopSEffective(s);
  const int M = g_nPoly;
  if (M < 2)
  {
    x = y = 0.0F;
    psi = 0.0F;
    return;
  }
  // g_px[M-1] == g_px[0] (S/F), g_s_at_vertex[M-1] == P. Edges: V0–V1 … V_{M-2}–V0, se in [0, P)
  int e = 0;
  for (; e < M - 1; ++e)
  {
    if (se < g_s_at_vertex[e + 1] - 1.0E-4F)
    {
      break;
    }
  }
  e = (std::min)(M - 2, (std::max)(0, e));
  const float sa = g_s_at_vertex[e];
  const float sb = g_s_at_vertex[e + 1];
  const float dlen = fmaxf(1.0E-3F, sb - sa);
  const float u = fminf(1.0F, fmaxf(0.0F, (se - sa) / dlen));
  x = g_px[e] * (1.0F - u) + g_px[e + 1] * u;
  y = g_py[e] * (1.0F - u) + g_py[e + 1] * u;
  const float dx = g_px[e + 1] - g_px[e];
  const float dy = g_py[e + 1] - g_py[e];
  psi = atan2f(dy, dx);
  wrapAngle(psi);
}

/** Write closed polyline (same as internal track) for matplotlib: <log_stem>_centerline.csv */
__host__ void writeTrackCenterlineForPlot(const std::string& log_csv_path)
{
  std::string out = log_csv_path;
  const size_t n = out.size();
  if (n > 4U && out.compare(n - 4U, 4U, ".csv") == 0)
  {
    out.insert(n - 4U, "_centerline");
  }
  else
  {
    out += "_centerline.csv";
  }
  std::ofstream f(out.c_str());
  if (!f)
  {
    std::cerr << "Could not write centerline: " << out << std::endl;
    return;
  }
  f << "x_m,y_m\n";
  for (int i = 0; i < g_nPoly; ++i)
  {
    f << g_px[i] << "," << g_py[i] << "\n";
  }
  f.close();
  std::cout << "Wrote closed centerline (for plot) to " << out << "  (perimeter " << g_perimeter << " m)\n";
}

__host__ void sAndVrefFromTime(const float time_s, float& s, float& v_ref)
{
  v_ref = kVRef;
  s = kVRef * time_s;
}

constexpr int o_pos_x = static_cast<int>(RacerDubinsParams::OutputIndex::BASELINK_POS_I_X);
constexpr int o_pos_y = static_cast<int>(RacerDubinsParams::OutputIndex::BASELINK_POS_I_Y);
constexpr int o_yaw = static_cast<int>(RacerDubinsParams::OutputIndex::YAW);
constexpr int o_v = static_cast<int>(RacerDubinsParams::OutputIndex::TOTAL_VELOCITY);

void fillTrackingWeights(QuadraticCostTrajectoryParams<RacerDubins, kRefHorizon>& p)
{
  for (int i = 0; i < RacerDubins::OUTPUT_DIM; ++i)
  {
    p.s_coeffs[i] = 0.0F;
  }
  p.s_coeffs[o_pos_x] = 1.0F;
  p.s_coeffs[o_pos_y] = 1.0F;
  p.s_coeffs[o_yaw] = 0.12F;
  p.s_coeffs[o_v] = 1.5F;
  // Quadratic *state* cost only; the rollout control penalty uses the Gaussian’s control_cost_coeff (set below).
  p.control_cost_coeff[static_cast<int>(RacerDubinsParams::ControlIndex::THROTTLE_BRAKE)] = 0.22F;
  p.control_cost_coeff[static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD)] = 0.48F;
}

void applyDemoRacerDubinsParams(RacerDubinsParams& p)
{
  p.c_0 = 0.0F;
  p.c_v[0] = 1.0F;
  p.c_t[0] = p.c_v[0] * kVRef;
  p.c_t[1] = p.c_t[0];
  p.c_t[2] = p.c_t[0];
  p.c_v[1] = p.c_v[0];
  p.c_v[2] = p.c_v[0];
}

void fillReferenceTrajectory(QuadraticCostTrajectoryParams<RacerDubins, kRefHorizon>& p)
{
  for (int t = 0; t < kRefHorizon; ++t)
  {
    for (int j = 0; j < RacerDubins::OUTPUT_DIM; ++j)
    {
      p.s_goal[t * RacerDubins::OUTPUT_DIM + j] = 0.0F;
    }
    const float time_s = t * kDt;
    float s = 0.0F;
    float v_ref = 0.0F;
    sAndVrefFromTime(time_s, s, v_ref);
    float xr, yr, pr;
    trackAtS(s, xr, yr, pr);
    p.s_goal[t * RacerDubins::OUTPUT_DIM + o_pos_x] = xr;
    p.s_goal[t * RacerDubins::OUTPUT_DIM + o_pos_y] = yr;
    p.s_goal[t * RacerDubins::OUTPUT_DIM + o_yaw] = pr;
    p.s_goal[t * RacerDubins::OUTPUT_DIM + o_v] = v_ref;
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
            << "  2D stylized Nordschleife-like closed loop (Catmull–Rom), CSV for:\n"
            << "  python3 examples/plot_racer_dubins_temporal_mppi.py <log.csv>\n"
            << "  (default: racer_dubins_racetrack_mppi_log.csv)\n";
}

int main(int argc, char** argv)
{
  buildStylizedNordschleifeTrack();

  std::string log_path = "racer_dubins_racetrack_mppi_log.csv";
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
  std::array<float2, DYN::CONTROL_DIM> u_rng{};
  u_rng[static_cast<int>(RacerDubinsParams::ControlIndex::THROTTLE_BRAKE)] = { -0.2F, 1.0F };
  u_rng[static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD)] = { -0.4F, 0.4F };
  model.setControlRanges(u_rng);

  COST cost;
  QuadraticCostTrajectoryParams<RacerDubins, kRefHorizon> cost_params;
  fillReferenceTrajectory(cost_params);
  fillTrackingWeights(cost_params);
  cost.setParams(cost_params);

  SAMPLER::SAMPLING_PARAMS_T sampler_params;
  sampler_params.std_dev[static_cast<int>(RacerDubinsParams::ControlIndex::THROTTLE_BRAKE)] = 0.12F;
  sampler_params.std_dev[static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD)] = 0.10F;
  for (int c = 0; c < DYN::CONTROL_DIM; ++c)
  {
    sampler_params.control_cost_coeff[c] = cost_params.control_cost_coeff[c];
  }
  SAMPLER sampler(sampler_params);

  FB feedback(&model, kDt);
  // Higher λ → stronger action cost in the MPPI exponential weights (smoother u, with less sampling noise).
  Mppi controller(&model, &cost, &feedback, &sampler, kDt, 1, 0.32F, 0.0F);
  {
    auto p = controller.getParams();
    p.dynamics_rollout_dim_ = dim3(32, 2, 1);
    p.cost_rollout_dim_ = dim3(32, 2, 1);
    controller.setParams(p);
  }

  model.GPUSetup();
  cost.GPUSetup();

  DYN::state_array x = model.getZeroState();
  {
    float xr, yr, pr;
    trackAtS(0.0F, xr, yr, pr);
    x(0) = 1.5F;
    x(1) = pr;
    x(2) = xr;
    x(3) = yr + 0.12F;
  }
  x(4) = 0.0F;
  x(5) = 0.0F;
  x(6) = 0.0F;

  DYN::state_array x_next = model.getZeroState();
  DYN::state_array xdot = model.getZeroState();
  DYN::output_array y = DYN::output_array::Zero();

  std::ofstream log(log_path.c_str());
  if (!log)
  {
    std::cerr << "Could not open log file: " << log_path << std::endl;
    return 1;
  }
  log << "t,pos_x,pos_y,yaw,vel_x,steer_angle,brake_state,u_throttle,u_steer,ref_x,ref_y,ref_yaw,ref_v,baseline\n";
  log << std::scientific;

  std::cout << "RacerDubins + temporal MPPI (Nordschleife-style loop, P=" << g_perimeter << " m, v=" << kVRef
            << " m/s, dt=" << kDt << ")\n";
  std::cout << "Logging to " << log_path << "\n";
  writeTrackCenterlineForPlot(log_path);

  {
    const int ri0 = 0;
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
    const int rrow = std::min(k + 1, kRefHorizon - 1);
    const int ri = rrow * RacerDubins::OUTPUT_DIM;
    const float ref_x = cost_params.s_goal[ri + o_pos_x];
    const float ref_y = cost_params.s_goal[ri + o_pos_y];
    const float ref_yaw = cost_params.s_goal[ri + o_yaw];
    const float ref_v = cost_params.s_goal[ri + o_v];
    log << t_end << "," << x_next(2) << "," << x_next(3) << "," << x_next(1) << "," << x_next(0) << "," << x_next(4)
        << "," << x_next(5) << "," << u(0) << "," << u(1) << "," << ref_x << "," << ref_y << "," << ref_yaw << ","
        << ref_v << "," << baseline << "\n";
    x = x_next;

    if (k % 20 == 0)
    {
      std::cout << "t=" << std::fixed << std::setprecision(2) << t_end << "  pos(" << x(2) << ", " << x(3) << ")"
                << "  ref(" << ref_x << ", " << ref_y << ")  v=" << x(0) << "\n";
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
  std::cout << "Plot: python3 examples/plot_racer_dubins_temporal_mppi.py " << log_path
            << "  (expect <log>_centerline.csv next to the log for a closed track overlay)\n";
  return 0;
}
