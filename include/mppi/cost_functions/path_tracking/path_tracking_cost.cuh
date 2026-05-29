/**
 * Temporal path-tracking cost for DubinsBicycle MPPI.
 *
 * Penalizes (per rollout timestep):
 *   - position error vs reference (x, y)
 *   - heading error on SO(2): w * (1 - cos(Δθ))
 *   - longitudinal speed error vs v_ref
 *   - lateral acceleration: w * a_L^2,  a_L = v^2 * K,  K = tan(δ)/L (curvature from actual steer)
 *   - lateral jerk: w * j_L^2,  j_L = v^2*K_dot + 3*v*a*K,  K_dot = (sec^2(δ)/L)*(u_steer-δ)/τ
 *   - steering rate: w * δ_dot^2,  δ_dot = (u_steer - δ) / τ  (first-order steer dynamics)
 *   - control effort (accel, steer) via control_cost_coeff
 */
#pragma once

#include <mppi/cost_functions/cost.cuh>
#include <mppi/dynamics/dubins_bicycle/dubins_bicycle.cuh>
#include <mppi/utils/angle_utils.cuh>

template <int REF_HORIZON>
struct PathTrackingCostParams : public CostParams<DubinsBicycle::CONTROL_DIM>
{
  static const int TIME_HORIZON = REF_HORIZON;
  static const int OUTPUT_DIM = DubinsBicycle::OUTPUT_DIM;

  float w_pos = 10.0F;
  float w_heading_so2 = 1.0F;
  float w_vel = 5.0F;
  float w_lat_accel = 10.0F;
  float w_lat_jerk = 100.0F;
  float w_steer_dot = 100.0F;

  float wheel_base = 0.32F;
  float steer_time_constant = 0.08F;

  // This is cleanly overwritten via fillCostFromPathReference every step on host
  float s_goal[OUTPUT_DIM * REF_HORIZON] = { 0 };

  PathTrackingCostParams()
  {
    control_cost_coeff[static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)] = 0.1F;
    control_cost_coeff[static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)] = 0.2F;
  }

  __host__ __device__ int goalIndex(const int timestep) const
  {
    // Clamp the local rollout lookahead timestep safely within the allocated window
    int idx = timestep;
    if (idx < 0)
    {
      idx = 0;
    }
    else if (idx >= TIME_HORIZON)
    {
      idx = TIME_HORIZON - 1;
    }
    return idx * OUTPUT_DIM;
  }

  __host__ __device__ const float* goalAt(const int timestep) const
  {
    return s_goal + goalIndex(timestep);
  }
};

template <class CLASS_T, int REF_HORIZON>
class PathTrackingCostImpl : public Cost<CLASS_T, PathTrackingCostParams<REF_HORIZON>, DubinsBicycleParams>
{
public:
  using PARENT_CLASS = Cost<CLASS_T, PathTrackingCostParams<REF_HORIZON>, DubinsBicycleParams>;
  using output_array = typename PARENT_CLASS::output_array;
  using control_array = typename PARENT_CLASS::control_array;
  static constexpr float MAX_COST_VALUE = 1e16;

  PathTrackingCostImpl(cudaStream_t stream = nullptr);

  __device__ void initializeCosts(float* output, float* control, float* theta_c, float t_0, float dt);

  std::string getCostFunctionName() const override
  {
    return "Path tracking cost (SO2 heading + lat accel/jerk + steer rate)";
  }

  float computeStateCost(const Eigen::Ref<const output_array>& y, int timestep = 0, int* crash_status = nullptr);

  float terminalCost(const Eigen::Ref<const output_array>& y);

  float computeRunningCost(const Eigen::Ref<const output_array>& y, const Eigen::Ref<const control_array>& u,
                           int timestep, int* crash);

  /** Quadratic penalty sum_i control_cost_coeff[i] * u[i]^2 (base Cost::computeControlCost returns 0). */
  float computeControlCost(const Eigen::Ref<const control_array>& u, int timestep, int* crash);

  __device__ float computeStateCost(float* y, int timestep = 0, float* theta_c = nullptr, int* crash_status = nullptr);

  __device__ float terminalCost(float* y, float* theta_c);

  __device__ float computeRunningCost(float* y, float* u, int timestep, float* theta_c, int* crash);

  __device__ float computeControlCost(float* u, int timestep, float* theta_c, int* crash);
};

#if __CUDACC__
#include "path_tracking_cost.cu"
#endif

template <int REF_HORIZON>
class PathTrackingCost : public PathTrackingCostImpl<PathTrackingCost<REF_HORIZON>, REF_HORIZON>
{
public:
  PathTrackingCost(cudaStream_t stream = nullptr)
    : PathTrackingCostImpl<PathTrackingCost<REF_HORIZON>, REF_HORIZON>(stream)
  {
  }
};
