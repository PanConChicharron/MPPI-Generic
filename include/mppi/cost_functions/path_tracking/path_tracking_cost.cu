#include <mppi/cost_functions/path_tracking/path_tracking_cost.cuh>

#include <cmath>

namespace
{
using O = DubinsBicycleParams::OutputIndex;
using C = DubinsBicycleParams::ControlIndex;

template <int REF_HORIZON>
__host__ __device__ float pathTrackingStateCost(const PathTrackingCostParams<REF_HORIZON>& params, const float* y,
                                                const int timestep)
{
  const float* goal = params.goalAt(timestep);
  const float dx = y[static_cast<int>(O::POS_X)] - goal[static_cast<int>(O::POS_X)];
  const float dy = y[static_cast<int>(O::POS_Y)] - goal[static_cast<int>(O::POS_Y)];
  const float dtheta =
      angle_utils::shortestAngularDistance(goal[static_cast<int>(O::YAW)], y[static_cast<int>(O::YAW)]);
  const float dv = y[static_cast<int>(O::VEL_X)] - goal[static_cast<int>(O::VEL_X)];

  return params.w_pos * (dx * dx + dy * dy) + params.w_heading_so2 * (1.0F - cosf(dtheta)) +
         params.w_vel * (dv * dv);
}

/**
 * Path curvature K = tan(δ)/L and K_dot from first-order steer dynamics.
 * a_L = v^2 * K
 * j_L = v^2 * K_dot + 3 * v * a * K   (a = longitudinal acceleration)
 */
/** δ_dot from first-order steering: (u_steer - δ) / τ. */
template <int REF_HORIZON>
__host__ __device__ float pathTrackingSteerDot(const PathTrackingCostParams<REF_HORIZON>& params, const float steer,
                                               const float u_steer)
{
  return (u_steer - steer) / fmaxf(params.steer_time_constant, 1.0E-4F);
}

template <int REF_HORIZON>
__host__ __device__ void pathTrackingCurvatureAndDeriv(const PathTrackingCostParams<REF_HORIZON>& params,
                                                        const float steer, const float u_steer, float& K, float& K_dot)
{
  const float L = params.wheel_base;
  const float inv_L = 1.0F / L;
  const float tan_steer = tanf(steer);
  const float sec2_steer = 1.0F + tan_steer * tan_steer;
  K = inv_L * tan_steer;
  const float steer_dot = pathTrackingSteerDot(params, steer, u_steer);
  K_dot = inv_L * sec2_steer * steer_dot;
}

template <int REF_HORIZON>
__host__ __device__ float pathTrackingComfortCost(const PathTrackingCostParams<REF_HORIZON>& params, const float* y,
                                                 const float* u)
{
  if (params.w_lat_accel <= 0.0F && params.w_lat_jerk <= 0.0F && params.w_steer_dot <= 0.0F)
  {
    return 0.0F;
  }

  const float steer = y[static_cast<int>(O::STEER_ANGLE)];
  const float u_steer = u[static_cast<int>(C::STEER)];
  const float steer_dot = pathTrackingSteerDot(params, steer, u_steer);

  float cost = 0.0F;
  if (params.w_steer_dot > 0.0F)
  {
    cost += params.w_steer_dot * steer_dot * steer_dot;
  }

  if (params.w_lat_accel <= 0.0F && params.w_lat_jerk <= 0.0F)
  {
    return cost;
  }

  if (params.wheel_base < 1.0E-6F)
  {
    return cost;
  }

  const float v = y[static_cast<int>(O::VEL_X)];
  const float a_lon = u[static_cast<int>(C::ACCEL)];

  float K = 0.0F;
  float K_dot = 0.0F;
  pathTrackingCurvatureAndDeriv(params, steer, u_steer, K, K_dot);

  const float a_lat = v * v * K;
  const float j_lat = v * v * K_dot + 3.0F * v * a_lon * K;

  if (params.w_lat_accel > 0.0F)
  {
    cost += params.w_lat_accel * a_lat * a_lat;
  }
  if (params.w_lat_jerk > 0.0F)
  {
    cost += params.w_lat_jerk * j_lat * j_lat;
  }
  return cost;
}

template <int REF_HORIZON>
__host__ __device__ float pathTrackingRunningCost(const PathTrackingCostParams<REF_HORIZON>& params, const float* y,
                                                 const float* u, const int timestep)
{
  return pathTrackingStateCost(params, y, timestep) + pathTrackingComfortCost(params, y, u);
}

/** Actuation effort: sum_i w_i * u_i^2 using control_cost_coeff (independent of MPPI sampler IS term). */
template <int REF_HORIZON>
__host__ __device__ float pathTrackingControlCost(const PathTrackingCostParams<REF_HORIZON>& params, const float* u)
{
  float cost = 0.0F;
  for (int i = 0; i < DubinsBicycle::CONTROL_DIM; ++i)
  {
    cost += params.control_cost_coeff[i] * u[i] * u[i];
  }
  return cost;
}

}  // namespace

template <class CLASS_T, int REF_HORIZON>
PathTrackingCostImpl<CLASS_T, REF_HORIZON>::PathTrackingCostImpl(cudaStream_t stream)
{
  this->bindToStream(stream);
}

template <class CLASS_T, int REF_HORIZON>
float PathTrackingCostImpl<CLASS_T, REF_HORIZON>::computeStateCost(const Eigen::Ref<const output_array>& y,
                                                                   const int timestep, int* crash_status)
{
  (void)crash_status;
  return pathTrackingStateCost(this->params_, y.data(), timestep);
}

template <class CLASS_T, int REF_HORIZON>
float PathTrackingCostImpl<CLASS_T, REF_HORIZON>::terminalCost(const Eigen::Ref<const output_array>& y)
{
  (void)y;
  return 0.0F;
}

template <class CLASS_T, int REF_HORIZON>
float PathTrackingCostImpl<CLASS_T, REF_HORIZON>::computeControlCost(const Eigen::Ref<const control_array>& u,
                                                                     const int timestep, int* crash)
{
  (void)timestep;
  (void)crash;
  return pathTrackingControlCost(this->params_, u.data());
}

template <class CLASS_T, int REF_HORIZON>
float PathTrackingCostImpl<CLASS_T, REF_HORIZON>::computeRunningCost(const Eigen::Ref<const output_array>& y,
                                                                     const Eigen::Ref<const control_array>& u,
                                                                     const int timestep, int* crash)
{
  float cost = pathTrackingRunningCost(this->params_, y.data(), u.data(), timestep);
  cost += this->computeControlCost(u, timestep, crash);
  return cost;
}

template <class CLASS_T, int REF_HORIZON>
__device__ float PathTrackingCostImpl<CLASS_T, REF_HORIZON>::computeStateCost(float* y, const int timestep, float*,
                                                                              int* crash_status)
{
  (void)crash_status;
  return pathTrackingStateCost(this->params_, y, timestep);
}

template <class CLASS_T, int REF_HORIZON>
__device__ float PathTrackingCostImpl<CLASS_T, REF_HORIZON>::terminalCost(float*, float*)
{
  return 0.0F;
}

template <class CLASS_T, int REF_HORIZON>
__device__ float PathTrackingCostImpl<CLASS_T, REF_HORIZON>::computeControlCost(float* u, const int timestep,
                                                                                float* theta_c, int* crash)
{
  (void)timestep;
  (void)theta_c;
  (void)crash;
  return pathTrackingControlCost(this->params_, u);
}

template <class CLASS_T, int REF_HORIZON>
__device__ float PathTrackingCostImpl<CLASS_T, REF_HORIZON>::computeRunningCost(float* y, float* u, const int timestep,
                                                                                float* theta_c, int* crash)
{
  float cost = pathTrackingRunningCost(this->params_, y, u, timestep);
  cost += this->computeControlCost(u, timestep, theta_c, crash);
  return cost;
}
