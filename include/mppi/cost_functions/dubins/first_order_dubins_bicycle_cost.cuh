/**
 * Analytic path-tracking cost for FirstOrderDubinsBicycle (reference polyline + parked-car OBBs).
 */
#pragma once

#ifndef MPPI_COST_FUNCTIONS_FIRST_ORDER_DUBINS_BICYCLE_COST_CUH_
#define MPPI_COST_FUNCTIONS_FIRST_ORDER_DUBINS_BICYCLE_COST_CUH_

#include <mppi/cost_functions/cost.cuh>
#include <mppi/dynamics/dubins/first_order_dubins_bicycle.cuh>

template <int NUM_TIMESTEPS>
struct FirstOrderDubinsBicycleCostParams : public CostParams<2>
{
  float desired_speed = 2.5F;
  float speed_coeff = 800.0F;
  float track_coeff = 300.0F;
  float crash_coeff = 10000.0F;
  float boundary_threshold = 0.8F;
  float accel_cmd_coeff = 0.0F;
  float steer_cmd_coeff = 0.0F;
  float lateral_acceleration_coeff = 10.0F;
  float lateral_jerk_coeff = 25.0F;
  float longitudinal_jerk_coeff = 10.0F;
  float wheel_base = 0.32F;
  float accel_time_constant = 0.15F;
  float steer_time_constant = 0.08F;
  /** Ego OBB for parked-car collision (rear axle at pose; box center offset forward). */
  float ego_length = 0.55F * 1.5F;
  float ego_width = 0.28F * 1.5F;
  float ego_axle_to_box_center = 0.2F;
};

template <class CLASS_T, int NUM_TIMESTEPS,
          class PARAMS_T = FirstOrderDubinsBicycleCostParams<NUM_TIMESTEPS>,
          class DYN_PARAMS_T = FirstOrderDubinsBicycleParams>
class FirstOrderDubinsBicycleCostImpl : public Cost<CLASS_T, PARAMS_T, DYN_PARAMS_T>
{
public:
  static constexpr int kMaxObstacles = 64;

  using PARENT_CLASS = Cost<CLASS_T, PARAMS_T, DYN_PARAMS_T>;
  using output_array = typename PARENT_CLASS::output_array;
  using control_array = typename PARENT_CLASS::control_array;

  FirstOrderDubinsBicycleCostImpl(cudaStream_t stream = 0);

  void paramsToDevice();

  void setReferenceTrajectory(const float* x, const float* y, int count);

  void setOrientedBoxObstacles(const float* x, const float* y, const float* yaw, const float* half_length,
                               const float* half_width, int count);

  __host__ __device__ float computeTrackValue(float x, float y) const;

  __host__ __device__ bool egoIntersectsParkedCar(const float x, const float y, const float yaw) const;

  float computeStateCost(const Eigen::Ref<const output_array>& y, int timestep, int* crash_status);

  __device__ float computeStateCost(float* y, int timestep, float* theta_c, int* crash_status);

  float computeControlCost(const Eigen::Ref<const control_array>& u, int timestep, int* crash);

  __device__ float computeControlCost(float* u, int timestep, float* theta_c, int* crash);

  float computeComfortCost(const Eigen::Ref<const control_array>& u, const Eigen::Ref<const output_array>& y,
                           int timestep);

  __device__ float computeComfortCost(float* u, float* y, int timestep);

  __device__ float terminalCost(float* y, float* theta_c);

  float computeRunningCost(const Eigen::Ref<const output_array>& y, const Eigen::Ref<const control_array>& u,
                           int timestep, int* crash);

  __device__ float computeRunningCost(float* y, float* u, int timestep, float* theta_c, int* crash);

  float ref_x_[NUM_TIMESTEPS] = {};
  float ref_y_[NUM_TIMESTEPS] = {};
  int num_obstacles_ = 0;
  float obs_x_[kMaxObstacles] = {};
  float obs_y_[kMaxObstacles] = {};
  float obs_yaw_[kMaxObstacles] = {};
  float obs_half_length_[kMaxObstacles] = {};
  float obs_half_width_[kMaxObstacles] = {};

private:
  void dataToDevice();
};

template <int NUM_TIMESTEPS>
class FirstOrderDubinsBicycleCost
    : public FirstOrderDubinsBicycleCostImpl<FirstOrderDubinsBicycleCost<NUM_TIMESTEPS>, NUM_TIMESTEPS>
{
public:
  FirstOrderDubinsBicycleCost(cudaStream_t stream = 0)
    : FirstOrderDubinsBicycleCostImpl<FirstOrderDubinsBicycleCost<NUM_TIMESTEPS>, NUM_TIMESTEPS>(stream)
  {
  }
};

#if __CUDACC__
#include "first_order_dubins_bicycle_cost.cu"
#endif

#endif  // MPPI_COST_FUNCTIONS_FIRST_ORDER_DUBINS_BICYCLE_COST_CUH_
