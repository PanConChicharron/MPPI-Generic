/**
 * Analytic path-tracking cost for RacerDubins (reference trajectory + obstacles, no costmap image).
 */
#pragma once

#ifndef MPPI_COST_FUNCTIONS_RACER_COST_CUH_
#define MPPI_COST_FUNCTIONS_RACER_COST_CUH_

#include <mppi/cost_functions/cost.cuh>
#include <mppi/dynamics/racer_dubins/racer_dubins.cuh>

template <int NUM_TIMESTEPS>
struct RacerCostParams : public CostParams<2>
{
  float desired_speed = 2.5F;
  float speed_coeff = 20.0F;
  float track_coeff = 500.0F;
  float crash_coeff = 10000.0F;
  float boundary_threshold = 0.8F;
  float steer_coeff = 50.0F;
  float track_dist_scale = 0.4F;  // world distance to track_val; use ppm / 25.0 for costmap parity
  float track_cost_cap = 0.75F;
};

template <class CLASS_T, int NUM_TIMESTEPS, class PARAMS_T = RacerCostParams<NUM_TIMESTEPS>,
          class DYN_PARAMS_T = RacerDubinsParams>
class RacerCostImpl : public Cost<CLASS_T, PARAMS_T, DYN_PARAMS_T>
{
public:
  static constexpr int kMaxObstacles = 64;

  using PARENT_CLASS = Cost<CLASS_T, PARAMS_T, DYN_PARAMS_T>;
  using output_array = typename PARENT_CLASS::output_array;
  using control_array = typename PARENT_CLASS::control_array;

  RacerCostImpl(cudaStream_t stream = 0);

  void paramsToDevice();

  void setReferenceTrajectory(const float* x, const float* y, int count);

  void setObstacles(const float* x, const float* y, const float* r, int count);

  __host__ __device__ float computeTrackValue(float x, float y) const;

  __device__ float computeStateCost(float* y, int timestep, float* theta_c, int* crash_status);

  float computeStateCost(const Eigen::Ref<const output_array>& y, int timestep, int* crash_status);

  __device__ float computeControlCost(float* u, int timestep, float* theta_c, int* crash);

  float computeControlCost(const Eigen::Ref<const control_array>& u, int timestep, int* crash);

  __device__ float terminalCost(float* y, float* theta_c);

  float ref_x_[NUM_TIMESTEPS] = {};
  float ref_y_[NUM_TIMESTEPS] = {};
  int num_obstacles_ = 0;
  float obs_x_[kMaxObstacles] = {};
  float obs_y_[kMaxObstacles] = {};
  float obs_r_[kMaxObstacles] = {};

private:
  void dataToDevice();
};

template <int NUM_TIMESTEPS>
class RacerCost : public RacerCostImpl<RacerCost<NUM_TIMESTEPS>, NUM_TIMESTEPS>
{
public:
  RacerCost(cudaStream_t stream = 0) : RacerCostImpl<RacerCost<NUM_TIMESTEPS>, NUM_TIMESTEPS>(stream)
  {
  }
};

#if __CUDACC__
#include "racer_cost.cu"
#endif

#endif  // MPPI_COST_FUNCTIONS_RACER_COST_CUH_
