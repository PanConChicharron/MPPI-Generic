/**
 * Kinematic bicycle for path-tracking demos.
 * State: longitudinal speed, yaw, position, steer angle.
 * Controls: longitudinal acceleration [m/s^2], steer angle command [rad].
 */
#pragma once

#ifndef MPPIGENERIC_DUBINS_BICYCLE_CUH
#define MPPIGENERIC_DUBINS_BICYCLE_CUH

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/utils/angle_utils.cuh>

struct DubinsBicycleParams : public DynamicsParams
{
  enum class StateIndex : int
  {
    VEL_X = 0,
    YAW,
    POS_X,
    POS_Y,
    STEER_ANGLE,
    NUM_STATES
  };

  enum class ControlIndex : int
  {
    ACCEL = 0,
    STEER,
    NUM_CONTROLS
  };

  enum class OutputIndex : int
  {
    POS_X = 0,
    POS_Y,
    YAW,
    VEL_X,
    /** Actual steer angle δ (state), not the steer command. */
    STEER_ANGLE,
    NUM_OUTPUTS
  };

  float wheel_base = 0.32F;
  /** First-order steering: steer_dot = (u_steer - steer) / steer_time_constant */
  float steer_time_constant = 0.08F;
  float max_steer_angle = 0.45F;
  float max_steer_rate = 3.0F;
  float max_accel = 4.0F;
  float min_accel = -6.0F;
};

using namespace MPPI_internal;

class DubinsBicycle : public Dynamics<DubinsBicycle, DubinsBicycleParams>
{
public:
  using PARENT_CLASS = Dynamics<DubinsBicycle, DubinsBicycleParams>;
  using PARENT_CLASS::updateState;

  DubinsBicycle(cudaStream_t stream = nullptr);

  DubinsBicycle(DubinsBicycleParams& params, cudaStream_t stream = nullptr);

  std::string getDynamicsModelName() const override
  {
    return "Dubins Bicycle (accel + steer)";
  }

  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                       Eigen::Ref<state_array> state_der);

  bool computeGrad(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                   Eigen::Ref<dfdx> A, Eigen::Ref<dfdu> B);

  void updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
                   Eigen::Ref<state_array> state_der, const float dt);

  void stateToOutput(const Eigen::Ref<const state_array>& state, Eigen::Ref<output_array> output);

  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta = nullptr);

  __device__ void updateState(float* state, float* next_state, float* state_der, const float dt);

  __host__ __device__ void stateToOutput(const float* state, float* output);

  state_array stateFromMap(const std::map<std::string, float>& map) override;
};

#if __CUDACC__
#include "dubins_bicycle.cu"
#endif

#endif  // MPPIGENERIC_DUBINS_BICYCLE_CUH
