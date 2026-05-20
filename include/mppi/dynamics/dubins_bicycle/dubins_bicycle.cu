#include <mppi/dynamics/dubins_bicycle/dubins_bicycle.cuh>

#include <cmath>

namespace
{
using S = DubinsBicycleParams::StateIndex;
using C = DubinsBicycleParams::ControlIndex;
using O = DubinsBicycleParams::OutputIndex;

__host__ __device__ void dubinsBicycleDeriv(const DubinsBicycleParams& p, const float* state, const float* control,
                                            float* state_der)
{
  const float v = state[static_cast<int>(S::VEL_X)];
  const float yaw = state[static_cast<int>(S::YAW)];
  const float steer = state[static_cast<int>(S::STEER_ANGLE)];
  const float accel = control[static_cast<int>(C::ACCEL)];
  const float steer_cmd = control[static_cast<int>(C::STEER)];

  state_der[static_cast<int>(S::VEL_X)] = accel;
  state_der[static_cast<int>(S::YAW)] = (v / p.wheel_base) * tanf(steer);
  float sin_yaw = 0.0F;
  float cos_yaw = 0.0F;
  sincosf(yaw, &sin_yaw, &cos_yaw);
  state_der[static_cast<int>(S::POS_X)] = v * cos_yaw;
  state_der[static_cast<int>(S::POS_Y)] = v * sin_yaw;

  float steer_dot = (steer_cmd - steer) / std::max(p.steer_time_constant, 1.0E-4F);
  steer_dot = fmaxf(fminf(steer_dot, p.max_steer_rate), -p.max_steer_rate);
  state_der[static_cast<int>(S::STEER_ANGLE)] = steer_dot;
}
}  // namespace

DubinsBicycle::DubinsBicycle(cudaStream_t stream) : Dynamics<DubinsBicycle, DubinsBicycleParams>(stream)
{
}

DubinsBicycle::DubinsBicycle(DubinsBicycleParams& params, cudaStream_t stream)
  : Dynamics<DubinsBicycle, DubinsBicycleParams>(params, stream)
{
}

void DubinsBicycle::computeDynamics(const Eigen::Ref<const state_array>& state,
                                    const Eigen::Ref<const control_array>& control, Eigen::Ref<state_array> state_der)
{
  dubinsBicycleDeriv(this->params_, state.data(), control.data(), state_der.data());
}

bool DubinsBicycle::computeGrad(const Eigen::Ref<const state_array>&, const Eigen::Ref<const control_array>&,
                                Eigen::Ref<dfdx>, Eigen::Ref<dfdu>)
{
  return false;
}

void DubinsBicycle::updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
                                  Eigen::Ref<state_array> state_der, const float dt)
{
  next_state = state + state_der * dt;
  next_state(static_cast<int>(S::YAW)) = angle_utils::normalizeAngle(next_state(static_cast<int>(S::YAW)));
  next_state(static_cast<int>(S::STEER_ANGLE)) =
      fmaxf(fminf(next_state(static_cast<int>(S::STEER_ANGLE)), this->params_.max_steer_angle),
            -this->params_.max_steer_angle);
}

void DubinsBicycle::stateToOutput(const Eigen::Ref<const state_array>& state, Eigen::Ref<output_array> output)
{
  stateToOutput(state.data(), output.data());
}

__device__ void DubinsBicycle::computeDynamics(float* state, float* control, float* state_der, float*)
{
  dubinsBicycleDeriv(this->params_, state, control, state_der);
}

__device__ void DubinsBicycle::updateState(float* state, float* next_state, float* state_der, const float dt)
{
  for (int i = threadIdx.y; i < STATE_DIM; i += blockDim.y)
  {
    next_state[i] = state[i] + state_der[i] * dt;
    if (i == static_cast<int>(S::YAW))
    {
      next_state[i] = angle_utils::normalizeAngle(next_state[i]);
    }
    if (i == static_cast<int>(S::STEER_ANGLE))
    {
      next_state[i] = fmaxf(fminf(next_state[i], this->params_.max_steer_angle), -this->params_.max_steer_angle);
    }
  }
}

__host__ __device__ void DubinsBicycle::stateToOutput(const float* state, float* output)
{
  output[static_cast<int>(O::POS_X)] = state[static_cast<int>(S::POS_X)];
  output[static_cast<int>(O::POS_Y)] = state[static_cast<int>(S::POS_Y)];
  output[static_cast<int>(O::YAW)] = state[static_cast<int>(S::YAW)];
  output[static_cast<int>(O::VEL_X)] = state[static_cast<int>(S::VEL_X)];
  output[static_cast<int>(O::STEER_ANGLE)] = state[static_cast<int>(S::STEER_ANGLE)];
}

DubinsBicycle::state_array DubinsBicycle::stateFromMap(const std::map<std::string, float>& map)
{
  state_array x = state_array::Zero();
  const auto set_if = [&map, &x](const char* key, const int idx) {
    const auto it = map.find(key);
    if (it != map.end())
    {
      x(idx) = it->second;
    }
  };
  set_if("vel_x", static_cast<int>(S::VEL_X));
  set_if("yaw", static_cast<int>(S::YAW));
  set_if("pos_x", static_cast<int>(S::POS_X));
  set_if("pos_y", static_cast<int>(S::POS_Y));
  set_if("steer", static_cast<int>(S::STEER_ANGLE));
  return x;
}
