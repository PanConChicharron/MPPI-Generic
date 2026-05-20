#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>
#include <mppi/utils/angle_utils.cuh>

template <class CLASS_T, class DYN_T, class PARAMS_T>
QuadraticCostImpl<CLASS_T, DYN_T, PARAMS_T>::QuadraticCostImpl(cudaStream_t stream)
{
  this->bindToStream(stream);
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
float QuadraticCostImpl<CLASS_T, DYN_T, PARAMS_T>::computeStateCost(const Eigen::Ref<const output_array> s,
                                                                    int timestep, int* crash_status)
{
  float cost = 0;
  output_array des_state = this->params_.getDesiredState(timestep);
  const int yaw_i = this->params_.yaw_wrapped_abs_cost_index;

  for (int i = 0; i < DYN_T::OUTPUT_DIM; i++)
  {
    if (yaw_i >= 0 && yaw_i < DYN_T::OUTPUT_DIM && i == yaw_i)
    {
      const float e = angle_utils::shortestAngularDistance(des_state(i), s(i));
      cost += this->params_.s_coeffs[i] * fabsf(e);
    }
    else
    {
      const float e = s(i) - des_state(i);
      cost += this->params_.s_coeffs[i] * e * e;
    }
  }

  return cost;
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
float QuadraticCostImpl<CLASS_T, DYN_T, PARAMS_T>::terminalCost(const Eigen::Ref<const output_array> s)
{
  return 0.0;
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
__device__ float QuadraticCostImpl<CLASS_T, DYN_T, PARAMS_T>::computeStateCost(float* s, int timestep, float* theta_c,
                                                                               int* crash_status)
{
  float cost = 0;

  float* desired_state = this->params_.getGoalStatePointer(timestep);
  const int yaw_i = this->params_.yaw_wrapped_abs_cost_index;

  for (int i = 0; i < DYN_T::OUTPUT_DIM; i++)
  {
    if (yaw_i >= 0 && yaw_i < DYN_T::OUTPUT_DIM && i == yaw_i)
    {
      const float e = angle_utils::shortestAngularDistance(desired_state[i], s[i]);
      cost += fabsf(e) * this->params_.s_coeffs[i];
    }
    else
    {
      cost += powf(s[i] - desired_state[i], 2) * this->params_.s_coeffs[i];
    }
  }

  return cost;
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
__device__ float QuadraticCostImpl<CLASS_T, DYN_T, PARAMS_T>::terminalCost(float* s, float* theta_c)
{
  return 0.0;
}
