#include <mppi/cost_functions/racer/racer_cost.cuh>

#include <algorithm>
#include <cmath>

namespace
{
#ifdef __CUDA_ARCH__
__device__ float clampUnitInterval(const float t)
{
  return fmaxf(0.0F, fminf(1.0F, t));
}

__device__ float vectorLength(const float dx, const float dy)
{
  return sqrtf(dx * dx + dy * dy);
}
#else
inline float clampUnitInterval(const float t)
{
  return std::max(0.0F, std::min(1.0F, t));
}

inline float vectorLength(const float dx, const float dy)
{
  return std::sqrt(dx * dx + dy * dy);
}
#endif

__host__ __device__ float distancePointToSegment(const float px, const float py, const float x0, const float y0,
                                                 const float x1, const float y1)
{
  const float dx = x1 - x0;
  const float dy = y1 - y0;
  const float len_sq = dx * dx + dy * dy;
  if (len_sq < 1.0E-8F)
  {
    return vectorLength(px - x0, py - y0);
  }

  const float t = clampUnitInterval(((px - x0) * dx + (py - y0) * dy) / len_sq);
  return vectorLength(px - (x0 + t * dx), py - (y0 + t * dy));
}
}  // namespace

template <class CLASS_T, int NUM_TIMESTEPS, class PARAMS_T, class DYN_PARAMS_T>
RacerCostImpl<CLASS_T, NUM_TIMESTEPS, PARAMS_T, DYN_PARAMS_T>::RacerCostImpl(cudaStream_t stream)
{
  this->bindToStream(stream);
}

template <class CLASS_T, int NUM_TIMESTEPS, class PARAMS_T, class DYN_PARAMS_T>
void RacerCostImpl<CLASS_T, NUM_TIMESTEPS, PARAMS_T, DYN_PARAMS_T>::paramsToDevice()
{
  PARENT_CLASS::paramsToDevice();
}

template <class CLASS_T, int NUM_TIMESTEPS, class PARAMS_T, class DYN_PARAMS_T>
void RacerCostImpl<CLASS_T, NUM_TIMESTEPS, PARAMS_T, DYN_PARAMS_T>::dataToDevice()
{
  if (!this->GPUMemStatus_)
  {
    return;
  }

  HANDLE_ERROR(cudaMemcpyAsync(this->cost_d_->ref_x_, ref_x_, sizeof(ref_x_), cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(this->cost_d_->ref_y_, ref_y_, sizeof(ref_y_), cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(&this->cost_d_->num_obstacles_, &num_obstacles_, sizeof(num_obstacles_),
                               cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(this->cost_d_->obs_x_, obs_x_, sizeof(obs_x_), cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(this->cost_d_->obs_y_, obs_y_, sizeof(obs_y_), cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(this->cost_d_->obs_r_, obs_r_, sizeof(obs_r_), cudaMemcpyHostToDevice, this->stream_));
}

template <class CLASS_T, int NUM_TIMESTEPS, class PARAMS_T, class DYN_PARAMS_T>
void RacerCostImpl<CLASS_T, NUM_TIMESTEPS, PARAMS_T, DYN_PARAMS_T>::setReferenceTrajectory(const float* x,
                                                                                             const float* y,
                                                                                             const int count)
{
  const int n = std::max(0, std::min(count, NUM_TIMESTEPS));
  for (int i = 0; i < n; ++i)
  {
    ref_x_[i] = x[i];
    ref_y_[i] = y[i];
  }
  if (n > 0)
  {
    for (int i = n; i < NUM_TIMESTEPS; ++i)
    {
      ref_x_[i] = x[n - 1];
      ref_y_[i] = y[n - 1];
    }
  }
  dataToDevice();
}

template <class CLASS_T, int NUM_TIMESTEPS, class PARAMS_T, class DYN_PARAMS_T>
void RacerCostImpl<CLASS_T, NUM_TIMESTEPS, PARAMS_T, DYN_PARAMS_T>::setObstacles(const float* x, const float* y,
                                                                                   const float* r, const int count)
{
  const int n = std::max(0, std::min(count, kMaxObstacles));
  num_obstacles_ = n;
  for (int i = 0; i < n; ++i)
  {
    obs_x_[i] = x[i];
    obs_y_[i] = y[i];
    obs_r_[i] = r[i];
  }
  dataToDevice();
}

template <class CLASS_T, int NUM_TIMESTEPS, class PARAMS_T, class DYN_PARAMS_T>
__host__ __device__ float RacerCostImpl<CLASS_T, NUM_TIMESTEPS, PARAMS_T, DYN_PARAMS_T>::computeTrackValue(float x,
                                                                                                           float y) const
{
  float min_dist = 0.0F;
  if (NUM_TIMESTEPS <= 1)
  {
    min_dist = vectorLength(x - ref_x_[0], y - ref_y_[0]);
  }
  else
  {
    min_dist = 1.0E8F;
    for (int i = 0; i < NUM_TIMESTEPS - 1; ++i)
    {
      const float segment_dist =
          distancePointToSegment(x, y, ref_x_[i], ref_y_[i], ref_x_[i + 1], ref_y_[i + 1]);
#ifdef __CUDA_ARCH__
      min_dist = fminf(min_dist, segment_dist);
#else
      min_dist = std::min(min_dist, segment_dist);
#endif
    }
  }

  float track_val = min_dist * this->params_.track_dist_scale;
#ifdef __CUDA_ARCH__
  track_val = fminf(track_val, this->params_.track_cost_cap);
#else
  track_val = std::min(track_val, this->params_.track_cost_cap);
#endif

  for (int i = 0; i < num_obstacles_; ++i)
  {
    const float obs_dx = x - obs_x_[i];
    const float obs_dy = y - obs_y_[i];
    const float obs_r = obs_r_[i];
    if (obs_dx * obs_dx + obs_dy * obs_dy < obs_r * obs_r)
    {
      track_val = 1.0F;
      break;
    }
  }

  return track_val;
}

template <class CLASS_T, int NUM_TIMESTEPS, class PARAMS_T, class DYN_PARAMS_T>
__device__ float RacerCostImpl<CLASS_T, NUM_TIMESTEPS, PARAMS_T, DYN_PARAMS_T>::computeStateCost(float* y, int timestep,
                                                                                                 float* theta_c,
                                                                                                 int* crash_status)
{
  (void)theta_c;

  const float x = y[static_cast<int>(RacerDubinsParams::OutputIndex::BASELINK_POS_I_X)];
  const float y_pos = y[static_cast<int>(RacerDubinsParams::OutputIndex::BASELINK_POS_I_Y)];
  const float vel = y[static_cast<int>(RacerDubinsParams::OutputIndex::TOTAL_VELOCITY)];

  const float track_val = computeTrackValue(x, y_pos);

  const float vel_diff = vel - this->params_.desired_speed;
  const float speed_cost = this->params_.speed_coeff * (vel_diff * vel_diff);
  const float track_cost = this->params_.track_coeff * track_val;
  float crash_cost = 0.0F;
  if (track_val >= this->params_.boundary_threshold)
  {
    crash_cost = this->params_.crash_coeff;
    if (crash_status != nullptr)
    {
      *crash_status = 1;
    }
  }

  return speed_cost + track_cost + crash_cost;
}

template <class CLASS_T, int NUM_TIMESTEPS, class PARAMS_T, class DYN_PARAMS_T>
float RacerCostImpl<CLASS_T, NUM_TIMESTEPS, PARAMS_T, DYN_PARAMS_T>::computeStateCost(
    const Eigen::Ref<const output_array>& y, int timestep, int* crash_status)
{
  const float x = y[static_cast<int>(RacerDubinsParams::OutputIndex::BASELINK_POS_I_X)];
  const float y_pos = y[static_cast<int>(RacerDubinsParams::OutputIndex::BASELINK_POS_I_Y)];
  const float vel = y[static_cast<int>(RacerDubinsParams::OutputIndex::TOTAL_VELOCITY)];

  const float track_val = computeTrackValue(x, y_pos);

  const float vel_diff = vel - this->params_.desired_speed;
  const float speed_cost = this->params_.speed_coeff * (vel_diff * vel_diff);
  const float track_cost = this->params_.track_coeff * track_val;
  float crash_cost = 0.0F;
  if (track_val >= this->params_.boundary_threshold)
  {
    crash_cost = this->params_.crash_coeff;
    if (crash_status != nullptr)
    {
      *crash_status = 1;
    }
  }

  return speed_cost + track_cost + crash_cost;
}

template <class CLASS_T, int NUM_TIMESTEPS, class PARAMS_T, class DYN_PARAMS_T>
__device__ float RacerCostImpl<CLASS_T, NUM_TIMESTEPS, PARAMS_T, DYN_PARAMS_T>::computeControlCost(float* u,
                                                                                                     int timestep,
                                                                                                     float* theta_c,
                                                                                                     int* crash)
{
  (void)timestep;
  (void)theta_c;
  (void)crash;
  const float steer = u[static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD)];
  return this->params_.steer_coeff * (steer * steer);
}

template <class CLASS_T, int NUM_TIMESTEPS, class PARAMS_T, class DYN_PARAMS_T>
float RacerCostImpl<CLASS_T, NUM_TIMESTEPS, PARAMS_T, DYN_PARAMS_T>::computeControlCost(
    const Eigen::Ref<const control_array>& u, int timestep, int* crash)
{
  (void)timestep;
  (void)crash;
  const float steer = u(static_cast<int>(RacerDubinsParams::ControlIndex::STEER_CMD));
  return this->params_.steer_coeff * (steer * steer);
}

template <class CLASS_T, int NUM_TIMESTEPS, class PARAMS_T, class DYN_PARAMS_T>
__device__ float RacerCostImpl<CLASS_T, NUM_TIMESTEPS, PARAMS_T, DYN_PARAMS_T>::terminalCost(float* y, float* theta_c)
{
  (void)theta_c;
  const float x = y[static_cast<int>(RacerDubinsParams::OutputIndex::BASELINK_POS_I_X)];
  const float y_pos = y[static_cast<int>(RacerDubinsParams::OutputIndex::BASELINK_POS_I_Y)];
  const float track_val = computeTrackValue(x, y_pos);
  return this->params_.track_coeff * track_val * 10.0F;
}
