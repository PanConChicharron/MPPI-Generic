/**
 * Path-tracking feedforward feedback: zero throttle, steering from path curvature.
 *
 * Vanilla MPPI does not call feedback inside rollout kernels; bias the nominal control
 * with applyFeedforwardToNominal() before each solve (see racer_dubins_stadium example).
 */
#pragma once

#include <mppi/feedback_controllers/feedback.cuh>
#include <mppi/path/path2d.hpp>
#include <mppi/path/path_reference_generator.hpp>
#include <mppi/utils/angle_utils.cuh>

#include <array>
#include <cmath>
#include <vector>

struct PathTrackerFeedbackParams
{
};

template <int NUM_TIMESTEPS>
struct PathTrackerFeedbackState : GPUState
{
  float ref_kappa[NUM_TIMESTEPS] = {};
  float goal_yaw[NUM_TIMESTEPS] = {};
  float wheel_base = 0.3F;
  float steer_angle_scale = 1.0F;
  float steer_command_angle_scale = 1.0F;
  float dt = 0.01F;
  int num_timesteps = NUM_TIMESTEPS;
  int ref_kappa_valid = 0;
  int goal_traj_valid = 0;
};

namespace mppi
{
namespace feedback
{
namespace detail
{

template <class Params>
struct HasBicyclePathSteer
{
private:
  template <class U, class = decltype(U::StateIndex::YAW), class = decltype(U::StateIndex::VEL_X),
            class = decltype(U::ControlIndex::STEER_CMD), class = decltype(std::declval<U&>().wheel_base),
            class = decltype(std::declval<U&>().steer_angle_scale),
            class = decltype(std::declval<U&>().steer_command_angle_scale)>
  static std::true_type test(int);
  template <class U>
  static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<Params>(0))::value;
};

template <class Params>
__host__ __device__ inline float steerCommandFromCurvature(const Params& p, float kappa)
{
  const float steer_angle = atanf(kappa * p.wheel_base) * p.steer_angle_scale;
  const float cmd = steer_angle / p.steer_command_angle_scale;
  return fmaxf(-1.0F, fminf(1.0F, cmd));
}

template <class DYN_T, int NUM_TIMESTEPS, bool Enabled = HasBicyclePathSteer<typename DYN_T::DYN_PARAMS_T>::value>
struct PathTrackerSteerControl;

template <class DYN_T, int NUM_TIMESTEPS>
struct PathTrackerSteerControl<DYN_T, NUM_TIMESTEPS, true>
{
  using Params = typename DYN_T::DYN_PARAMS_T;
  using control_array = typename DYN_T::control_array;
  using state_trajectory = Eigen::Matrix<float, DYN_T::STATE_DIM, NUM_TIMESTEPS>;

  static control_array compute(DYN_T& dyn, const state_trajectory& goal_traj, const typename DYN_T::state_array& x_act,
                               int t, float dt, float path_kappa, bool path_kappa_valid)
  {
    control_array u = control_array::Zero();
    constexpr int kYaw = static_cast<int>(Params::StateIndex::YAW);
    constexpr int kVel = static_cast<int>(Params::StateIndex::VEL_X);
    constexpr int kSteer = static_cast<int>(Params::ControlIndex::STEER_CMD);

    float kappa = path_kappa;
    if (!path_kappa_valid)
    {
      const int t_next = std::min(t + 1, NUM_TIMESTEPS - 1);
      const float yaw0 = goal_traj(kYaw, t);
      const float yaw1 = goal_traj(kYaw, t_next);
      const float vel = std::max(std::fabs(x_act(kVel)), 0.1F);
      const float ds = std::max(vel * dt * static_cast<float>(t_next - t), 1.0E-3F);
      kappa = angle_utils::shortestAngularDistance(yaw0, yaw1) / ds;
    }

    u(kSteer) = steerCommandFromCurvature(dyn.getParams(), kappa);
    return u;
  }
};

template <class DYN_T, int NUM_TIMESTEPS>
struct PathTrackerSteerControl<DYN_T, NUM_TIMESTEPS, false>
{
  using control_array = typename DYN_T::control_array;

  static control_array compute(const DYN_T&, const Eigen::Matrix<float, DYN_T::STATE_DIM, NUM_TIMESTEPS>&,
                               const typename DYN_T::state_array&, int, float, float, bool)
  {
    return control_array::Zero();
  }
};

}  // namespace detail

template <class DYN_T, int NUM_TIMESTEPS>
Eigen::Matrix<float, DYN_T::STATE_DIM, NUM_TIMESTEPS> goalTrajectoryFromPathReference(
    const std::vector<mppi::path::PathReferenceSample>& ref)
{
  using Params = typename DYN_T::DYN_PARAMS_T;
  Eigen::Matrix<float, DYN_T::STATE_DIM, NUM_TIMESTEPS> goal_traj =
      Eigen::Matrix<float, DYN_T::STATE_DIM, NUM_TIMESTEPS>::Zero();

  constexpr int kVel = static_cast<int>(Params::StateIndex::VEL_X);
  constexpr int kYaw = static_cast<int>(Params::StateIndex::YAW);
  constexpr int kX = static_cast<int>(Params::StateIndex::POS_X);
  constexpr int kY = static_cast<int>(Params::StateIndex::POS_Y);

  for (int t = 0; t < NUM_TIMESTEPS; ++t)
  {
    const mppi::path::PathReferenceSample& s = ref[static_cast<size_t>(std::min(t, static_cast<int>(ref.size()) - 1))];
    goal_traj(kVel, t) = s.v;
    goal_traj(kYaw, t) = s.yaw;
    goal_traj(kX, t) = s.x;
    goal_traj(kY, t) = s.y;
  }
  return goal_traj;
}

template <int NUM_TIMESTEPS>
std::array<float, NUM_TIMESTEPS> referenceCurvaturesFromPath(const mppi::path::Path2D& path,
                                                             const std::vector<mppi::path::PathReferenceSample>& ref)
{
  std::array<float, NUM_TIMESTEPS> kappa{};
  for (int t = 0; t < NUM_TIMESTEPS; ++t)
  {
    const mppi::path::PathReferenceSample& s = ref[static_cast<size_t>(std::min(t, static_cast<int>(ref.size()) - 1))];
    kappa[static_cast<size_t>(t)] = path.curvatureAt(s.arc_length_s);
  }
  return kappa;
}

}  // namespace feedback
}  // namespace mppi

template <class DYN_T, int NUM_TIMESTEPS>
class PathTrackerFeedbackImpl
    : public GPUFeedbackController<PathTrackerFeedbackImpl<DYN_T, NUM_TIMESTEPS>, DYN_T,
                                   PathTrackerFeedbackState<NUM_TIMESTEPS>>
{
public:
  using FEEDBACK_STATE_T = PathTrackerFeedbackState<NUM_TIMESTEPS>;
  using Params = typename DYN_T::DYN_PARAMS_T;

  PathTrackerFeedbackImpl(cudaStream_t stream = 0)
    : GPUFeedbackController<PathTrackerFeedbackImpl<DYN_T, NUM_TIMESTEPS>, DYN_T, FEEDBACK_STATE_T>(stream)
  {
  }

  __device__ void k(const float* __restrict__ x_act, const float* __restrict__ x_goal, const int t,
                    float* __restrict__ theta, float* __restrict__ control_output)
  {
    (void)x_goal;
    (void)theta;
    for (int i = 0; i < DYN_T::CONTROL_DIM; ++i)
    {
      control_output[i] = 0.0F;
    }

    const FEEDBACK_STATE_T& st = this->state_;
    if (st.goal_traj_valid == 0)
    {
      return;
    }

    constexpr int kVel = static_cast<int>(Params::StateIndex::VEL_X);
    constexpr int kSteer = static_cast<int>(Params::ControlIndex::STEER_CMD);

    const int t_clamped = min(max(t, 0), st.num_timesteps - 1);
    float kappa = 0.0F;
    if (st.ref_kappa_valid != 0)
    {
      kappa = st.ref_kappa[t_clamped];
    }
    else
    {
      const int t_next = min(t_clamped + 1, st.num_timesteps - 1);
      const float vel = fmaxf(fabsf(x_act[kVel]), 0.1F);
      const float ds = fmaxf(vel * st.dt * static_cast<float>(t_next - t_clamped), 1.0E-3F);
      kappa = angle_utils::shortestAngularDistance(st.goal_yaw[t_clamped], st.goal_yaw[t_next]) / ds;
    }

    Params p;
    p.wheel_base = st.wheel_base;
    p.steer_angle_scale = st.steer_angle_scale;
    p.steer_command_angle_scale = st.steer_command_angle_scale;
    control_output[kSteer] = mppi::feedback::detail::steerCommandFromCurvature(p, kappa);
  }
};

template <class DYN_T, int NUM_TIMESTEPS>
class PathTrackerFeedback
    : public FeedbackController<PathTrackerFeedbackImpl<DYN_T, NUM_TIMESTEPS>, PathTrackerFeedbackParams, NUM_TIMESTEPS>
{
public:
  using PARENT_CLASS =
      FeedbackController<PathTrackerFeedbackImpl<DYN_T, NUM_TIMESTEPS>, PathTrackerFeedbackParams, NUM_TIMESTEPS>;
  using control_array = typename PARENT_CLASS::control_array;
  using control_trajectory = typename PARENT_CLASS::control_trajectory;
  using state_array = typename PARENT_CLASS::state_array;
  using state_trajectory = typename PARENT_CLASS::state_trajectory;
  using TEMPLATED_FEEDBACK_STATE = typename PARENT_CLASS::TEMPLATED_FEEDBACK_STATE;
  using FEEDBACK_STATE_T = PathTrackerFeedbackState<NUM_TIMESTEPS>;

  PathTrackerFeedback(DYN_T* dyn = nullptr, float dt = 0.01F) : PARENT_CLASS(dt, NUM_TIMESTEPS), dyn_(dyn)
  {
  }

  void initTrackingController() override
  {
  }

  void updateReference(const mppi::path::Path2D& path, const std::vector<mppi::path::PathReferenceSample>& ref)
  {
    ref_kappa_ = mppi::feedback::referenceCurvaturesFromPath<NUM_TIMESTEPS>(path, ref);
    ref_kappa_valid_ = true;
  }

  control_array k_(const Eigen::Ref<const state_array>& x_act, const Eigen::Ref<const state_array>& x_goal, int t,
                  TEMPLATED_FEEDBACK_STATE& fb_state) override
  {
    (void)x_goal;
    (void)fb_state;
    if (dyn_ == nullptr || !goal_traj_valid_)
    {
      return control_array::Zero();
    }
    const float path_kappa = ref_kappa_valid_ ? ref_kappa_[static_cast<size_t>(t)] : 0.0F;
    return mppi::feedback::detail::PathTrackerSteerControl<DYN_T, NUM_TIMESTEPS>::compute(
        *dyn_, goal_traj_, x_act, t, this->getDt(), path_kappa, ref_kappa_valid_);
  }

  void computeFeedback(const Eigen::Ref<const state_array>& init_state,
                       const Eigen::Ref<const state_trajectory>& goal_traj,
                       const Eigen::Ref<const control_trajectory>& control_traj) override
  {
    (void)init_state;
    (void)control_traj;
    goal_traj_ = goal_traj;
    goal_traj_valid_ = true;
    syncFeedbackStateToGpu();
  }

  /**
   * Set nominal steer from path curvature (throttle entries unchanged).
   * Replaces steer each step; do not accumulate on top of the previous MPPI solution.
   */
  void applyFeedforwardToNominal(Eigen::Ref<control_trajectory> u_nom, const Eigen::Ref<const state_array>& x,
                                 const Eigen::Ref<const state_trajectory>& goal_traj)
  {
    computeFeedback(x, goal_traj, u_nom);
    using Params = typename DYN_T::DYN_PARAMS_T;
    constexpr int kSteer = static_cast<int>(Params::ControlIndex::STEER_CMD);
    TEMPLATED_FEEDBACK_STATE fb_state{};
    for (int t = 0; t < NUM_TIMESTEPS; ++t)
    {
      const control_array u_ff = k_(x, goal_traj.col(t), t, fb_state);
      u_nom(kSteer, t) = u_ff(kSteer);
    }
  }

private:
  void syncFeedbackStateToGpu()
  {
    if (dyn_ == nullptr || !goal_traj_valid_)
    {
      return;
    }

    FEEDBACK_STATE_T gpu_state{};
    gpu_state.num_timesteps = NUM_TIMESTEPS;
    gpu_state.dt = this->getDt();
    gpu_state.goal_traj_valid = 1;
    gpu_state.ref_kappa_valid = ref_kappa_valid_ ? 1 : 0;

    using Params = typename DYN_T::DYN_PARAMS_T;
    constexpr int kYaw = static_cast<int>(Params::StateIndex::YAW);
    for (int t = 0; t < NUM_TIMESTEPS; ++t)
    {
      gpu_state.goal_yaw[t] = goal_traj_(kYaw, t);
      gpu_state.ref_kappa[t] = ref_kappa_valid_ ? ref_kappa_[static_cast<size_t>(t)] : 0.0F;
    }

    const auto& p = dyn_->getParams();
    gpu_state.wheel_base = p.wheel_base;
    gpu_state.steer_angle_scale = p.steer_angle_scale;
    gpu_state.steer_command_angle_scale = p.steer_command_angle_scale;

    this->getHostPointer()->setFeedbackState(gpu_state);
  }

  DYN_T* dyn_ = nullptr;
  state_trajectory goal_traj_ = state_trajectory::Zero();
  std::array<float, NUM_TIMESTEPS> ref_kappa_{};
  bool goal_traj_valid_ = false;
  bool ref_kappa_valid_ = false;
};
