/**
 * Host-side bridge: fill PathTrackingCostParams from path reference samples.
 */
#pragma once

#include <mppi/cost_functions/path_tracking/path_tracking_cost.cuh>
#include <mppi/path/path2d.hpp>
#include <mppi/path/path_reference_generator.hpp>
#include <mppi/utils/angle_utils.cuh>

#include <algorithm>
#include <cmath>

namespace mppi
{
namespace path
{

template <int REF_HORIZON>
inline void fillPathTrackingCostWeights(PathTrackingCostParams<REF_HORIZON>& p, const float w_pos = 10.0F,
                                        const float w_heading_so2 = 1.0F, const float w_vel = 5.0F,
                                        const float w_lat_accel = 0.0F, const float w_lat_jerk = 0.0F,
                                        const float w_steer_dot = 0.0F, const float w_accel = 0.05F,
                                        const float w_steer = 0.1F)
{
  p.w_pos = w_pos;
  p.w_heading_so2 = w_heading_so2;
  p.w_vel = w_vel;
  p.w_lat_accel = w_lat_accel;
  p.w_lat_jerk = w_lat_jerk;
  p.w_steer_dot = w_steer_dot;
  p.control_cost_coeff[static_cast<int>(DubinsBicycleParams::ControlIndex::ACCEL)] = w_accel;
  p.control_cost_coeff[static_cast<int>(DubinsBicycleParams::ControlIndex::STEER)] = w_steer;
}

template <int REF_HORIZON>
inline void fillPathTrackingBicycleGeometry(PathTrackingCostParams<REF_HORIZON>& p, const DubinsBicycleParams& dyn)
{
  p.wheel_base = dyn.wheel_base;
  p.steer_time_constant = dyn.steer_time_constant;
}

template <int REF_HORIZON>
inline void fillCostFromPathReference(PathTrackingCostParams<REF_HORIZON>& cost_params,
                                      const std::vector<PathReferenceSample>& ref, const Path2D* path = nullptr,
                                      const DubinsBicycleParams* dyn = nullptr)
{
  using O = DubinsBicycleParams::OutputIndex;
  const int n = std::min(static_cast<int>(ref.size()), REF_HORIZON);
  for (int k = 0; k < n; ++k)
  {
    const int base = k * DubinsBicycle::OUTPUT_DIM;
    const PathReferenceSample& r = ref[static_cast<size_t>(k)];
    cost_params.s_goal[base + static_cast<int>(O::POS_X)] = r.x;
    cost_params.s_goal[base + static_cast<int>(O::POS_Y)] = r.y;
    cost_params.s_goal[base + static_cast<int>(O::YAW)] = r.yaw;
    cost_params.s_goal[base + static_cast<int>(O::VEL_X)] = r.v;
    float steer_ref = 0.0F;
    if (path != nullptr && dyn != nullptr)
    {
      steer_ref = std::atan(dyn->wheel_base * path->curvatureAt(r.arc_length_s));
    }
    cost_params.s_goal[base + static_cast<int>(O::STEER_ANGLE)] = steer_ref;
  }
  cost_params.setCurrentTime(0);
}

/** Lateral offset along the path left-normal from poseAt(s). */
inline void applyInitialLateralOffset(const Path2D& path, const float s, const float lateral_offset,
                                     float& px, float& py)
{
  const Pose2D p = path.poseAt(s);
  float tx = 0.0F;
  float ty = 0.0F;
  path.tangentAt(s, tx, ty);
  const float nx = -ty;
  const float ny = tx;
  px = p.x + lateral_offset * nx;
  py = p.y + lateral_offset * ny;
}

/**
 * Feedforward nominal control sequence for MPPI importance sampling.
 * Accel tracks v_ref along the horizon; steer combines curvature feedforward, heading, and cross-track terms.
 */
template <class ControlTrajectory>
inline void fillNominalControlFromReference(ControlTrajectory& u_seq,
                                            const Eigen::Matrix<float, DubinsBicycle::STATE_DIM, 1>& x,
                                            const std::vector<PathReferenceSample>& ref, const DubinsBicycleParams& dyn,
                                            const float dt, const Path2D* path = nullptr,
                                            const float k_lat_steer = 1.5F, const float k_heading_steer = 0.5F)
{
  using S = DubinsBicycleParams::StateIndex;
  using C = DubinsBicycleParams::ControlIndex;

  const int horizon = std::min(static_cast<int>(ref.size()), static_cast<int>(u_seq.cols()));
  float v_nom = x(static_cast<int>(S::VEL_X));
  float yaw_nom = x(static_cast<int>(S::YAW));
  float px_nom = x(static_cast<int>(S::POS_X));
  float py_nom = x(static_cast<int>(S::POS_Y));

  for (int k = 0; k < horizon; ++k)
  {
    const PathReferenceSample& r = ref[static_cast<size_t>(k)];
    const float v_err = r.v - v_nom;
    float accel = v_err / std::max(dt, 1.0E-4F);
    accel = std::max(dyn.min_accel, std::min(accel, dyn.max_accel));

    const float cy = std::cos(yaw_nom);
    const float sy = std::sin(yaw_nom);
    const float e_lat = -sy * (px_nom - r.x) + cy * (py_nom - r.y);
    const float dpsi = angle_utils::shortestAngularDistance(yaw_nom, r.yaw);
    float steer = 0.0F;
    if (path != nullptr)
    {
      const float kappa = path->curvatureAt(r.arc_length_s);
      steer = std::atan(dyn.wheel_base * kappa);
    }
    steer += k_heading_steer * dpsi;
    if (std::fabs(v_nom) > 0.5F)
    {
      steer -= std::atan2(k_lat_steer * e_lat, std::fabs(v_nom));
    }
    else
    {
      steer -= k_lat_steer * e_lat;
    }
    steer = std::max(-dyn.max_steer_angle, std::min(steer, dyn.max_steer_angle));

    u_seq(static_cast<int>(C::ACCEL), k) = accel;
    u_seq(static_cast<int>(C::STEER), k) = steer;

    v_nom += accel * dt;
    const float yaw_rate = (v_nom / dyn.wheel_base) * std::tan(steer);
    yaw_nom = angle_utils::normalizeAngle(yaw_nom + yaw_rate * dt);
    px_nom += v_nom * cy * dt;
    py_nom += v_nom * sy * dt;
  }
}

}  // namespace path
}  // namespace mppi
