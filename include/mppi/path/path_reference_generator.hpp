/**
 * Generate a time-stamped reference trajectory along a Path2D for MPPI tracking.
 * Samples are spaced by dt (default 0.1 s) in time; arc length advances by v_ref(s) * dt.
 *
 * The arc-length-only `generate(path, start_s, count)` produces a pure path-following reference.
 * The pose-aware overload starts at the vehicle pose and smoothly merges onto the path reference
 * while ramping speed toward the configured target.
 */
#pragma once

#include <mppi/path/path2d.hpp>
#include <mppi/utils/angle_utils.cuh>

#include <algorithm>
#include <cmath>
#include <vector>

namespace mppi
{
namespace path
{

struct PathReferenceSample
{
  float t = 0.0F;
  float arc_length_s = 0.0F;
  float x = 0.0F;
  float y = 0.0F;
  float yaw = 0.0F;
  float v = 0.0F;
};

class PathReferenceGenerator
{
public:
  PathReferenceGenerator() = default;

  explicit PathReferenceGenerator(const float dt) : dt_(dt)
  {
  }

  void setDt(const float dt)
  {
    dt_ = dt;
  }

  float dt() const
  {
    return dt_;
  }

  void setSpeedCap(const float v_max)
  {
    v_max_ = v_max;
    if (target_speed_ > v_max_)
    {
      target_speed_ = v_max_;
    }
  }

  void setTargetSpeed(const float v_target)
  {
    target_speed_ = std::min(v_target, v_max_);
  }

  float targetSpeed() const
  {
    return target_speed_;
  }

  void setLateralAccelCap(const float a_lat_max)
  {
    a_lat_max_ = a_lat_max;
  }

  /** Number of horizon steps over which pose/speed blend from the vehicle state onto the path. */
  void setMergeHorizonSteps(const int steps)
  {
    merge_horizon_steps_ = std::max(1, steps);
  }

  int mergeHorizonSteps() const
  {
    return merge_horizon_steps_;
  }

  /** Curvature-limited cruise speed at arc length s, capped by target_speed_. */
  float speedAt(const Path2D& path, const float s) const
  {
    const float kappa = std::fabs(path.curvatureAt(s));
    const float kappa_eps = 1.0E-4F;
    if (kappa < kappa_eps)
    {
      return target_speed_;
    }
    return std::min(target_speed_, std::sqrt(a_lat_max_ / kappa));
  }

  /**
   * Build reference samples at t = 0, dt, 2*dt, ... (count-1)*dt along the path forward from start_s.
   * @param count  Number of samples (e.g. MPPI horizon + 1)
   */
  std::vector<PathReferenceSample> generate(const Path2D& path, const float start_s, const int count) const
  {
    std::vector<PathReferenceSample> out;
    if (count <= 0 || path.empty())
    {
      return out;
    }
    out.resize(static_cast<size_t>(count));
    float s = path.wrapArcLength(start_s);
    for (int k = 0; k < count; ++k)
    {
      PathReferenceSample& r = out[static_cast<size_t>(k)];
      r.t = static_cast<float>(k) * dt_;
      r.arc_length_s = s;
      const Pose2D p = path.poseAt(s);
      r.x = p.x;
      r.y = p.y;
      r.yaw = p.yaw;
      r.v = speedAt(path, s);
      if (!path.closed())
      {
        const float dist_to_end = path.length() - s;
        if (dist_to_end < 5.0F)
        {
          r.v = std::min(r.v, target_speed_ * std::max(0.0F, dist_to_end / 5.0F));
        }
        if (dist_to_end <= 0.0F)
        {
          r.v = 0.0F;
        }
      }
      if (k + 1 < count)
      {
        const float v_mid = r.v;
        const float s_next = s + v_mid * dt_;
        if (path.closed())
        {
          s = path.wrapArcLength(s_next);
        }
        else
        {
          s = std::min(s_next, path.length());
        }
      }
    }
    return out;
  }

  /**
   * Pose-aware reference: sample 0 is the current vehicle state; over merge_horizon_steps_ the
   * reference blends onto the path-following trajectory while speed ramps toward target_speed_.
   */
  std::vector<PathReferenceSample> generate(const Path2D& path, const float start_s, const int count, const float x,
                                          const float y, const float yaw, const float v) const
  {
    const std::vector<PathReferenceSample> path_ref = generate(path, start_s, count);
    if (count <= 0)
    {
      return path_ref;
    }

    std::vector<PathReferenceSample> out = path_ref;
    const int merge = std::min(merge_horizon_steps_, std::max(1, count - 1));

    for (int k = 0; k < count; ++k)
    {
      PathReferenceSample& r = out[static_cast<size_t>(k)];
      const PathReferenceSample& p = path_ref[static_cast<size_t>(k)];
      const float tk = static_cast<float>(k) * dt_;
      const float alpha = smoothstep01(static_cast<float>(k) / static_cast<float>(merge));
      const float speed_blend = smoothstep01(static_cast<float>(k) / static_cast<float>(merge));
      const float v_ramped = v + (target_speed_ - v) * speed_blend;

      if (k == 0)
      {
        r.x = x;
        r.y = y;
        r.yaw = yaw;
        r.v = v;
        r.t = 0.0F;
        r.arc_length_s = p.arc_length_s;
        continue;
      }

      const float x_free = x + v_ramped * std::cos(yaw) * tk;
      const float y_free = y + v_ramped * std::sin(yaw) * tk;

      r.x = (1.0F - alpha) * x_free + alpha * p.x;
      r.y = (1.0F - alpha) * y_free + alpha * p.y;
      r.yaw = angle_utils::interpolateEulerAngleLinear(yaw, p.yaw, alpha);
      r.v = (1.0F - alpha) * v_ramped + alpha * p.v;
      r.t = tk;
      r.arc_length_s = p.arc_length_s;
    }
    return out;
  }

private:
  static float smoothstep01(float t)
  {
    t = std::max(0.0F, std::min(1.0F, t));
    return t * t * (3.0F - 2.0F * t);
  }

  float dt_ = 0.1F;
  float v_max_ = 3.0F;
  float target_speed_ = 3.0F;
  float a_lat_max_ = 2.0F;
  int merge_horizon_steps_ = 20;
};

}  // namespace path
}  // namespace mppi
