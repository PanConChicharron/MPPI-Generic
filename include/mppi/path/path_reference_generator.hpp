/**
 * Generate a time-stamped reference trajectory along a Path2D for MPPI tracking.
 * Samples are spaced by dt (default 0.1 s) in time; arc length advances by v_ref(s) * dt.
 *
 * The arc-length-only `generate(path, start_s, count)` produces a pure path-following reference.
 * The pose-aware overload projects (x, y) to the closest path point and samples forward from there
 * (no merge from the vehicle pose).
 */
#pragma once

#include <mppi/path/path2d.hpp>
#include <mppi/path/path_projection.hpp>

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
   * Pose-aware reference: project (x, y) onto the path, then sample the path forward from the
   * footpoint. yaw and v are unused (kept for call-site compatibility).
   */
  std::vector<PathReferenceSample> generate(const Path2D& path, const float start_s, const int count, const float x,
                                          const float y, const float yaw, const float v) const
  {
    (void)yaw;
    (void)v;
    if (count <= 0 || path.empty())
    {
      return {};
    }
    const float s_hint = path.wrapArcLength(start_s);
    const PathProjection proj = projectPoseOntoPath(path, x, y, s_hint);
    return generate(path, proj.arc_length_s, count);
  }

private:
  float dt_ = 0.1F;
  float v_max_ = 3.0F;
  float target_speed_ = 3.0F;
  float a_lat_max_ = 8.0F;
};

}  // namespace path
}  // namespace mppi
