/**
 * OpenCV visualization helpers for 2D path-tracking examples.
 */
#pragma once

#include <mppi/path/path2d.hpp>
#include <mppi/path/path_reference_generator.hpp>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

namespace mppi
{
namespace viz
{

inline cv::Mat makeWhiteFrame(const int img_w, const int img_h)
{
  return cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(255, 255, 255));
}

inline cv::Point2f worldToPixel(const float x, const float y, const int img_w, const int img_h,
                                const float scale = 15.0F)
{
  const float u = static_cast<float>(img_w) / 2.0F + x * scale;
  const float v = static_cast<float>(img_h) / 2.0F - y * scale;
  return cv::Point2f(u, v);
}

inline cv::Scalar lerpBgr(const cv::Scalar& a, const cv::Scalar& b, const float t)
{
  const float s = std::max(0.0F, std::min(1.0F, t));
  return cv::Scalar(a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2]));
}

inline cv::Scalar costGradientColor(const float cost, const float min_cost, const float max_cost)
{
  const cv::Scalar k_teal(128, 128, 0);
  const cv::Scalar k_purple(128, 0, 128);
  if (max_cost <= min_cost)
  {
    return k_teal;
  }
  const float t = (cost - min_cost) / (max_cost - min_cost);
  return lerpBgr(k_teal, k_purple, t);
}

inline void drawCenterline(cv::Mat& img, const path::Path2D& path, const float scale = 15.0F)
{
  cv::Mat overlay = img.clone();
  const auto& anchors = path.anchors();
  for (size_t i = 0; i + 1 < anchors.size(); ++i)
  {
    cv::line(overlay, worldToPixel(anchors[i].x, anchors[i].y, img.cols, img.rows, scale),
             worldToPixel(anchors[i + 1].x, anchors[i + 1].y, img.cols, img.rows, scale), cv::Scalar(200, 200, 200),
             2);
  }
  cv::addWeighted(overlay, 0.85, img, 0.15, 0, img);
}

inline void drawReferencePath(cv::Mat& img, const std::vector<path::PathReferenceSample>& ref,
                              const float scale = 15.0F)
{
  if (ref.size() < 2)
  {
    return;
  }

  cv::Mat overlay = img.clone();
  for (size_t i = 0; i + 1 < ref.size(); ++i)
  {
    cv::line(overlay, worldToPixel(ref[i].x, ref[i].y, img.cols, img.rows, scale),
             worldToPixel(ref[i + 1].x, ref[i + 1].y, img.cols, img.rows, scale), cv::Scalar(0, 165, 255), 2);
  }
  cv::addWeighted(overlay, 0.7, img, 0.3, 0, img);
}

template <typename TrajectoryMatrix>
inline void drawTrajectory(cv::Mat& img, const TrajectoryMatrix& traj, const int x_idx, const int y_idx,
                           const cv::Scalar& color = cv::Scalar(255, 64, 0), const int thickness = 3,
                           const float overlay_alpha = 0.85F, const float scale = 15.0F)
{
  cv::Mat overlay = img.clone();
  for (int i = 0; i + 1 < traj.cols(); ++i)
  {
    cv::line(overlay, worldToPixel(traj(x_idx, i), traj(y_idx, i), img.cols, img.rows, scale),
             worldToPixel(traj(x_idx, i + 1), traj(y_idx, i + 1), img.cols, img.rows, scale), color, thickness);
  }
  cv::addWeighted(overlay, overlay_alpha, img, 1.0F - overlay_alpha, 0, img);
}

template <typename TrajectoryMatrix>
inline void drawSampledTrajectories(cv::Mat& img, const std::vector<TrajectoryMatrix>& sampled_trajectories,
                                    const int x_idx, const int y_idx, const int num_timesteps,
                                    const std::vector<float>& rollout_costs, const int thickness = 1,
                                    const float scale = 15.0F)
{
  if (sampled_trajectories.empty() || num_timesteps <= 1 ||
      rollout_costs.size() != sampled_trajectories.size())
  {
    return;
  }

  const int num_valid_cols = num_timesteps - 1;
  const float min_cost = *std::min_element(rollout_costs.begin(), rollout_costs.end());
  const float max_cost = *std::max_element(rollout_costs.begin(), rollout_costs.end());

  std::vector<size_t> draw_order(sampled_trajectories.size());
  std::iota(draw_order.begin(), draw_order.end(), 0U);
  std::sort(draw_order.begin(), draw_order.end(),
            [&rollout_costs](const size_t a, const size_t b) { return rollout_costs[a] > rollout_costs[b]; });

  for (const size_t idx : draw_order)
  {
    const TrajectoryMatrix& traj = sampled_trajectories[idx];
    const cv::Scalar color = costGradientColor(rollout_costs[idx], min_cost, max_cost);
    const int cols_to_draw = std::min(num_valid_cols, static_cast<int>(traj.cols()));
    for (int i = 0; i + 1 < cols_to_draw; ++i)
    {
      cv::line(img, worldToPixel(traj(x_idx, i), traj(y_idx, i), img.cols, img.rows, scale),
               worldToPixel(traj(x_idx, i + 1), traj(y_idx, i + 1), img.cols, img.rows, scale), color, thickness);
    }
  }
}

}  // namespace viz
}  // namespace mppi
