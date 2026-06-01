/**
 * OpenCV visualization helpers for 2D path-tracking examples.
 */
#pragma once

#include <mppi/path/path2d.hpp>
#include <mppi/path/path_reference_generator.hpp>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <vector>

namespace mppi
{
namespace viz
{

inline cv::Point2f worldToPixel(const float x, const float y, const int img_w, const int img_h,
                                const float scale = 15.0F)
{
  const float u = static_cast<float>(img_w) / 2.0F + x * scale;
  const float v = static_cast<float>(img_h) / 2.0F - y * scale;
  return cv::Point2f(u, v);
}

inline void drawCenterline(cv::Mat& img, const path::Path2D& path, const float scale = 15.0F)
{
  cv::Mat overlay = img.clone();
  const auto& anchors = path.anchors();
  for (size_t i = 0; i + 1 < anchors.size(); ++i)
  {
    cv::line(overlay, worldToPixel(anchors[i].x, anchors[i].y, img.cols, img.rows, scale),
             worldToPixel(anchors[i + 1].x, anchors[i + 1].y, img.cols, img.rows, scale), cv::Scalar(128, 128, 128),
             2);
  }
  cv::addWeighted(overlay, 0.5, img, 0.5, 0, img);
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
             worldToPixel(ref[i + 1].x, ref[i + 1].y, img.cols, img.rows, scale), cv::Scalar(255, 0, 0), 2);
  }
  cv::addWeighted(overlay, 0.5, img, 0.5, 0, img);
}

template <typename TrajectoryMatrix>
inline void drawTrajectory(cv::Mat& img, const TrajectoryMatrix& traj, const int x_idx, const int y_idx,
                           const cv::Scalar& color = cv::Scalar(0, 255, 0), const int thickness = 2,
                           const float overlay_alpha = 0.75F, const float scale = 15.0F)
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
                                    const cv::Scalar& color = cv::Scalar(180, 180, 180), const int thickness = 1,
                                    const float overlay_alpha = 0.4F, const float scale = 15.0F)
{
  if (sampled_trajectories.empty() || num_timesteps <= 1)
  {
    return;
  }

  // GPU->host copy stores outputs in columns [0, num_timesteps - 2]; column num_timesteps - 1 stays zero.
  const int num_valid_cols = num_timesteps - 1;

  cv::Mat overlay = img.clone();
  for (const TrajectoryMatrix& traj : sampled_trajectories)
  {
    const int cols_to_draw = std::min(num_valid_cols, static_cast<int>(traj.cols()));
    for (int i = 0; i + 1 < cols_to_draw; ++i)
    {
      cv::line(overlay, worldToPixel(traj(x_idx, i), traj(y_idx, i), img.cols, img.rows, scale),
               worldToPixel(traj(x_idx, i + 1), traj(y_idx, i + 1), img.cols, img.rows, scale), color, thickness);
    }
  }
  cv::addWeighted(overlay, overlay_alpha, img, 1.0F - overlay_alpha, 0, img);
}

}  // namespace viz
}  // namespace mppi
