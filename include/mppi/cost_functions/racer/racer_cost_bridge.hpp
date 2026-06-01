/**
 * Host-side helpers to populate RacerCost reference trajectories and obstacles.
 */
#pragma once

#include <mppi/cost_functions/racer/racer_cost.cuh>
#include <mppi/path/path2d.hpp>
#include <mppi/path/path_reference_generator.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace mppi
{
namespace cost
{

/** Parked vehicle on the road shoulder (oriented box for cost and viz). */
struct ParkedCarObstacle
{
  float ox = 0.0F;
  float oy = 0.0F;
  float yaw = 0.0F;
  float length = 0.55F * 1.5F;
  float width = 0.28F * 2.5F;
};

/**
 * Place parked cars along both shoulders near the road boundary, alternating left/right with gaps
 * so the ego vehicle weaves through a traffic corridor. lateral offset is a fraction of road_half_width.
 */
inline std::vector<ParkedCarObstacle> generateParkedCarsAlongRoad(const mppi::path::Path2D& path,
                                                                  const float road_half_width,
                                                                  const unsigned int seed,
                                                                  const float along_spacing = 5.5F,
                                                                  const int max_cars = 48)
{
  std::vector<ParkedCarObstacle> cars;
  if (path.empty() || road_half_width <= 0.0F)
  {
    return cars;
  }

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist_jitter_s(-0.9F, 0.9F);
  std::uniform_real_distribution<float> dist_gap(0.0F, 1.0F);
  std::uniform_real_distribution<float> dist_lat_frac(0.58F, 0.88F);

  constexpr float kGapProbability = 0.24F;
  float s = along_spacing * 0.35F;
  int side_sign = 1;

  while (s < path.length() && static_cast<int>(cars.size()) < max_cars)
  {
    if (dist_gap(rng) < kGapProbability)
    {
      s += along_spacing * 0.55F;
      continue;
    }

    const float s_wrapped = path.wrapArcLength(s + dist_jitter_s(rng));
    const mppi::path::Pose2D p = path.poseAt(s_wrapped);
    float tx = 0.0F;
    float ty = 0.0F;
    path.tangentAt(s_wrapped, tx, ty);

    const float lateral = road_half_width * dist_lat_frac(rng);
    const float sign = (side_sign > 0) ? 1.0F : -1.0F;

    ParkedCarObstacle car;
    car.ox = p.x - sign * lateral * ty;
    car.oy = p.y + sign * lateral * tx;
    car.yaw = p.yaw;
    cars.push_back(car);

    side_sign = -side_sign;
    s += along_spacing + dist_jitter_s(rng) * 0.35F;
  }

  return cars;
}

/** Ego OBB from rear axle: box center offset forward from axle using wheel_base overhang model. */
template <int NUM_TIMESTEPS>
inline void setRacerCostEgoFootprint(RacerCostParams<NUM_TIMESTEPS>& params, const float wheel_base,
                                     const float ego_length, const float ego_width)
{
  params.ego_length = ego_length;
  params.ego_width = ego_width;
  const float rear_overhang = 0.08F * wheel_base;
  params.ego_axle_to_box_center = 0.5F * ego_length - rear_overhang;
}

template <int NUM_TIMESTEPS>
inline void fillRacerCostParkedCars(RacerCost<NUM_TIMESTEPS>& cost, const std::vector<ParkedCarObstacle>& cars)
{
  float obs_x[RacerCost<NUM_TIMESTEPS>::kMaxObstacles] = {};
  float obs_y[RacerCost<NUM_TIMESTEPS>::kMaxObstacles] = {};
  float obs_yaw[RacerCost<NUM_TIMESTEPS>::kMaxObstacles] = {};
  float obs_half_length[RacerCost<NUM_TIMESTEPS>::kMaxObstacles] = {};
  float obs_half_width[RacerCost<NUM_TIMESTEPS>::kMaxObstacles] = {};

  const int n =
      static_cast<int>(std::min(cars.size(), static_cast<size_t>(RacerCost<NUM_TIMESTEPS>::kMaxObstacles)));
  for (int i = 0; i < n; ++i)
  {
    const ParkedCarObstacle& car = cars[static_cast<size_t>(i)];
    obs_x[i] = car.ox;
    obs_y[i] = car.oy;
    obs_yaw[i] = car.yaw;
    obs_half_length[i] = car.length * 0.5F;
    obs_half_width[i] = car.width * 0.5F;
  }

  cost.setOrientedBoxObstacles(obs_x, obs_y, obs_yaw, obs_half_length, obs_half_width, n);
}

template <int NUM_TIMESTEPS>
inline void fillRacerCostFromPathReference(RacerCost<NUM_TIMESTEPS>& cost,
                                           const std::vector<mppi::path::PathReferenceSample>& ref)
{
  float ref_x[NUM_TIMESTEPS];
  float ref_y[NUM_TIMESTEPS];

  for (int t = 0; t < NUM_TIMESTEPS; ++t)
  {
    const size_t idx = ref.empty() ? 0U : static_cast<size_t>(std::min(t, static_cast<int>(ref.size()) - 1));
    ref_x[t] = ref[idx].x;
    ref_y[t] = ref[idx].y;
  }

  cost.setReferenceTrajectory(ref_x, ref_y, NUM_TIMESTEPS);
}

}  // namespace cost
}  // namespace mppi
