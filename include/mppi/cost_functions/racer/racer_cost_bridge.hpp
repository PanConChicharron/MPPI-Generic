/**
 * Host-side helpers to populate RacerCost reference trajectories and obstacles.
 */
#pragma once

#include <mppi/cost_functions/racer/racer_cost.cuh>
#include <mppi/path/path_reference_generator.hpp>

#include <algorithm>
#include <vector>

namespace mppi
{
namespace cost
{

struct RacerCostObstacle
{
  float ox = 0.0F;
  float oy = 0.0F;
  float r = 0.0F;

  RacerCostObstacle() = default;

  RacerCostObstacle(const float ox_in, const float oy_in, const float r_in) : ox(ox_in), oy(oy_in), r(r_in)
  {
  }
};

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

template <int NUM_TIMESTEPS>
inline void fillRacerCostObstacles(RacerCost<NUM_TIMESTEPS>& cost, const std::vector<RacerCostObstacle>& obstacles)
{
  float obs_x[RacerCost<NUM_TIMESTEPS>::kMaxObstacles] = {};
  float obs_y[RacerCost<NUM_TIMESTEPS>::kMaxObstacles] = {};
  float obs_r[RacerCost<NUM_TIMESTEPS>::kMaxObstacles] = {};

  const int n = static_cast<int>(
      std::min(obstacles.size(), static_cast<size_t>(RacerCost<NUM_TIMESTEPS>::kMaxObstacles)));
  for (int i = 0; i < n; ++i)
  {
    obs_x[i] = obstacles[static_cast<size_t>(i)].ox;
    obs_y[i] = obstacles[static_cast<size_t>(i)].oy;
    obs_r[i] = obstacles[static_cast<size_t>(i)].r;
  }

  cost.setObstacles(obs_x, obs_y, obs_r, n);
}

}  // namespace cost
}  // namespace mppi
