/**
 * Host-side helpers to populate FirstOrderDubinsBicycleCost reference trajectories and obstacles.
 */
#pragma once

#include <mppi/cost_functions/dubins/first_order_dubins_bicycle_cost.cuh>
#include <mppi/cost_functions/parked_car_obstacles.hpp>
#include <mppi/dynamics/dubins/first_order_dubins_bicycle.cuh>
#include <mppi/path/path_reference_generator.hpp>

#include <algorithm>
#include <vector>

namespace mppi
{
namespace cost
{

template <int NUM_TIMESTEPS>
inline void fillFirstOrderDubinsBicycleCostGeometry(FirstOrderDubinsBicycleCostParams<NUM_TIMESTEPS>& cost_params,
                                                    const FirstOrderDubinsBicycleParams& dyn)
{
  cost_params.wheel_base = dyn.wheel_base;
  cost_params.accel_time_constant = dyn.accel_time_constant;
  cost_params.steer_time_constant = dyn.steer_time_constant;
}

template <int NUM_TIMESTEPS>
inline void setFirstOrderDubinsBicycleCostEgoFootprint(FirstOrderDubinsBicycleCostParams<NUM_TIMESTEPS>& params,
                                                       const float wheel_base, const float ego_length,
                                                       const float ego_width)
{
  params.ego_length = ego_length;
  params.ego_width = ego_width;
  const float rear_overhang = 0.08F * wheel_base;
  params.ego_axle_to_box_center = 0.5F * ego_length - rear_overhang;
}

template <int NUM_TIMESTEPS>
inline void fillFirstOrderDubinsBicycleCostParkedCars(FirstOrderDubinsBicycleCost<NUM_TIMESTEPS>& cost,
                                                      const std::vector<ParkedCarObstacle>& cars)
{
  float obs_x[FirstOrderDubinsBicycleCost<NUM_TIMESTEPS>::kMaxObstacles] = {};
  float obs_y[FirstOrderDubinsBicycleCost<NUM_TIMESTEPS>::kMaxObstacles] = {};
  float obs_yaw[FirstOrderDubinsBicycleCost<NUM_TIMESTEPS>::kMaxObstacles] = {};
  float obs_half_length[FirstOrderDubinsBicycleCost<NUM_TIMESTEPS>::kMaxObstacles] = {};
  float obs_half_width[FirstOrderDubinsBicycleCost<NUM_TIMESTEPS>::kMaxObstacles] = {};

  const int n = static_cast<int>(std::min(
      cars.size(), static_cast<size_t>(FirstOrderDubinsBicycleCost<NUM_TIMESTEPS>::kMaxObstacles)));
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

/** Per-horizon obstacle poses: buffers sized obstacle_count * num_timesteps (obstacle-major). */
template <int NUM_TIMESTEPS>
inline void fillFirstOrderDubinsBicycleCostObstacleTrajectories(FirstOrderDubinsBicycleCost<NUM_TIMESTEPS>& cost,
                                                                const float* x, const float* y, const float* yaw,
                                                                const float* half_length, const float* half_width,
                                                                const int obstacle_count, const int num_timesteps)
{
  cost.setOrientedBoxObstacleTrajectories(x, y, yaw, half_length, half_width, obstacle_count, num_timesteps);
}

template <int NUM_TIMESTEPS>
inline void fillFirstOrderDubinsBicycleCostFromPathReference(
    FirstOrderDubinsBicycleCost<NUM_TIMESTEPS>& cost, const std::vector<mppi::path::PathReferenceSample>& ref)
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
