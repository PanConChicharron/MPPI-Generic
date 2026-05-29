#include <gtest/gtest.h>

#include <mppi/cost_functions/path_tracking/path_tracking_cost.cuh>
#include <mppi/path/path2d.hpp>
#include <mppi/path/path_reference_generator.hpp>
#include <mppi/path/path_tracking_bridge.hpp>

namespace
{
constexpr int kRefHorizon = 8;

using PathTrackingCostT = PathTrackingCost<kRefHorizon>;
using O = DubinsBicycleParams::OutputIndex;

PathTrackingCostT::output_array outputFromReference(const mppi::path::PathReferenceSample& r,
                                                    const float steer_ref = 0.0F)
{
  PathTrackingCostT::output_array output = PathTrackingCostT::output_array::Zero();
  output(static_cast<int>(O::POS_X)) = r.x;
  output(static_cast<int>(O::POS_Y)) = r.y;
  output(static_cast<int>(O::YAW)) = r.yaw;
  output(static_cast<int>(O::VEL_X)) = r.v;
  output(static_cast<int>(O::STEER_ANGLE)) = steer_ref;
  return output;
}

PathTrackingCostParams<kRefHorizon> makeStateOnlyCostParams()
{
  PathTrackingCostParams<kRefHorizon> p;
  mppi::path::fillPathTrackingCostWeights<kRefHorizon>(p, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F);
  return p;
}
}  // namespace

TEST(PathTrackingCost, ZeroWhenOutputMatchesPathReference)
{
  const mppi::path::Path2D path = mppi::path::Path2D::straightLine(0.0F, 0.0F, 50.0F, 0.0F, 32);
  mppi::path::PathReferenceGenerator ref_gen(0.1F);
  ref_gen.setSpeedCap(3.0F);
  const std::vector<mppi::path::PathReferenceSample> ref = ref_gen.generate(path, 0.0F, kRefHorizon, 2.5F);

  PathTrackingCostParams<kRefHorizon> params = makeStateOnlyCostParams();
  mppi::path::fillCostFromPathReference<kRefHorizon>(params, ref);

  PathTrackingCostT cost;
  cost.setParams(params);

  const PathTrackingCostT::control_array u = PathTrackingCostT::control_array::Zero();
  int crash = 0;

  for (int k = 0; k < kRefHorizon; ++k)
  {
    const PathTrackingCostT::output_array output = outputFromReference(ref[static_cast<size_t>(k)]);
    EXPECT_NEAR(cost.computeStateCost(output, k), 0.0F, 1.0E-6F) << "timestep " << k;
    EXPECT_NEAR(cost.computeRunningCost(output, u, k, &crash), 0.0F, 1.0E-6F) << "timestep " << k;
  }
}

TEST(PathTrackingCost, FirstStateErrorEqualsTotalStateCost)
{
  const mppi::path::Path2D path = mppi::path::Path2D::straightLine(0.0F, 0.0F, 50.0F, 0.0F, 32);
  mppi::path::PathReferenceGenerator ref_gen(0.1F);
  ref_gen.setSpeedCap(3.0F);
  const std::vector<mppi::path::PathReferenceSample> ref = ref_gen.generate(path, 0.0F, kRefHorizon, 2.5F);

  PathTrackingCostParams<kRefHorizon> params = makeStateOnlyCostParams();
  mppi::path::fillCostFromPathReference<kRefHorizon>(params, ref);

  PathTrackingCostT cost;
  cost.setParams(params);

  constexpr float kPosError = 1.5F;
  PathTrackingCostT::output_array output0 = outputFromReference(ref[0]);
  output0(static_cast<int>(O::POS_Y)) += kPosError;

  const float expected_first = params.w_pos * kPosError * kPosError;

  EXPECT_NEAR(cost.computeStateCost(output0, 0), expected_first, 1.0E-5F);

  float total = 0.0F;
  for (int k = 0; k < kRefHorizon; ++k)
  {
    PathTrackingCostT::output_array output = outputFromReference(ref[static_cast<size_t>(k)]);
    if (k == 0)
    {
      output(static_cast<int>(O::POS_Y)) += kPosError;
    }
    total += cost.computeStateCost(output, k);
  }
  EXPECT_NEAR(total, expected_first, 1.0E-5F);
}
