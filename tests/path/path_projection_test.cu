#include <gtest/gtest.h>

#include <mppi/path/path2d.hpp>
#include <mppi/path/path_projection.hpp>
#include <mppi/path/path_reference_generator.hpp>

#include <cmath>

TEST(PathProjection, AnalyticCircleFootpointOnRadius)
{
  const mppi::path::Path2D path = mppi::path::Path2D::circle(0.0F, 0.0F, 10.0F, 0.0F);
  ASSERT_TRUE(path.isAnalyticCircle());
  const mppi::path::PathProjection proj = mppi::path::projectPoseOntoPath(path, 10.5F, 0.0F);
  EXPECT_NEAR(proj.footpoint.x, 10.0F, 1.0E-4F);
  EXPECT_NEAR(proj.footpoint.y, 0.0F, 1.0E-4F);
  EXPECT_NEAR(proj.distance, 0.5F, 1.0E-4F);
  EXPECT_NEAR(proj.signed_lateral_error, -0.5F, 1.0E-4F);
}

TEST(PathProjection, AnalyticCircleConstantCurvature)
{
  const mppi::path::Path2D path = mppi::path::Path2D::circle(0.0F, 0.0F, 20.0F, 0.0F);
  EXPECT_NEAR(path.curvatureAt(0.0F), 0.05F, 1.0E-6F);
  EXPECT_NEAR(path.curvatureAt(50.0F), 0.05F, 1.0E-6F);
}

TEST(PathProjection, StraightLineLateralOffset)
{
  const mppi::path::Path2D path = mppi::path::Path2D::straightLine(0.0F, 0.0F, 50.0F, 0.0F, 32);
  const mppi::path::PathProjection proj = mppi::path::projectPoseOntoPath(path, 25.0F, 2.0F);
  EXPECT_NEAR(proj.footpoint.x, 25.0F, 1.0E-3F);
  EXPECT_NEAR(proj.footpoint.y, 0.0F, 1.0E-3F);
  EXPECT_NEAR(proj.signed_lateral_error, 2.0F, 1.0E-3F);
}

TEST(PathReferenceGenerator, SpacingAdvancesArcLength)
{
  const mppi::path::Path2D path = mppi::path::Path2D::straightLine(0.0F, 0.0F, 100.0F, 0.0F, 64);
  mppi::path::PathReferenceGenerator gen(0.1F);
  gen.setSpeedCap(3.0F);
  const auto ref = gen.generate(path, 0.0F, 10);
  ASSERT_EQ(ref.size(), 10U);
  EXPECT_NEAR(ref[1].arc_length_s, 0.3F, 0.15F);
  EXPECT_NEAR(ref[1].t, 0.1F, 1.0E-5F);
}

TEST(PathReferenceGenerator, ClosedCircleWrapsArcLength)
{
  const mppi::path::Path2D path = mppi::path::Path2D::circle(0.0F, 0.0F, 20.0F, 0.0F);
  ASSERT_TRUE(path.closed());
  mppi::path::PathReferenceGenerator gen(0.1F);
  gen.setSpeedCap(3.0F);
  const float s_near_end = path.length() - 1.0F;
  const auto ref = gen.generate(path, s_near_end, 20);
  ASSERT_EQ(ref.size(), 20U);
  EXPECT_NEAR(ref[0].arc_length_s, s_near_end, 0.5F);
  EXPECT_NEAR(ref[1].arc_length_s, path.wrapArcLength(s_near_end + 0.3F), 0.2F);
}
