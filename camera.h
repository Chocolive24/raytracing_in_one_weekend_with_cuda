#pragma once

#include "color.h"
#include "hittable.h"

class Camera {
 public:
  __host__ __device__ constexpr Camera() noexcept = default;

  __host__ __device__ void Initialize() noexcept {
    // Determine viewport dimensions.
    constexpr float kFocalLength = 1.f;
    constexpr float kViewportHeight = 2.f;
    constexpr float kViewportWidth =
        kViewportHeight * (static_cast<float>(kImageWidth) / kImageHeight);

    // Calculate the vectors across the horizontal and down the vertical
    // viewport edges.
    constexpr auto kViewportU = Vec3F(kViewportWidth, 0, 0);
    constexpr auto kViewportV = Vec3F(0, -kViewportHeight, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u = kViewportU / kImageWidth;  // Offset to pixel to the right
    pixel_delta_v = kViewportV / kImageHeight;  // Offset to pixel below

    // Calculate the location of the upper left pixel.
    const auto kViewportUpperLeft = kLookFrom_ -
                                               Vec3F(0, 0, kFocalLength) -
                                               kViewportU / 2 - kViewportV / 2;
    pixel_00_loc = kViewportUpperLeft + 0.5 * (pixel_delta_u + pixel_delta_v);
  }

  __device__  [[nodiscard]] static Color GetRayColor(const RayF& r, Hittable** world) noexcept {
    const HitResult hit_result =
        (*world)->DetectHit(r, IntervalF(0.f, math_utility::kInfinity));

    if (hit_result.has_hit) {
      return 0.5f * (hit_result.normal + Color(1, 1, 1));
    }

    const Vec3F unit_direction = r.direction().Normalized();
    const float a = 0.5f * (unit_direction.y + 1.f);
    return (1.f - a) * Color(1.f, 1.f, 1.f) + a * Color(0.5f, 0.7f, 1.f);
  }

  __device__ [[nodiscard]] RayF GetRayAtLocation(const int x, const int y) const noexcept {
    const auto pixel_center = pixel_00_loc + (x * pixel_delta_u) +
                              (y * pixel_delta_v);
    const auto ray_direction = pixel_center - kLookFrom_;
    return RayF{kLookFrom_, ray_direction};
  }

  __device__ [[nodiscard]] static Vec3F SampleSquare() noexcept{
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    return Vec3F{math_utility::GetRandomDouble<float>() - 0.5f, 
        math_utility::GetRandomDouble<float>() - 0.5f, 0.f};
  }

  static constexpr float kAspectRatio = 16.f / 9.f;
  static constexpr int kImageWidth = 400;
  static constexpr int kImageHeight = static_cast<int>(kImageWidth / kAspectRatio);
  static constexpr short kSamplesPerPixel = 10;  // Count of random samples for each pixel
  static constexpr float kPixelSamplesScale = 1.f / kSamplesPerPixel; // Color scale factor for a sum of pixel samples.
  // My Vec3F class is undefined in the device code when used as constexpr and I don't
  // know why so it is not constexpr for the moment.
  Vec3F kLookFrom_ = Vec3F(0, 0, 0);  // look from.

  // Calculate the horizontal and vertical delta vectors from pixel to pixel.
  Vec3F pixel_delta_u{}; // Offset to pixel to the right
  Vec3F pixel_delta_v{}; // Offset to pixel below
  Vec3F pixel_00_loc{};
};