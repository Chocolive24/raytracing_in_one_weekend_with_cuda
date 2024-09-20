#pragma once

#include "color.h"
#include "hittable.h"
#include "device_random.h"
#include "material.h"

class Camera {
 public:
  __host__ __device__ constexpr Camera() noexcept = default;

  __host__ __device__ void Initialize() noexcept {
    // Determine viewport dimensions.
    constexpr auto theta = math_utility::DegreesToRadians(kFov);
    const auto h = std::tan(theta / 2);
    const auto viewport_height = 2 * h * focus_dist;

    // Viewport widths less than one are ok since they are real valued.
    // constexpr auto kViewportHeight = 2.0;
    const auto viewport_width =
        viewport_height * (static_cast<double>(kImageWidth) / kImageHeight);

    // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
    w = (look_from - look_at).Normalized();
    u = (v_up.CrossProduct(w)).Normalized();
    v = w.CrossProduct(u);

    // Calculate the vectors across the horizontal and down the vertical
    // viewportedges.
    const Vec3F viewport_u = viewport_width * u;  // Vector across viewport horizontal edge
    const Vec3F viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u = viewport_u / kImageWidth;
    pixel_delta_v = viewport_v / kImageHeight;

    // Calculate the location of the upper left pixel.
    const auto viewport_upper_left =
        look_from - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;

    pixel_00_loc =
        viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Calculate the camera defocus disk basis vectors.
    const auto defocus_radius =
        focus_dist * std::tan(math_utility::DegreesToRadians(defocus_angle / 2));
    defocus_disk_u = u * defocus_radius;
    defocus_disk_v = v * defocus_radius;
  }

  __device__  [[nodiscard]] static Color CalculatePixelColor(
      const RayF& r, Hittable** world, curandState* local_rand_state) noexcept {
    RayF cur_ray = r;
    Vec3F cur_attenuation{1.f, 1.f, 1.f};

    for (int i = 0; i < kMaxBounceCount; i++) {
     
      const HitResult hit_result = (*world)->DetectHit(cur_ray, 
          IntervalF(math_utility::kEpsilon, math_utility::kInfinity));
      
      if (hit_result.has_hit) {
        RayF scattered{};
        Color attenuation{};

        if (hit_result.material->Scatter(cur_ray, hit_result, attenuation, scattered, 
            local_rand_state)) {
          cur_attenuation *= attenuation;
          cur_ray = scattered;
        }
        else {
          return Vec3F{0.f, 0.f, 0.f};
        }
      }
      else
      {
        const Vec3F unit_direction = cur_ray.direction().Normalized();
        const float a = 0.5f * (unit_direction.y + 1.f);
        const auto color =
            (1.f - a) * Color(1.f, 1.f, 1.f) + a * Color(0.5f, 0.7f, 1.f);

        return cur_attenuation * color;
      }
    }

    return Vec3F{0.f, 0.f, 0.f}; // exceeded recursion.
  }

  __device__ [[nodiscard]] Vec3F DefocusDiskSample(curandState* local_rand_state) const noexcept {
    // Returns a random point in the camera defocus disk.
    const auto p = GetRandomVecInUnitDisk(local_rand_state);
    return look_from + (p.x * defocus_disk_u) + (p.y * defocus_disk_v);
  }

  __device__ [[nodiscard]] RayF GetRayAtLocation(
      const int x, const int y, curandState* local_rand_state) const noexcept {
    // Construct a camera ray originating from the defocus disk and directed at
    // a randomly sampled point around the pixel location i, j.
    const auto offset = Vec3F(curand_uniform(local_rand_state) - 0.5f,
                              curand_uniform(local_rand_state) - 0.5f, 0.f);
    const auto pixel_sample = pixel_00_loc +
                              ((x + offset.x) * pixel_delta_u) +
                              ((y + offset.y) * pixel_delta_v);

    const auto ray_origin = (defocus_angle <= 0) ? look_from :
                                  DefocusDiskSample(local_rand_state);
    const auto ray_direction = pixel_sample - ray_origin;
    const auto ray_time = curand_uniform(local_rand_state);

    return RayF{ray_origin, ray_direction, ray_time};
  }

  static constexpr float kAspectRatio = 1.f; // 16.f / 9.f;
  static constexpr int kImageWidth = 400;
  static constexpr int kImageHeight = static_cast<int>(kImageWidth / kAspectRatio);
  // Count of random samples for each pixel
  static constexpr short kSamplesPerPixel = 100;  
  // Color scale factor for a sum of pixel samples.
  static constexpr float kPixelSamplesScale = 1.f / kSamplesPerPixel;
  static constexpr int kMaxBounceCount = 50;
  static constexpr float kFov = 80.f; //20.f // Vertical view angle (field of view)
  float defocus_angle = 0.f; //0.6f // Variation angle of rays through each pixel
  float focus_dist = 10.f;  // Distance from camera lookfrom point to plane of perfect focus
 

  // My Vec3F class is undefined in the device code when used as constexpr and I don't
  // know why so it is not constexpr for the moment.
  Vec3F look_from = Vec3F(0, 0, 9); // Vec3F(13, 2, 3);  // look from.

  Vec3F look_at = Vec3F(0, 0, 0);  // Point camera is looking at
  Vec3F v_up = Vec3F(0, 1, 0);     // Camera-relative "up" direction

  Vec3F u{};
  Vec3F v{};
  Vec3F w{};  // Camera frame basis vectors

  Vec3F defocus_disk_u{};  // Defocus disk horizontal radius
  Vec3F defocus_disk_v{};  // Defocus disk vertical radius

  // Calculate the horizontal and vertical delta vectors from pixel to pixel.
  Vec3F pixel_delta_u{}; // Offset to pixel to the right
  Vec3F pixel_delta_v{}; // Offset to pixel below
  Vec3F pixel_00_loc{};
};