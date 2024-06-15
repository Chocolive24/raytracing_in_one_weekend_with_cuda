#pragma once

#include "interval.h"
#include "ray.h"

class Material;

class HitResult {
 public:
  Vec3F point{};
  Vec3F normal{};
  //std::shared_ptr<Material> material = nullptr;
  float t = 0;
  bool has_hit = false;
  bool front_face;

  /**
   * \brief Sets the hit record normal vector.
    // NOTE: the parameter `outward_normal` is assumed to have unit length.
   * \param r The ray.
   * \param outward_normal The outward normal.
   */
  __device__ void SetFaceNormal(const RayF& r, const Vec3F& outward_normal) {
    front_face = r.direction().DotProduct(outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class Hittable {
 public:
  __device__ constexpr Hittable() noexcept = default;
  __device__ Hittable(Hittable&& other) noexcept = default;
  __device__ Hittable& operator=(Hittable&& other) noexcept = default;
  __device__ Hittable(const Hittable& other) noexcept = default;
  __device__ Hittable& operator=(const Hittable& other) noexcept = default;
  __device__ virtual ~Hittable() = default;

  __device__ [[nodiscard]] virtual HitResult DetectHit(const RayF& r, 
      const IntervalF& ray_interval) const noexcept = 0;
};

class Sphere final : public Hittable {
 public:
  __device__ constexpr Sphere() noexcept = default;
  __device__ Sphere(const Vec3F& center, const float radius) noexcept
      : center_(center), radius_(radius) {}

   __device__ [[nodiscard]] HitResult DetectHit(
      const RayF& r, const IntervalF& ray_interval) const noexcept override
    {
      const Vec3F oc = center_ - r.origin();
      const auto a = r.direction().LengthSquared();
      const auto h = r.direction().DotProduct(oc);
      const auto c = oc.LengthSquared() - radius_ * radius_;
      const auto discriminant = h * h - a * c;

      HitResult hit_result{};

      // If the ray didn't hit the sphere, assume the normal is -1.
      if (discriminant < 0) {
        hit_result.has_hit = false;
        return hit_result;
      }

      const auto sqrt_d = std::sqrt(discriminant);

      // Find the nearest root that lies in the acceptable range.
      auto root = (h - sqrt_d) / a;
      if (!ray_interval.Surrounds(root)) {
        root = (h + sqrt_d) / a;
        if (!ray_interval.Surrounds(root)) {
          hit_result.has_hit = false;
          return hit_result;
        }
      }

      hit_result.t = root;
      hit_result.point = r.GetPointAt(hit_result.t);
      const Vec3F outward_normal = (hit_result.point - center_) / radius_;
      hit_result.SetFaceNormal(r, outward_normal);
      hit_result.has_hit = true;
      //hit_result.material = material_;

      return hit_result;
    }

 private:
  Vec3F center_{};
  float radius_ = 0.f;
  //std::shared_ptr<Material> material_ = nullptr;
};