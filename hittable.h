#pragma once

#include "aabb.h"
#include "interval.h"
#include "ray.h"

class Material;

struct TextureCoord {
  float u = 0;
  float v = 0;
};

class HitResult {
 public:
  Vec3F point{};
  Vec3F normal{};
  Material* material = nullptr;
  float t = 0;
  TextureCoord tex_coord{};
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
  __host__ __device__ constexpr Hittable() noexcept = default;
  __host__ __device__ Hittable(Hittable&& other) noexcept = default;
  __host__ __device__ Hittable& operator=(Hittable&& other) noexcept = default;
  __host__ __device__ Hittable(const Hittable& other) noexcept = default;
  __host__ __device__ Hittable& operator=(const Hittable& other) noexcept = default;
  __host__ __device__ virtual ~Hittable() = default;

  __device__ [[nodiscard]] virtual HitResult DetectHit(const RayF& r, 
      const IntervalF& ray_interval) const noexcept = 0;

  __host__ __device__ virtual [[nodiscard]] AABB GetBoundingBox() const noexcept = 0;
};

class Sphere final : public Hittable {
 public:
  __host__ __device__ Sphere() noexcept = default;

  // Stationary sphere.
  __host__ __device__ Sphere(const Vec3F& static_center, const float radius,
                            Material* material) noexcept
      : center_(static_center, Vec3F{0.f, 0.f, 0.f}),
        radius_(radius),
        is_moving_(false),
        material(material) {
    const auto r_vec = Vec3F(radius, radius, radius);
    aabb_ = AABB(static_center - r_vec, static_center + r_vec);
  }

  // Moving sphere.
  __host__ __device__ Sphere(const Vec3F& center_1,
                    const Vec3F& center_2, const float radius,
                    Material* material) noexcept
      : center_(center_1, center_2 - center_1),
        radius_(radius),
        is_moving_(true), material(material)
  {
    //center_vec_ = center_2 - center_1;

    const auto r_vec = Vec3F(radius, radius, radius);
    const AABB box1(center_.GetPointAt(0) - r_vec, center_.GetPointAt(0) + r_vec);
    const AABB box2(center_.GetPointAt(1) - r_vec, center_.GetPointAt(1) + r_vec);
    aabb_ = AABB(box1, box2);
  }

  __device__ [[nodiscard]] HitResult DetectHit(
      const RayF& r, const IntervalF& ray_interval) const noexcept override
    {
      //const auto current_center = is_moving_ ? LerpSphereCenter(r.time()) : center_1_;
      const auto current_center = center_.GetPointAt(r.time());
      const Vec3F oc = current_center - r.origin();
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
      const Vec3F outward_normal = (hit_result.point - current_center) / radius_;
      hit_result.SetFaceNormal(r, outward_normal);
      hit_result.tex_coord = ComputeSphereUv(outward_normal);
      hit_result.has_hit = true;
      hit_result.material = material;

      return hit_result;
    }

   __device__ static TextureCoord ComputeSphereUv(const Vec3F& p) noexcept {
      // p: a given point on the sphere of radius one, centered at the origin.
      // u: returned value [0,1] of angle around the Y axis from X=-1.
      // v: returned value [0,1] of angle from Y=-1 to Y=+1.
      //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
      //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
      //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

      const auto theta = acos(-p.y);
      const auto phi = atan2(-p.z, p.x) + math_utility::kPi;

      TextureCoord tex_coord;

      tex_coord.u = phi / (2 * math_utility::kPi);
      tex_coord.v = theta / math_utility::kPi;

      return tex_coord;
  }

  __host__ __device__ [[nodiscard]] AABB GetBoundingBox()
      const noexcept override {
    return aabb_;
  }

 private:
    //__device__ [[nodiscard]] Vec3F LerpSphereCenter(const float time) const noexcept {
    //  // Linearly interpolate from center1 to center2 according to time, where
    //  // t=0 yields center1, and t=1 yields center2.
    //  return center_1_ + time * center_vec_;
    //}

  AABB aabb_{};
  //Vec3F center_1_{};
  //Vec3F center_vec_{};
  RayF center_{};
  float radius_ = 0.f;
  bool is_moving_ = false;

public:
  Material* material = nullptr;
};