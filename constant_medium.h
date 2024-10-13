#pragma once

#include "hittable.h"
#include "material.h"
#include "texture.h"

class ConstantMedium final : public Hittable {
 public:
  //__host__ __device__ ConstantMedium(Hittable* boundary, float density,
  //               Isotropic* isotropic, curandState* rand_state)
  //    : boundary_(boundary),
  //      neg_inv_density_(-1 / density),
  //      phase_function_(isotropic),
  //      rand_state_(rand_state) {}

  __host__ __device__ ConstantMedium(Hittable* boundary, float density,
                  Isotropic* isotropic, curandState* rand_state)
      : boundary_(boundary),
        neg_inv_density_(-1 / density),
        phase_function_(isotropic),
        rand_state_(rand_state){}

  __device__ [[nodiscard]] HitResult DetectHit(
      const RayF& r, const IntervalF& ray_interval) const noexcept override {
 
    HitResult hit_result{};
    float rec1_t = 0.f;
    float rec2_t = 0.f;
    bool has_hit = false;

    const IntervalF i_universe{-1e+10f, 1e+10f};

    hit_result = boundary_->DetectHit(r, i_universe);
    rec1_t = hit_result.t;
    has_hit = hit_result.has_hit;

    if (!has_hit) 
      return hit_result;

    hit_result = boundary_->DetectHit(r, IntervalF(hit_result.t + 0.0001f, 1e+10f));
    rec2_t = hit_result.t;
    has_hit = hit_result.has_hit;

    if (!has_hit) 
      return hit_result;

    if (rec1_t < ray_interval.min) rec1_t = ray_interval.min;
    if (rec2_t > ray_interval.max) rec2_t = ray_interval.max;

    if (rec1_t >= rec2_t)
    {
      hit_result.has_hit = false;
      return hit_result;
    }

    if (rec1_t < 0) 
        rec1_t = 0;

    const auto ray_length = r.direction().Length();
    const auto distance_inside_boundary = (rec2_t - rec1_t) * ray_length;
    const auto hit_distance = neg_inv_density_ * std::log(GetRandomFloat(rand_state_));

    // From now I reuse the rec2 hit result as the final result in order to avoid
    // a GPU stack overflow by having a 3rd HitResult variable...

    if (hit_distance > distance_inside_boundary)
    {
      hit_result.has_hit = false;
      return hit_result;
    }

    hit_result.t = rec1_t + hit_distance / ray_length;
    hit_result.point = r.GetPointAt(rec2_t);
    hit_result.normal = Vec3F(1, 0, 0);    // arbitrary
    hit_result.front_face = true;        // also arbitrary
    hit_result.material = phase_function_;

    return hit_result;
  }

  __host__ __device__ [[nodiscard]] AABB GetBoundingBox()
      const noexcept override {
    return boundary_->GetBoundingBox();
  }

 private:
  Hittable* boundary_ = nullptr;
  float neg_inv_density_ = 0.f;
  Material* phase_function_ = nullptr;
  curandState* rand_state_ = nullptr;
};
