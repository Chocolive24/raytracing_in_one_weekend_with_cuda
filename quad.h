#pragma once

#include "hittable.h"

class Quad : public Hittable {
 public:
  __device__ Quad(const Vec3F& Q, const Vec3F& u, const Vec3F& v, Material* mat)
      : q_(Q), u_(u), v_(v), mat_(mat) {
    const auto n = u.CrossProduct(v);
    normal = n.Normalized();
    d_ = normal.DotProduct(Q);
    w_ = n / n.DotProduct(n);

    SetBoundingBox();
  }

  __device__ virtual void SetBoundingBox() noexcept {
    // Compute the bounding box of all four vertices.
    auto bbox_diagonal1 = AABB(q_, q_ + u_ + v_);
    auto bbox_diagonal2 = AABB(q_ + u_, q_ + v_);
    aabb_ = AABB(bbox_diagonal1, bbox_diagonal2);
  }

  __device__ [[nodiscard]] HitResult DetectHit(
      const RayF& r, const IntervalF& ray_interval) const noexcept override {
    HitResult hit_result{};
    const auto denom = normal.DotProduct(r.direction());

    // No hit if the ray is parallel to the plane.
    if (std::fabs(denom) < 1e-8) {
      hit_result.has_hit = false;
      return hit_result;
    }

    // Return false if the hit point parameter t is outside the ray interval.
    const auto t = (d_ - normal.DotProduct(r.origin())) / denom;

    if (!ray_interval.Contains(t)) {
      hit_result.has_hit = false;
      return hit_result;
    }

    // Determine if the hit point lies within the planar shape using its plane
    // coordinates.
    const auto intersection = r.GetPointAt(t);
    const Vec3F planar_hitpt_vector = intersection - q_;
    const auto alpha = w_.DotProduct(planar_hitpt_vector.CrossProduct(v_));
    const auto beta = w_.DotProduct(u_.CrossProduct(planar_hitpt_vector));

    if (!IsInterior(alpha, beta, hit_result)) {
      hit_result.has_hit = false;
      return hit_result;
    }

    // Ray hits the 2D shape; set the rest of the hit record and return true.

    hit_result.t = t;
    hit_result.point = intersection;
    hit_result.material = mat_;
    hit_result.SetFaceNormal(r, normal);
    hit_result.has_hit = true;

    return hit_result;
  }

  __device__ virtual [[nodiscard]] bool IsInterior(
      float a, float b, HitResult& rec) const noexcept {
    const IntervalF unit_interval = IntervalF(0, 1);
    // Given the hit point in plane coordinates, return false if it is outside
    // the primitive, otherwise set the hit record UV coordinates and return
    // true.

    if (!unit_interval.Contains(a) || !unit_interval.Contains(b)) return false;

    rec.tex_coord.u = a;
    rec.tex_coord.v = b;
    return true;
  }

  __host__ __device__ [[nodiscard]] AABB GetBoundingBox()
      const noexcept override {
    return aabb_;
  }

 private:
  Vec3F q_;
  Vec3F u_, v_;
  Vec3F w_{};
  Material* mat_ = nullptr;
  AABB aabb_{};
  Vec3F normal{};
  float d_ = 0.f;
};