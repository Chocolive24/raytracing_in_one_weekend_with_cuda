#pragma once

#include "interval.h"
#include "ray.h"
#include "vec3.h"

class AABB {
public:
  IntervalF x, y, z;

 // The default AABB is empty, since intervals are empty by default,
  __host__ __device__ AABB() noexcept = default;

  __host__ __device__ AABB(const IntervalF& x, const IntervalF& y,
                           const IntervalF& z) noexcept
      : x(x), y(y), z(z) {}

  __host__ __device__ AABB(const Vec3F& a, const Vec3F& b) noexcept {
    // Treat the two points a and b as extrema for the bounding box, so we don't
    // require a particular minimum/maximum coordinate order.

    x = (a.x <= b.x) ? IntervalF(a.x, b.x) : IntervalF(b.x, a.x);
    y = (a.y <= b.y) ? IntervalF(a.y, b.y) : IntervalF(b.y, a.y);
    z = (a.z <= b.z) ? IntervalF(a.z, b.z) : IntervalF(b.z, a.z);
  }

  __host__ __device__ AABB(const AABB& box0, const AABB& box1) {
    x = IntervalF(box0.x, box1.x);
    y = IntervalF(box0.y, box1.y);
    z = IntervalF(box0.z, box1.z);
  }

  __host__ __device__ [[nodiscard]] const IntervalF& AxisInterval(
      const int n) const noexcept {
    if (n == 1) return y;
    if (n == 2) return z;
    return x;
  }

  __device__ [[nodiscard]] bool Hit(const RayF& r,
                                             IntervalF ray_t) const noexcept {
    const Vec3F& ray_orig = r.origin();
    const Vec3F& ray_dir = r.direction();

    float axis_it = 0; // Iterator variable to iterate threw all 3D axis.

    for (int axis = 0; axis < 3; axis++) {
      const IntervalF& ax = AxisInterval(axis);

      switch (axis)
      {
        case 0:
          axis_it = ray_dir.x;
          break;
        case 1:
          axis_it = ray_dir.y;
          break;
        case 2:
          axis_it = ray_dir.z;
          break;
      default:
          break;
      }

      const float adinv = 1.f / axis_it;

      const auto t0 = (ax.min - axis_it) * adinv;
      const auto t1 = (ax.max - axis_it) * adinv;

      if (t0 < t1) {
        if (t0 > ray_t.min) ray_t.min = t0;
        if (t1 < ray_t.max) ray_t.max = t1;
      } else {
        if (t1 > ray_t.min) ray_t.min = t1;
        if (t0 < ray_t.max) ray_t.max = t0;
      }

      if (ray_t.max <= ray_t.min) return false;
    }
    return true;
  }
};
