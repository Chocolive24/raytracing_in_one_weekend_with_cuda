#pragma once

#include "hittable.h"

class HittableList final : public Hittable {
 public:
  __host__ __device__ HittableList() noexcept = default;
  __host__ HittableList(Hittable** objects, const int object_count, bool host) {
    objects_ = objects;
    object_count_ = object_count;

    for (int i = 0; i < object_count_; i++) {
      const Hittable* obj = objects_[i];

      aabb_ = AABB(aabb_, obj->GetBoundingBox());
    }
  }

  __host__ __device__ HittableList(Hittable** objects, const int object_count) {
    objects_ = objects;
    object_count_ = object_count;

    for (int i = 0; i < object_count_; i++) {
      const Hittable* obj = objects_[i];

      aabb_ = AABB(aabb_, obj->GetBoundingBox());
    }
  }

  __device__  [[nodiscard]] HitResult DetectHit(
      const RayF& r, const IntervalF& ray_interval) const noexcept override {
    HitResult result{};
    auto closest_so_far = ray_interval.max;

    for (int i = 0; i < object_count_; i++) {
      const auto object = objects_[i];
      const HitResult temp_result =
          object->DetectHit(r, IntervalF(ray_interval.min, closest_so_far));
      if (temp_result.has_hit) {
        closest_so_far = temp_result.t;
        result = temp_result;
      }
    }

    return result;
  }

  __host__  __device__ [[nodiscard]] AABB GetBoundingBox()
      const noexcept override {
    return aabb_;
  }

  __host__ __device__ [[nodiscard]] Hittable** objects() const noexcept {
    return objects_;
  }
  __host__ __device__ [[nodiscard]] int object_count() const noexcept {
    return object_count_;
  }

private:
  Hittable** objects_ = nullptr;
  int object_count_ = 0;
  AABB aabb_{};
};