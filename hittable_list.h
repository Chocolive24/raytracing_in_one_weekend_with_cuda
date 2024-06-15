#pragma once

#include "hittable.h"

class HittableList final : public Hittable {
 public:
  __device__ constexpr HittableList() noexcept = default;
  __device__ HittableList(Hittable** objects, const int object_count) {
    objects_ = objects;
    object_count_ = object_count;
  }

  //__device__ void Clear() { objects.clear(); }

  //__device__ void Add(const std::shared_ptr<Hittable>& object) {
  //  objects.push_back(object);
  //}

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

private:
  //std::vector<std::shared_ptr<Hittable>> objects{};
  Hittable** objects_ = nullptr;
  int object_count_ = 0;
};