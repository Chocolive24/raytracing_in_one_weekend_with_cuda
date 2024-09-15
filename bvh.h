#pragma once

#include "device_random.h"
#include "hittable.h"
#include "hittable_list.h"

#include <algorithm>

class BVH_Node final : public Hittable {
 public:
  __device__ BVH_Node(HittableList list, curandState* local_rand_state)
      : BVH_Node(list.objects(), 0, list.object_count(), local_rand_state) {
    // There's a C++ subtlety here. This constructor (without span indices)
    // creates an implicit copy of the hittable list, which we will modify. The
    // lifetime of the copied list only extends until this constructor exits.
    // That's OK, because we only need to persist the resulting bounding volume
    // hierarchy.
  }

 __device__ BVH_Node(Hittable** objects, std::size_t start, std::size_t end,
           curandState* local_rand_state) {

    // RANDOM only returns value between 0 and 1 in float. Because we want a rnd int
    // between 0 and 2, we need to multiply the rnd result by 3 to get a number between
    // 0 and 3 and then we can cast it in int to have a number between 0 and 2.
    const int axis = 0;  //static_cast<int>(RANDOM * 3);

    const auto comparator = 
        (axis == 0) ? BoxCompareX : (axis == 1) ? BoxCompareY : BoxCompareZ;

    const size_t object_span = end - start;

    if (object_span == 1) {
      left_ = right_ = objects[start];
    }
    else if (object_span == 2) {
      left_ = objects[start];
      right_ = objects[start + 1];
    }
    else {
      /*std::sort(objects[0] + start, objects[0] + end,
                comparator);*/

      const auto mid = start + object_span / 2;
      left_ = new BVH_Node(objects, start, mid, local_rand_state);
      right_ = new BVH_Node(objects, mid, end, local_rand_state);
    }

    aabb_ = AABB(left_->GetBoundingBox(), right_->GetBoundingBox());
  }

   __device__ [[nodiscard]] HitResult DetectHit(
      const RayF& r, const IntervalF& ray_interval) const noexcept override {
    HitResult hit_result{};

    if (!aabb_.Hit(r, ray_interval))
    {
      hit_result.has_hit = false;
      return hit_result;
    }

    hit_result = left_->DetectHit(r, ray_interval);
    const bool hit_left = hit_result.has_hit;

    hit_result = right_->DetectHit(r, IntervalF(ray_interval.min,
                     hit_left ? hit_result.t : ray_interval.max));
    const bool hit_right = hit_result.has_hit;

    hit_result.has_hit = hit_left || hit_right;

    return hit_result;
  }

  __device__ [[nodiscard]] AABB GetBoundingBox() const noexcept override {
    return aabb_;
  }

private:
  Hittable* left_ = nullptr;
  Hittable* right_ = nullptr;
  AABB aabb_{};

  __device__ static bool BoxCompare(const Hittable* a,
                          const Hittable* b, int axis_index) {
    const auto a_axis_interval = a->GetBoundingBox().AxisInterval(axis_index);
    const auto b_axis_interval = b->GetBoundingBox().AxisInterval(axis_index);
    return a_axis_interval.min < b_axis_interval.min;
  }

  __device__ static bool BoxCompareX(const Hittable* a,
                            const Hittable* b) {
    return BoxCompare(a, b, 0);
  }

  __device__ static bool BoxCompareY(const Hittable* a,
                            const Hittable* b) {
    return BoxCompare(a, b, 1);
  }

  __device__ static bool BoxCompareZ(const Hittable* a,
                            const Hittable* b) {
    return BoxCompare(a, b, 2);
  }
};