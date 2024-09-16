#pragma once

#include "vec3.h"

// ===================================================================================
// The Ray class is only used by the GPU, that's why only the __device__
// keyword is used.
// Also note that current GPUs run fastest when they do calculations
// in single precision.
// Double precision calculations can be several times slower on some GPUs.
// That's why I will use RayF in the entire program.
// ===================================================================================

template<typename T>
class Ray {
 public:
  __device__ constexpr Ray() noexcept = default;

  __device__ Ray(const Vec3<T>& origin, const Vec3<T>& direction, T time = 0.f)
      : origin_(origin), direction_(direction), time_(time) {}

  __device__ [[nodiscard]] const Vec3<T>& origin() const noexcept { return origin_; }
  __device__ [[nodiscard]] const Vec3<T>& direction() const noexcept { return direction_; }
  __device__ [[nodiscard]] T time() const noexcept { return time_; }
  
  __host__ __device__ [[nodiscard]] Vec3<T> GetPointAt(const T& t) const {
    return origin_ + t * direction_;
  }

 private:
  Vec3<T> origin_{};
  Vec3<T> direction_{};
  T time_{0};
};

using RayF = Ray<float>;
// Commented to be sure to have 0 double precision instructions on the GPU.
//using RayD = Ray<double>;