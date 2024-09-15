#pragma once

#include "math_utility.h"

template<typename T>
class Interval {
 public:
  __host__ __device__ constexpr Interval() noexcept = default;
  __host__ __device__ constexpr Interval(const T& min, const T& max)
      : min(min), max(max) {}

  __host__ __device__ Interval(const Interval& a, const Interval& b)  noexcept{
    // Create the interval tightly enclosing the two input intervals.
    min = a.min <= b.min ? a.min : b.min;
    max = a.max >= b.max ? a.max : b.max;
  }

  __host__ __device__ [[nodiscard]] constexpr T Size() const noexcept {
    return max - min;
  }

  __host__ __device__ [[nodiscard]] constexpr bool Contains(
      const T& x) const noexcept {
    return min <= x && x <= max;
  }

  __host__ __device__ [[nodiscard]] constexpr bool Surrounds(
      const T& x) const noexcept {
    return min < x && x < max;
  }

  __host__ __device__ [[nodiscard]] constexpr T Clamp(const T& x) const {
    if (x < min) return min;
    if (x > max) return max;
    return x;
  }

  __host__ __device__ [[nodiscard]] static constexpr Interval empty() noexcept {
    return Interval{+math_utility::kInfinity, -math_utility::kInfinity};
  }

  __host__ __device__ [[nodiscard]] static constexpr Interval universe() noexcept {
    return Interval{-math_utility::kInfinity, +math_utility::kInfinity};
  }

  __host__ __device__ Interval Expand(const float delta) const noexcept {
    float padding = delta / 2;
    return Interval(min - padding, max + padding);
  }

  T min = +math_utility::kInfinity;
  T max = -math_utility::kInfinity;
};

using IntervalF = Interval<float>;