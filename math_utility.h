#pragma once

#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>

namespace math_utility {

// Constants
static constexpr float kInfinity = std::numeric_limits<float>::infinity();
static constexpr float kPi = 3.1415926535897932385f;
static constexpr float kEpsilon = 0.001f;

// Utility Functions
template<typename T>
[[nodiscard]] constexpr T DegreesToRadians(const T& degrees) noexcept {
  return degrees * kPi / 180.f;
}

template <typename T>
[[nodiscard]] T SchlickApproxReflectance(const T& cosine, const T& refraction_index) noexcept {
  auto r0 = (1 - refraction_index) / (1 + refraction_index);
  r0 = r0 * r0;
  return r0 + (1 - r0) * std::pow((1 - cosine), 5);
}

template <typename T>
[[nodiscard]] T GetRandomDouble() noexcept {
  // Returns a random real in [0,1).
  return rand() / (RAND_MAX + 1.0);
}

template<typename T>
[[nodiscard]] T GetRandomDoubleInRange(const T& min, const T& max) {
  // Returns a random real in [min,max).
  return min + (max - min) * GetRandomDouble<T>();
}

}  // namespace math_utility
