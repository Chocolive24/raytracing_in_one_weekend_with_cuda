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
__host__ __device__ [[nodiscard]] constexpr T DegreesToRadians(const T& degrees) noexcept {
  return degrees * kPi / 180.f;
}

__host__ __device__ [[nodiscard]] inline float SchlickApproxReflectance(
    const float cosine, const float refraction_index) noexcept {
  auto r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * std::pow((1.0f - cosine), 5.0f);
}

//template <typename T>
//__host__ __device__ [[nodiscard]] T GetRandomNumber() noexcept {
//  // Returns a random real in [0,1).
//  return rand() / (RAND_MAX + 1.0);
//}
//
//template<typename T>
//__host__ __device__ [[nodiscard]] T GetRandomNbrInRange(const T& min,
//                                                           const T& max) {
//  // Returns a random real in [min,max).
//  return min + (max - min) * GetRandomNumber<T>();
//}

}  // namespace math_utility
