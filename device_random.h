#pragma once

#include "vec3.h"

#include <curand_kernel.h>

#define RANDOM (curand_uniform(&local_rand_state))

__device__ [[nodiscard]] inline int GetRandomInt(
    [[maybe_unused]] const int min, const int max,
    curandState* local_rand_state) noexcept {
  // RANDOM only returns value between 0 and 1 in float. Because we want a rnd
  // int
  // between 0 and 2, we need to multiply the rnd result by 3 to get a number
  // between 0 and 3 and then we can cast it in int to have a number between 0
  // and 2.
  return static_cast<int>(curand_uniform(local_rand_state) * (max + 1));
}

__device__ [[nodiscard]] inline Vec3F GetRandomVector(
    curandState* local_rand_state) noexcept {
  return Vec3F{curand_uniform(local_rand_state),
               curand_uniform(local_rand_state),
               curand_uniform(local_rand_state)};
}

__device__ [[nodiscard]] inline Vec3F GetRandVecInUnitSphere(
    curandState* local_rand_state) noexcept {
  Vec3F p{};
  do {
    // Transform random value in range [0 ; 1] to range [-1 ; 1].
    p = 2.0f * GetRandomVector(local_rand_state) - Vec3F(1, 1, 1);
  } while (p.LengthSquared() >= 1.0f);
  return p;
}

__device__ [[nodiscard]] inline Vec3F GetRandVecOnHemisphere(
    curandState* local_rand_state, const Vec3F& hit_normal) noexcept {
  const Vec3F on_unit_sphere = GetRandVecInUnitSphere(local_rand_state).Normalized();
  if (on_unit_sphere.DotProduct(hit_normal) > 0.f) // In the same hemisphere as the normal
    return on_unit_sphere;

  return -on_unit_sphere;
}

__device__ [[nodiscard]] inline Vec3F GetRandomVecInUnitDisk(curandState* local_rand_state) {
  while (true) {
    const auto p = 2.f * Vec3F(curand_uniform(local_rand_state),
                         curand_uniform(local_rand_state), 0) - Vec3F(1, 1, 0);
    if (p.LengthSquared() < 1.f) 
        return p;
  }
}