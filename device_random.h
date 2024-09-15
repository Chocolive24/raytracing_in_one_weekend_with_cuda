#pragma once

#include "vec3.h"

#include <curand_kernel.h>

#define RANDOM (curand_uniform(&local_rand_state))

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