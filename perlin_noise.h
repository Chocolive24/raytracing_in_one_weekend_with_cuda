#pragma once

#include "device_random.h"

// TODO: Refactor pour mettre en host après avoir compris comment passer mémoire
// de CPU à GPU.

class PerlinNoise {
 public:
  __device__ PerlinNoise(curandState* local_rand_state) noexcept {
    for (Vec3F& rand_vec : rand_vecs_) {
      // Transform random value in range [0 ; 1] to range [-1 ; 1].
      auto vec = 2.0f * GetRandomVector(local_rand_state) - Vec3F(1, 1, 1);
      rand_vec = vec.Normalized();
    }

    GeneratePermutation(perm_x_, local_rand_state);
    GeneratePermutation(perm_y_, local_rand_state);
    GeneratePermutation(perm_z_, local_rand_state);
  }

  __host__ __device__ [[nodiscard]] float GetNoiseValueAtPoint(
      const Vec3F& p) const noexcept {
    const float u = p.x - std::floor(p.x);
    const float v = p.y - std::floor(p.y);
    const float w = p.z - std::floor(p.z);

    const auto i = static_cast<int>(std::floor(p.x));
    const auto j = static_cast<int>(std::floor(p.y));
    const auto k = static_cast<int>(std::floor(p.z));
    Vec3F c[2][2][2];

    for (int di = 0; di < 2; di++)
      for (int dj = 0; dj < 2; dj++)
        for (int dk = 0; dk < 2; dk++)
          c[di][dj][dk] =
              rand_vecs_[perm_x_[(i + di) & 255] ^ perm_y_[(j + dj) & 255] ^
                         perm_z_[(k + dk) & 255]];

    return PerlinInterpolation(c, u, v, w);
  }

  __host__ __device__ [[nodiscard]] float ComputeTurbulence(
      const Vec3F& p, const int depth) const noexcept {
    float accum = 0.f;
    Vec3F temp_p = p;
    float weight = 1.f;

    for (int i = 0; i < depth; i++) {
      accum += weight * GetNoiseValueAtPoint(temp_p);
      weight *= 0.5f;
      temp_p *= 2;
    }

    return std::fabs(accum);
  }

 private:
  static constexpr int kPointCount_ = 256;
  Vec3F rand_vecs_[kPointCount_];
  int perm_x_[kPointCount_];
  int perm_y_[kPointCount_];
  int perm_z_[kPointCount_];

  __device__ static void GeneratePermutation(
      int* p, curandState* local_rand_state) noexcept {
    for (int i = 0; i < kPointCount_; i++) {
      p[i] = i;
    }

    Permute(p, kPointCount_, local_rand_state);
  }

  __device__ static void Permute(int* p, const int n,
                                 curandState* local_rand_state) noexcept {
    for (int i = n - 1; i > 0; i--) {
      const int target = GetRandomInt(0, i, local_rand_state);
      const int tmp = p[i];
      p[i] = p[target];
      p[target] = tmp;
    }
  }

  //__host__ __device__ static [[nodiscard]] float TrilinearInterpolation(
  //    float c[2][2][2], const float u, const float v, const float w) {
  //  float accum = 0.f;
  //  for (int i = 0; i < 2; i++)
  //    for (int j = 0; j < 2; j++)
  //      for (int k = 0; k < 2; k++)
  //        accum += (i * u + (1 - i) * (1 - u)) * (j * v + (1 - j) * (1 - v)) *
  //                 (k * w + (1 - k) * (1 - w)) * c[i][j][k];

  //  return accum;
  //}

  __host__ __device__ static [[nodiscard]] float PerlinInterpolation(
      const Vec3F c[2][2][2], const float u, const float v, const float w) {
    const float uu = u * u * (3 - 2 * u);
    const float vv = v * v * (3 - 2 * v);
    const float ww = w * w * (3 - 2 * w);
    float accum = 0.f;

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) {
          Vec3F weight_v(u - i, v - j, w - k);
          accum +=
              (i * uu + (1 - i) * (1 - uu)) * (j * vv + (1 - j) * (1 - vv)) *
              (k * ww + (1 - k) * (1 - ww)) * c[i][j][k].DotProduct(weight_v);
        }

    return accum;
  }
};
