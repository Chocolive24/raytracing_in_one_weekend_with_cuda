#pragma once

#include <cmath>
#include <ostream>
#include <crt/host_defines.h>

// ===================================================================================
// The use of __host__ and __device__ enable the methods to be executed on
// the CPU and the GPU.
// Also note that current GPUs run fastest when they do calculations in single
// precision. Double precision calculations can be several times slower on some
// GPUs. That's why I will use Vec3F in the entire program.
// ===================================================================================

template<typename T>
class Vec3 {
public:
  __host__ __device__ constexpr Vec3<T>() noexcept = default;
  __host__ __device__ constexpr Vec3<T>(const T x, const T y, const T z) noexcept
      : x(x),
        y(y),
        z(z) {}

  __host__ __device__ [[nodiscard]] constexpr Vec3<T> operator-()
      const noexcept {
    return Vec3<T>(-x, -y, -z);
  }

  __host__ __device__ [[nodiscard]] constexpr Vec3<T> operator+(
      const Vec3<T>& v) const noexcept {
    return Vec3<T>(x + v.x, y + v.y, z + v.z);
  }

  __host__ __device__ constexpr Vec3<T> operator+=(const Vec3<T>& v) noexcept {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }

  __host__ __device__ [[nodiscard]] constexpr Vec3<T> operator-(
      const Vec3<T>& v) const noexcept {
    return Vec3<T>(x - v.x, y - v.y, z - v.z);
  }

  __host__ __device__ constexpr Vec3<T> operator-=(const Vec3<T>& v) noexcept {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }

  __host__ __device__ [[nodiscard]] constexpr Vec3<T> operator*(
      const T& scalar) const noexcept {
    return Vec3<T>(x * scalar, y * scalar, z * scalar);
  }

  __host__ __device__ [[nodiscard]] constexpr Vec3<T> operator*(
      const Vec3<T>& v) const noexcept {
    return Vec3<T>(x * v.x, y * v.y, z * v.z);
  }

  __host__ __device__  constexpr Vec3<T> operator*=(const T& scalar) noexcept {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
  }

  __host__ __device__ constexpr Vec3<T> operator*=(const Vec3<T>& v) noexcept {
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
  }

  __host__ __device__ [[nodiscard]] constexpr Vec3<T> operator/(
      const T& scalar) const noexcept {
    return Vec3<T>(x / scalar, y / scalar, z / scalar);
  }

  __host__ __device__  constexpr Vec3<T> operator/=(const T& scalar) noexcept {
    return *this *= 1.f/scalar;
  }

  __host__ __device__ [[nodiscard]] constexpr T LengthSquared() const noexcept {
    return static_cast<T>(x * x + y * y + z * z);
  }

  __host__ __device__ [[nodiscard]] T Length() const noexcept {
    return std::sqrt(LengthSquared());
  }

  __host__ __device__ [[nodiscard]] constexpr T DotProduct(
      const Vec3<T>& v) const noexcept {
    return x * v.x + y * v.y + z * v.z;
  }

  __host__ __device__ [[nodiscard]] constexpr Vec3<T> CrossProduct(
      const Vec3<T>& v) const noexcept {
    return Vec3<T>(y * v.z - z * v.y,
                   z * v.x - x * v.z,
                   x * v.y - y * v.x);
  }

  __host__ __device__ [[nodiscard]] Vec3<T> Normalized() const noexcept {
    return *this / Length();
  }

  __host__ __device__ [[nodiscard]] constexpr Vec3<T> Reflect(const Vec3<T>& v) const noexcept {
    return *this - 2.f * DotProduct(v) * v;
  }

  __host__ __device__ [[nodiscard]] Vec3<T> Refract(
      const Vec3<T>& n, const T& etai_over_etat) const noexcept {
    const auto cos_theta = std::fmin(-(*this).DotProduct(n), 1.f);
    const auto r_out_perp = etai_over_etat * (*this + cos_theta * n);
    const auto r_out_parallel = -std::sqrt(std::fabs(1.f - r_out_perp.LengthSquared())) * n;
    return r_out_perp + r_out_parallel;
  }

  __host__ __device__ [[nodiscard]] constexpr bool IsNearZero() const noexcept {
    // Return true if the vector is close to zero in all dimensions.
    constexpr float s = 1e-8f;
    return (std::fabs(x) < s) && (std::fabs(y) < s) && (std::fabs(z) < s);
  }

  //__host__ __device__ [[nodiscard]] static Vec3<T> random() noexcept {
  //  return Vec3<T>(random_double(), random_double(), random_double());
  //}

  //__host__ __device__ [[nodiscard]] static Vec3<T> random(
  //    const T& min, const T& max) noexcept {
  //  return Vec3<T>(random_double(min, max), random_double(min, max),
  //                 random_double(min, max));
  //}

  //__host__ __device__ [[nodiscard]] static Vec3<T> random_in_unit_sphere() {
  //  while (true) {
  //    const auto p = random(-1, 1);
  //    if (p.LengthSquared() < 1) 
  //        return p;
  //  }
  //}

  //__host__ __device__ [[nodiscard]] static Vec3<T> random_in_unit_disk() {
  //  while (true) {
  //    const auto p = Vec3<T>(random_double(-1, 1), random_double(-1, 1), 0);
  //    if (p.LengthSquared() < 1) 
  //        return p;
  //  }
  //}

  //__host__ __device__ [[nodiscard]] static Vec3<T> random_unit() {
  //  return random_in_unit_sphere().Normalized();
  //}

  //__host__ __device__ [[nodiscard]] static Vec3<T> random_in_hemisphere(
  //    const Vec3<T>& normal) {
  //  const auto on_unit_sphere = random_unit();
  //  if (on_unit_sphere.DotProduct(normal) > 0.0)  // In the same hemisphere as the normal
  //    return on_unit_sphere;
  //  else
  //    return -on_unit_sphere;
  //}

  T x = 0;
  T y = 0;
  T z = 0;
};

template <typename T>
constexpr std::ostream& operator<<(std::ostream& out, const Vec3<T>& v) noexcept {
  return out << v.x << ' ' << v.y << ' ' << v.z;
}

template<typename T>
__host__ __device__ [[nodiscard]] constexpr Vec3<T> operator-(
    const Vec3<T>& p, const Vec3<T>& v) noexcept {
  return Vec3<T>(p.x - v.x, p.y - p.y, p.z - v.z);
}

template<typename T, typename U>
__host__ __device__ [[nodiscard]] constexpr Vec3<T> operator*(const U& scalar,
                                          const Vec3<T>& v) noexcept {
  return Vec3<T>(v.x * scalar, v.y * scalar, v.z * scalar);
}

using Vec3F = Vec3<float>;
// Commented to be sure to have 0 double precision instructions on the GPU.
// using Vec3D = Vec3<double>;