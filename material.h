#pragma once

#include "color.h"
#include "hittable.h"
#include "ray.h"
#include "device_random.h"
#include "texture.h"

class Material {
 public:
  __device__ constexpr Material() noexcept = default;
  __device__ Material(Material&& other) noexcept = default;
  __device__ Material& operator=(Material&& other) noexcept = default;
  __device__ Material(const Material& other) noexcept = default;
  __device__ Material& operator=(const Material& other) noexcept = default;
  __device__ virtual ~Material() noexcept = default;

  __device__ virtual bool Scatter(const RayF& r_in, const HitResult& hit,
                                  Color& attenuation, RayF& scattered,
                                  curandState* local_rand_state) const = 0;
};

class Lambertian final : public Material {
 public:
  //__device__ Lambertian(const Color& albedo) noexcept : texture_(new SolidColor(albedo)) {}
  __device__ Lambertian(Texture* texture) noexcept : texture_(texture) {}
  __device__ Lambertian(Lambertian&& other) noexcept = default;
  __device__ Lambertian& operator=(Lambertian&& other) noexcept = default;
  __device__ Lambertian(const Lambertian& other) noexcept = default;
  __device__ Lambertian& operator=(const Lambertian& other) noexcept = default;
  __device__ ~Lambertian() noexcept override = default;

  __device__ bool Scatter(const RayF& r_in, const HitResult& hit,
                          Color& attenuation, RayF& scattered,
                          curandState* local_rand_state) const override {
    auto scatter_direction =
        hit.normal + GetRandVecInUnitSphere(local_rand_state).Normalized();

    // Catch degenerate scatter direction
    if (scatter_direction.IsNearZero()) 
        scatter_direction = hit.normal;

    scattered = RayF(hit.point, scatter_direction, r_in.time());
    attenuation = texture_->ComputeColor(hit.tex_coord.u, hit.tex_coord.v, hit.point);
    return true;
  }

 private:
  //Color albedo_{};
  Texture* texture_ = nullptr;
};

class Metal final : public Material {
 public:
  __device__ constexpr Metal(const Color& albedo, const float fuzz) noexcept
      : albedo_(albedo), fuzz_(fuzz) {}
  __device__ Metal(Metal&& other) noexcept = default;
  __device__ Metal& operator=(Metal&& other) noexcept = default;
  __device__ Metal(const Metal& other) noexcept = default;
  __device__ Metal& operator=(const Metal& other) noexcept = default;
  __device__ ~Metal() noexcept override = default;

  __device__ bool Scatter(const RayF& r_in, const HitResult& hit,
                          Color& attenuation, RayF& scattered,
                          curandState* local_rand_state) const override {
    Vec3F reflected = r_in.direction().Reflect(hit.normal);
    reflected = reflected.Normalized() + 
        (fuzz_ * GetRandVecInUnitSphere(local_rand_state).Normalized());
    scattered = RayF(hit.point, reflected, r_in.time());
    attenuation = albedo_;

    return scattered.direction().DotProduct(hit.normal) > 0;
  }

 private:
  Color albedo_{};
  float fuzz_ = 0.f;
};

class Dielectric final : public Material {
 public:
  __device__ Dielectric(const float refraction_index)
      : refraction_index_(refraction_index) {}
  __device__ Dielectric(Dielectric&& other) noexcept = default;
  __device__ Dielectric& operator=(Dielectric&& other) noexcept = default;
  __device__ Dielectric(const Dielectric& other) noexcept = default;
  __device__ Dielectric& operator=(const Dielectric& other) noexcept = default;
  __device__ ~Dielectric() noexcept override = default;

  __device__ bool Scatter(const RayF& r_in, const HitResult& hit, Color& attenuation, RayF& scattered,
                          curandState* local_rand_state) const override {
    attenuation = Color(1.0, 1.0, 1.0);
    const float ri = hit.front_face ? (1.f / refraction_index_) : refraction_index_;

    const Vec3F unit_direction = r_in.direction().Normalized();
    const float cos_theta = std::fmin(-unit_direction.DotProduct(hit.normal), 1.0f);
    const float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);

    const bool cannot_refract = ri * sin_theta > 1.0f;
    Vec3F direction{};

    if (cannot_refract ||
      math_utility::SchlickApproxReflectance(cos_theta, ri) > 
        curand_uniform(local_rand_state))
      direction = unit_direction.Reflect(hit.normal);
    else
      direction = unit_direction.Refract(hit.normal, ri);

    scattered = RayF(hit.point, direction, r_in.time());
    return true;
  }

 private:
  // Refractive index in vacuum or air, or the ratio of the material's
  // refractive index over the refractive index of the enclosing media
  float refraction_index_ = 1.f;
};