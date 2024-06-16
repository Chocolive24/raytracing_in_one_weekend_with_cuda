#include "camera.h"
#include "color.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "hittable.h"
#include "hittable_list.h"
#include "ray.h"

#include <curand_kernel.h>

#include <ctime>
#include <iostream>

#define CHECK_CUDA_ERRORS(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const* const func,
                const char* const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << ":" << line << " '" << func << "' \n";

    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    std::exit(99);
  }
}
//
//#define RANDOM_VEC3 Vec3F{curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)}
//
//
////__device__ [[nodiscard]] Vec3F GetRandomVector(curandState* local_rand_state) noexcept {
////  return Vec3F{curand_uniform(local_rand_state),
////               curand_uniform(local_rand_state),
////               curand_uniform(local_rand_state)};
////}
//
//__device__ [[nodiscard]] Vec3F GetRandVecInUnitSphere(curandState* local_rand_state) noexcept {
//  Vec3F p{};
//  do {
//    // Transform random value in range [0 ; 1] to range [-1 ; 1].
//    p = 2.0f * RANDOM_VEC3 - Vec3F(1, 1, 1);
//  } while (p.LengthSquared() >= 1.0f);
//  return p;
//}
//
//__device__ [[nodiscard]] Vec3F GetRandVecOnHemisphere(
//    curandState* local_rand_state, const Vec3F& hit_normal) noexcept {
//  const Vec3F on_unit_sphere = GetRandVecInUnitSphere(local_rand_state).Normalized();
//  if (on_unit_sphere.DotProduct(hit_normal) > 0.f) // In the same hemisphere as the normal
//    return on_unit_sphere;
//
//  return -on_unit_sphere;
//}
//
//
//  __device__ [[nodiscard]] Color CalculatePixelColor(
//    const RayF& r, Hittable** world, curandState* local_rand_state) noexcept {
//  const HitResult hit_result =
//      (*world)->DetectHit(r, IntervalF(0.f, math_utility::kInfinity));
//  RayF cur_ray = r;
//  float cur_attenuation = 1.f;
//  for (int i = 0; i < 50; i++) {
//    if (hit_result.has_hit) {
//      const auto direction =
//          GetRandVecOnHemisphere(local_rand_state, hit_result.normal);
//
//      cur_attenuation *= 0.5f;
//      cur_ray = RayF{hit_result.point, direction};
//    } else {
//      const Vec3F unit_direction = cur_ray.direction().Normalized();
//      const float a = 0.5f * (unit_direction.y + 1.f);
//      const auto color =
//          (1.f - a) * Color(1.f, 1.f, 1.f) + a * Color(0.5f, 0.7f, 1.f);
//
//      return cur_attenuation * color;
//    }
//  }
//
//  return Vec3F{0.f, 0.f, 0.f};  // exceeded recursion.
//}

__global__ void RenderInit(curandState* rand_state) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= Camera::kImageWidth) || (j >= Camera::kImageHeight)) 
    return;

  const int pixel_index = j * Camera::kImageWidth + i;
  // Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void Render(Vec3F* fb, Camera** camera, Hittable** world,
                       const curandState* rand_state) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= Camera::kImageWidth) || (j >= Camera::kImageHeight)) return;

  const int pixel_index = j * Camera::kImageWidth + i;

  curandState local_rand_state = rand_state[pixel_index];
  Vec3F col(0, 0, 0);

  for (int s = 0; s < Camera::kSamplesPerPixel; s++) {
    const auto offset = Vec3F(curand_uniform(&local_rand_state) - 0.5f,
                              curand_uniform(&local_rand_state) - 0.5f, 0.f);
    const auto pixel_sample = (*camera)->pixel_00_loc +
                              ((i + offset.x) * (*camera)->pixel_delta_u) +
                              ((j + offset.y) * (*camera)->pixel_delta_v);
    const auto ray_direction = pixel_sample - (*camera)->kLookFrom;

    const RayF r((*camera)->kLookFrom, ray_direction);

    col += Camera::CalculatePixelColor(r, world, &local_rand_state);
  }

  fb[pixel_index] = col / Camera::kSamplesPerPixel;
}

__global__ void CreateWorld(Camera** d_camera, Hittable** d_list, Hittable** d_world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_list[0] = new Sphere(Vec3F(0.0f, -100.5f, -1.0f), 100.f,
                           new Lambertian(Color(0.8f, 0.8f, 0.0f)));

    d_list[1] = new Sphere(Vec3F(0.0f, 0.0f, -1.2f), 0.5f,
                           new Lambertian(Color(0.1f, 0.2f, 0.5f)));

    d_list[2] = new Sphere(Vec3F(-1.0f, 0.0f, -1.0f), 0.5f,
        new Dielectric(1.50f));

    d_list[3] = new Sphere(Vec3F(-1.0f, 0.0f, -1.0f), 0.4f, 
            new Dielectric(1.00f / 1.50f));

    d_list[4] = new Sphere(Vec3F(1.0f, 0.0f, -1.0f), 0.5f,
                           new Metal(Color(0.8f, 0.6f, 0.2f), 1.0f));

    *d_world = new HittableList(d_list, 5);
    *d_camera = new Camera();
    (*d_camera)->Initialize();
  }
}

__global__ void FreeWorld(Camera** d_camera, Hittable** d_list, Hittable** d_world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (*d_camera)
    {
      delete *d_camera;
      d_camera = nullptr;
    }
    if (*d_world) {
      delete *d_world;
      *d_world = nullptr;  // Set the pointer to nullptr after deletion
    }
    for (int i = 0; i < 5; i++) {
      // Dynamic_cast is not allowed in device code so we use static_cast and yes I
      // know that is a real code smell here but the objective is to write the code
      // in a weekend so I don't want to search for a better architecture now sorry.
      delete static_cast<Sphere*>(d_list[i])->material;
      delete d_list[i];
    }
  }
}

int main() {
  // FrameBuffer
  constexpr int kNumPixels = Camera::kImageWidth * Camera::kImageHeight;
  constexpr std::size_t kFbSize = sizeof(Vec3F) * kNumPixels;

  // Allocate FB
  Vec3F* fb = nullptr;
  CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&fb), kFbSize));

  constexpr int kTx = 8;
  constexpr int kTy = 8;

  const std::clock_t start = clock();

  // Render our buffer
  dim3 blocks(Camera::kImageWidth / kTx + 1, Camera::kImageHeight / kTy + 1);
  dim3 threads(kTx, kTy);

  // Allocate the camera on the GPU.
  Camera** d_camera = nullptr;
  CHECK_CUDA_ERRORS(
      cudaMalloc(reinterpret_cast<void**>(&d_camera), sizeof(Camera*)));
  // Allocate the world on the GPU.
  Hittable** d_list;
  CHECK_CUDA_ERRORS(cudaMalloc(reinterpret_cast<void**>(&d_list), 5 * sizeof(Hittable*)));
  Hittable** d_world;
  CHECK_CUDA_ERRORS(cudaMalloc(reinterpret_cast<void**>(&d_world), sizeof(Hittable*)));
  CreateWorld<<<1, 1>>>(d_camera, d_list, d_world);
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

  curandState* d_rand_state;
  CHECK_CUDA_ERRORS(cudaMalloc(reinterpret_cast<void**>(&d_rand_state),
                     kNumPixels * sizeof(curandState)));

  RenderInit<<<blocks, threads>>>(d_rand_state);
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

  Render<<<blocks, threads>>>(fb, d_camera, d_world, d_rand_state);
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

  const std::clock_t stop = clock();
  const double timer_seconds =
      static_cast<double>(stop - start) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image
  std::cout << "P3\n"
            << Camera::kImageWidth << " " << Camera::kImageHeight << "\n255\n";
  // The j loop is inverted compared to the raytracing in a weekend book because
  // the Render function executed on the GPU has an inverted Y.
  for (int j = 0; j < Camera::kImageHeight; j++) {
    for (int i = 0; i < Camera::kImageWidth; i++) {
      const size_t pixel_index = j * Camera::kImageWidth + i;
      write_color(std::cout, fb[pixel_index]);
    }
  }

  // clean up
  //CHECK_CUDA_ERRORS(cudaGetLastError());
  //CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  //FreeWorld<<<1, 1>>>(d_camera, d_list, d_world);
  //CHECK_CUDA_ERRORS(cudaGetLastError());
  //CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  CHECK_CUDA_ERRORS(cudaFree(d_rand_state));
  CHECK_CUDA_ERRORS(cudaFree(d_camera));
  CHECK_CUDA_ERRORS(cudaFree(d_world));
  CHECK_CUDA_ERRORS(cudaFree(d_list));
  CHECK_CUDA_ERRORS(cudaFree(fb));

  // Useful for cuda-memcheck --leak-check full
  CHECK_CUDA_ERRORS(cudaDeviceReset());

  return EXIT_SUCCESS;
}
