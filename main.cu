
#include <curand_kernel.h>

#include <ctime>
#include <iostream>

#include "camera.h"
#include "color.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "hittable.h"
#include "hittable_list.h"
#include "ray.h"

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

__global__ void RenderInit(curandState* rand_state) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= Camera::kImageWidth) || (j >= Camera::kImageHeight)) return;

  const int pixel_index = j * Camera::kImageWidth + i;
  // Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void Render(Vec3F* fb, Hittable** world,
                       const curandState* rand_state) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= Camera::kImageWidth) || (j >= Camera::kImageHeight)) return;

  const int pixel_index = j * Camera::kImageWidth + i;

  Camera camera{};
  camera.Initialize();

  curandState local_rand_state = rand_state[pixel_index];
  Vec3F col(0, 0, 0);

  for (int s = 0; s < Camera::kSamplesPerPixel; s++) {
    const auto offset = Vec3F(curand_uniform(&local_rand_state) - 0.5f,
                              curand_uniform(&local_rand_state) - 0.5f, 0.f);
    const auto pixel_sample = camera.pixel_00_loc +
                              ((i + offset.x) * camera.pixel_delta_u) +
                              ((j + offset.y) * camera.pixel_delta_v);
    const auto ray_direction = pixel_sample - camera.kLookFrom_;

    const RayF r(camera.kLookFrom_, ray_direction);

    col += Camera::GetRayColor(r, world);
  }

  fb[pixel_index] = col / static_cast<float>(Camera::kSamplesPerPixel);
}

__global__ void create_world(Hittable** d_list, Hittable** d_world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list) = new Sphere(Vec3F{0, 0, -1}, 0.5f);
    *(d_list + 1) = new Sphere(Vec3F(0, -100.5, -1), 100);
    *d_world = new HittableList(d_list, 2);
  }
}

__global__ void free_world(Hittable** d_list, Hittable** d_world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (*d_list) {
      delete *d_list;
      *d_list = nullptr;  // Set the pointer to nullptr after deletion
    }
    if (*(d_list + 1)) {
      delete *(d_list + 1);
      *(d_list + 1) = nullptr;  // Set the pointer to nullptr after deletion
    }
    if (*d_world) {
      delete *d_world;
      *d_world = nullptr;  // Set the pointer to nullptr after deletion
    }
  }
}

int main() {
  // Camera camera{};
  // camera.Initialize();

  // FrameBuffer
  constexpr int num_pixels = Camera::kImageWidth * Camera::kImageHeight;
  constexpr std::size_t fb_size = sizeof(Vec3F) * num_pixels;

  // Allocate FB
  Vec3F* fb = nullptr;
  CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&fb), fb_size));

  constexpr int tx = 8;
  constexpr int ty = 8;

  const std::clock_t start = clock();

  // Render our buffer
  dim3 blocks(Camera::kImageWidth / tx + 1, Camera::kImageHeight / ty + 1);
  dim3 threads(tx, ty);

  // Allocate the world on the GPU.
  Hittable** d_list;
  CHECK_CUDA_ERRORS(
      cudaMalloc(reinterpret_cast<void**>(&d_list), 2 * sizeof(Hittable*)));
  Hittable** d_world;
  CHECK_CUDA_ERRORS(
      cudaMalloc(reinterpret_cast<void**>(&d_world), sizeof(Hittable*)));
  create_world<<<1, 1>>>(d_list, d_world);
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

  curandState* d_rand_state;
  CHECK_CUDA_ERRORS(cudaMalloc(reinterpret_cast<void**>(&d_rand_state),
                               num_pixels * sizeof(curandState)));

  RenderInit<<<blocks, threads>>>(d_rand_state);
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

  Render<<<blocks, threads>>>(fb, d_world, d_rand_state);
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
  // the Render function executed on the GPU has an inverted Y loop.
  for (int j = 0; j < Camera::kImageHeight; j++) {
    for (int i = 0; i < Camera::kImageWidth; i++) {
      const size_t pixel_index = j * Camera::kImageWidth + i;
      write_color(std::cout, fb[pixel_index]);
    }
  }

  // clean up
  // CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  // free_world<<<1, 1>>>(d_list, d_world);
  // CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaFree(d_world));
  CHECK_CUDA_ERRORS(cudaFree(d_list));
  CHECK_CUDA_ERRORS(cudaFree(d_rand_state));
  CHECK_CUDA_ERRORS(cudaFree(fb));

  // Useful for cuda-memcheck --leak-check full
  CHECK_CUDA_ERRORS(cudaDeviceReset());

  return EXIT_SUCCESS;
}
