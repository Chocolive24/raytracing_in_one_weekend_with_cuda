#include "camera.h"
#include "color.h"
#include "hittable.h"
#include "hittable_list.h"
#include "ray.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <ctime>
#include <iostream>

#include "bvh.h"

#define CHECK_CUDA_ERRORS(val) check_cuda((val), #val, __FILE__, __LINE__)

static constexpr short kXVal = 11;
static constexpr short kYVal = 11;
static constexpr short kObjectCount = kXVal * 2 * kYVal * 2 + 1 + 3;

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

__global__ void RandInit(curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(1984, 0, 0, rand_state);
  }
}

__global__ void RenderInit(curandState* rand_state) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= Camera::kImageWidth) || (j >= Camera::kImageHeight)) 
    return;

  const int pixel_index = j * Camera::kImageWidth + i;
  // Original: Each thread gets same seed, a different sequence number, no
  // offset curand_init(1984, pixel_index, 0, &rand_state[pixel_index]); BUGFIX,
  // see Issue#2: Each thread gets different seed, same sequence for performance
  // improvement of about 2x!
  curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void Render(Vec3F* fb, Camera* camera, Hittable** world,
                        curandState* rand_state) {

  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= Camera::kImageWidth) || (j >= Camera::kImageHeight)) return;

  const int pixel_index = j * Camera::kImageWidth + i;

  curandState local_rand_state = rand_state[pixel_index];
  Vec3F col(0, 0, 0);

  for (int s = 0; s < Camera::kSamplesPerPixel; s++) {

    const RayF r((camera)->GetRayAtLocation(i, j, &local_rand_state));
    col += Camera::CalculatePixelColor(r, world, &local_rand_state);
  }

  rand_state[pixel_index] = local_rand_state;
  fb[pixel_index] = col / Camera::kSamplesPerPixel;
}

__global__ void BouncingSpheres(Camera** d_camera, Hittable** d_list, 
    Hittable** d_world, curandState* rand_state) {
  // Step 1: Create the world's objects.
  if (threadIdx.x == 0 && blockIdx.x == 0) {

    curandState local_rand_state = *rand_state;
    d_list[0] = new Sphere(Vec3F(0, -1000.0, -1), 1000,
                           new Lambertian(new SolidColor(0.5, 0.5, 0.5)));

    int i = 1;
    for (int a = -kXVal; a < kXVal; a++) {
      for (int b = -kYVal; b < kYVal; b++) {
        const float choose_mat = RANDOM;
        const Vec3F center(a + RANDOM, 0.2f, b + RANDOM);

        if (choose_mat < 0.8f) {
          auto albedo = GetRandomVector(&local_rand_state) *
                        GetRandomVector(&local_rand_state);
          const auto center2 = 
              center + Vec3F(0, RANDOM * 0.5f, 0.f);
          d_list[i++] = new Sphere(center, center2, 0.2f, 
              new Lambertian(new SolidColor(albedo)));
        }
        else if (choose_mat < 0.95f) {
          d_list[i++] = new Sphere(center, 0.2f,
              new Metal(Vec3F(0.5f * (1.0f + RANDOM), 0.5f * (1.0f + RANDOM),
                             0.5f * (1.0f + RANDOM)),0.5f * RANDOM));
        }
        else {
          d_list[i++] = new Sphere(center, 0.2f, new Dielectric(1.5f));
        }
      }
    }

    d_list[i++] = new Sphere(Vec3F(0, 1, 0), 1.0, new Dielectric(1.5f));
    d_list[i++] = new Sphere(Vec3F(-4, 1, 0), 1.0, 
        new Lambertian(new SolidColor(0.4f, 0.2f, 0.1f)));
    d_list[i++] = new Sphere(Vec3F(4, 1, 0), 1.0, new Metal(Color(0.7f, 0.6f, 0.5f), 0.0));

    *rand_state = local_rand_state;

    *d_world = new HittableList(d_list, kObjectCount);

    *d_camera = new Camera();
    (*d_camera)->Initialize();
  }
}



__global__ void BVH(Camera** d_camera, Hittable** d_list,
                                Hittable** d_world, curandState* rand_state,
                                Hittable** d_bvh_node) {
  // Step 1: Create the world's objects.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;
    d_list[0] = new Sphere(Vec3F(0, -1000.0, -1), 1000,
                           new Lambertian(new SolidColor(0.5, 0.5, 0.5)));

    int i = 1;
    for (int a = -kXVal; a < kXVal; a++) {
      for (int b = -kYVal; b < kYVal; b++) {
        const float choose_mat = RANDOM;
        const Vec3F center(a + RANDOM, 0.2f, b + RANDOM);

        if (choose_mat < 0.8f) {
          auto albedo = GetRandomVector(&local_rand_state) *
                        GetRandomVector(&local_rand_state);
          const auto center2 = center + Vec3F(0, RANDOM * 0.5f, 0.f);
          d_list[i++] = new Sphere(center, center2, 0.2f,
                                   new Lambertian(new SolidColor(albedo)));
        } else if (choose_mat < 0.95f) {
          d_list[i++] = new Sphere(
              center, 0.2f,
              new Metal(Vec3F(0.5f * (1.0f + RANDOM), 0.5f * (1.0f + RANDOM),
                              0.5f * (1.0f + RANDOM)),
                        0.5f * RANDOM));
        } else {
          d_list[i++] = new Sphere(center, 0.2f, new Dielectric(1.5f));
        }
      }
    }

    d_list[i++] = new Sphere(Vec3F(0, 1, 0), 1.0, new Dielectric(1.5f));
    d_list[i++] = new Sphere(Vec3F(-4, 1, 0), 1.0,
                             new Lambertian(new SolidColor(0.4f, 0.2f, 0.1f)));
    d_list[i++] = new Sphere(Vec3F(4, 1, 0), 1.0,
                             new Metal(Color(0.7f, 0.6f, 0.5f), 0.0));

    *rand_state = local_rand_state;

    *d_world = new HittableList(d_list, kObjectCount);

    *d_camera = new Camera();
    (*d_camera)->Initialize();
  }
}

__global__ void CheckeredSpheres(Camera** d_camera, Hittable** u_list,
                                 Hittable** d_list,
                                 Hittable** d_world, curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;

    //const auto checker =
    //    new CheckerTexture(0.32f, Color(.2f, .3f, .1f), Color(.9f, .9f, .9f));

    //d_list[0] = new Sphere(Vec3F(0, -10.f, 0), 10, new Lambertian(checker));
    //d_list[1] = new Sphere(Vec3F(0, 10, 0), 10, new Lambertian(checker));

    *rand_state = local_rand_state;

    d_list = u_list;

    *d_world = new HittableList(d_list, 1);
   /* *d_camera = new Camera();
    (*d_camera)->Initialize();*/
  }
}

__global__ void Earth(Camera** d_camera, Hittable** d_list, Hittable** d_world,
                      curandState* rand_state, unsigned char* d_image_data,
                      ImageAttributes* img_attrib, ImageTexture** tex) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;

    *tex = new ImageTexture(d_image_data, *img_attrib);

    d_list[0] = new Sphere(Vec3F(0, 0, 0), 2, new Lambertian(*tex));

    *rand_state = local_rand_state;
    *d_world = new HittableList(d_list, 1);
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
      *d_world = nullptr;
    }
    for (int i = 0; i < kObjectCount; i++) {
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

  curandState* d_rand_state;
  CHECK_CUDA_ERRORS(cudaMalloc(reinterpret_cast<void**>(&d_rand_state),
                               kNumPixels * sizeof(curandState)));

  curandState* d_rand_state2;
  CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&d_rand_state2), 1 * sizeof(curandState)));

  // we need that 2nd random state to be initialized for the world creation
  RandInit<<<1, 1>>>(d_rand_state2);
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

  // Allocate the camera on the GPU.
  Camera** d_camera = nullptr;
  CHECK_CUDA_ERRORS(
      cudaMallocManaged(reinterpret_cast<void**>(&d_camera), sizeof(Camera*)));

  // Allocate the world on the GPU.
  Hittable** d_list = nullptr;
  CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&d_list),
                               kObjectCount * sizeof(Hittable*)));

  Hittable** d_world = nullptr;
  CHECK_CUDA_ERRORS(
      cudaMallocManaged(reinterpret_cast<void**>(&d_world), sizeof(Hittable*)));

  Hittable** d_bvh_node;
  CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&d_bvh_node),
                                      sizeof(Hittable*)));

  //switch(3)
  //{
  //  case 1:
  //    BouncingSpheres<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2);
  //    break;
  //  case 2: {
  //    BVH<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2, d_bvh_node);
  //    break;
  //  }
  //  case 3: {
  //    Sphere* sphere;
  //    CheckerTexture* checker;
  //    Lambertian* material;
  //    cudaMallocManaged(&sphere,sizeof(Sphere));  // Allocate unified memory for Sphere
  //    cudaMallocManaged(&checker,sizeof(CheckerTexture));  // Allocate unified memory for Checker
  //    cudaMallocManaged(&material,sizeof(Lambertian));  // Allocate unified memory for Material

  //    new (checker) CheckerTexture(0.32f, Color(.2f, .3f, .1f), Color(.9f, .9f, .9f));
  //    new (material) Lambertian(checker);
  //    new (sphere) Sphere(Vec3F(0, -10.f, 0), 10, material);

  //    CheckeredSpheres<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2, sphere);
  //    break;
  //  }
  //  case 4: {
  //    const auto earth_image = ImageFileBuffer("../../images/earthmap.jpg");

  //    // Allocate Unified Memory so that both the CPU and GPU can access it
  //    unsigned char* d_image_data;
  //    const std::size_t image_size = earth_image.size();
  //    CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&d_image_data), image_size));

  //    // Copy the image data to Unified Memory
  //    memcpy(d_image_data, earth_image.b_data_, image_size);

  //    ImageAttributes* d_img_attrib = nullptr;
  //    CHECK_CUDA_ERRORS(cudaMallocManaged(
  //        reinterpret_cast<void**>(&d_img_attrib), sizeof(ImageAttributes)));

  //    *d_img_attrib = earth_image.attributes;

  //    // Allocate the texture on the GPU.
  //    ImageTexture** d_texture = nullptr;
  //    CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&d_texture),
  //                                 sizeof(ImageTexture*)));

  //    Earth<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2, d_image_data,
  //                    d_img_attrib, d_texture);
  //    break;
  //  }
  //  default:
  //    break;
  //}

  //CheckeredSpheres<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2);
  /*CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());*/

  // Allocate device memory
  Texture* u_checker;
  Material* u_material;
  Sphere* u_sphere1;

  CHECK_CUDA_ERRORS(cudaMallocManaged(&u_checker, sizeof(CheckerTexture)));
  CHECK_CUDA_ERRORS(cudaMallocManaged(&u_material, sizeof(Lambertian)));
  CHECK_CUDA_ERRORS(cudaMallocManaged(&u_sphere1, sizeof(Sphere)));

  // Initialize objects on the host
  new (u_checker) CheckerTexture(0.32f, Color(.2f, .3f, .1f), Color(.9f, .9f, .9f));
  new (u_material) Lambertian(u_checker);
  new (u_sphere1) Sphere(Vec3F(0, -10, 0), 10, u_material);

  // Allocate the world on the GPU.
  Hittable** u_list = nullptr;
  CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&u_list),
                                      kObjectCount * sizeof(Hittable*)));

  u_list[0] = u_sphere1;

  Hittable* u_world;
  CHECK_CUDA_ERRORS(cudaMallocManaged(&u_world, sizeof(Hittable)));

  new (u_world) HittableList(u_list, 1, true);
  std::cout << u_world->GetBoundingBox().x.max << '\n';
  *d_world = u_world;

  Camera* u_cam;
  CHECK_CUDA_ERRORS(cudaMallocManaged(&u_cam, sizeof(Camera)));

  new (u_cam) Camera();
  u_cam->Initialize();
  std::cout << u_cam->defocus_angle << '\n';
  *d_camera = u_cam;
  //(*d_camera)->Initialize();

  CheckeredSpheres<<<1, 1>>>(d_camera, u_list, d_list, d_world, d_rand_state2);
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

  constexpr int kTx = 16;
  constexpr int kTy = 16;

  const std::clock_t start = clock();

  // Render our buffer
  dim3 blocks(Camera::kImageWidth / kTx + 1, Camera::kImageHeight / kTy + 1);
  dim3 threads(kTx, kTy);

  RenderInit<<<blocks, threads>>>(d_rand_state);
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

  Render<<<blocks, threads>>>(fb, u_cam, d_world, d_rand_state);
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
