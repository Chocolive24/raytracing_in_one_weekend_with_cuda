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

#include "constant_medium.h"
#include "quad.h"

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

__global__ void Render(Vec3F* fb, Camera** camera, Hittable** world,
                        curandState* rand_state) {

  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= Camera::kImageWidth) || (j >= Camera::kImageHeight)) return;

  const int pixel_index = j * Camera::kImageWidth + i;

  curandState local_rand_state = rand_state[pixel_index];
  Vec3F col(0, 0, 0);

  for (int s = 0; s < Camera::kSamplesPerPixel; s++) {

    const RayF r((*camera)->GetRayAtLocation(i, j, &local_rand_state));
    col += (*camera)->CalculatePixelColor(r, world, &local_rand_state);
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
          auto albedo = GetRandomUnitVector(&local_rand_state) *
                        GetRandomUnitVector(&local_rand_state);
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


__global__ void CheckeredSpheres(Camera** d_camera, Hittable** d_list,
                                 Hittable** d_world, curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;

    const auto checker =
        new CheckerTexture(0.32f, Color(.2f, .3f, .1f), Color(.9f, .9f, .9f));

    d_list[0] = new Sphere(Vec3F(0, -10.f, 0), 10, new Lambertian(checker));
    d_list[1] = new Sphere(Vec3F(0, 10, 0), 10, new Lambertian(checker));

    *rand_state = local_rand_state;
    *d_world = new HittableList(d_list, 2);
    *d_camera = new Camera();
    (*d_camera)->Initialize();
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

__global__ void PerlinScene(Camera** d_camera, Hittable** d_list,
                            Hittable** d_world, curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;

    *rand_state = local_rand_state;

    auto pertext = new NoiseTexture(4, rand_state);
    d_list[0] = new Sphere(Vec3F(0, -1000, 0), 1000, new Lambertian(pertext));
    d_list[1] = new Sphere(Vec3F(0, 2, 0), 2, new Lambertian(pertext));

    *d_world = new HittableList(d_list, 2);
    *d_camera = new Camera();
    (*d_camera)->Initialize();
  }
}

__global__ void QuadScene(Camera** d_camera, Hittable** d_list,
                            Hittable** d_world, curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;

        // Materials
    const auto left_red = new Lambertian(new SolidColor(1.0f, 0.2f, 0.2f));
    const auto back_green = new Lambertian(new SolidColor(0.2f, 1.0f, 0.2f));
    const auto right_blue = new Lambertian(new SolidColor(0.2f, 0.2f, 1.0f));
    const auto upper_orange = new Lambertian(new SolidColor(1.0f, 0.5f, 0.0f));
    const auto lower_teal = new Lambertian(new SolidColor(0.2f, 0.8f, 0.8f));

    // Quads
    d_list[0] = new Quad(Vec3F(-3, -2, 5), Vec3F(0, 0, -4),
                                Vec3F(0, 4, 0), left_red);
    d_list[1] = new Quad(Vec3F(-2, -2, 0), Vec3F(4, 0, 0), Vec3F(0, 4, 0),
                                back_green);
    d_list[2] = new Quad(Vec3F(3, -2, 1), Vec3F(0, 0, 4), Vec3F(0, 4, 0),
                                right_blue);
    d_list[3] = new Quad(Vec3F(-2, 3, 1), Vec3F(4, 0, 0), Vec3F(0, 0, 4),
                                upper_orange);
    d_list[4] = new Quad(Vec3F(-2, -3, 5), Vec3F(4, 0, 0),
                                Vec3F(0, 0, -4), lower_teal);

    
    *rand_state = local_rand_state;
    *d_world = new HittableList(d_list, 5);
    *d_camera = new Camera();
    (*d_camera)->Initialize();
  }
}

__global__ void SimpleLightScene(Camera** d_camera, Hittable** d_list,
                            Hittable** d_world, curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;

    *rand_state = local_rand_state;

    const auto pertext = new NoiseTexture(4, rand_state);
    d_list[0] = new Sphere(Vec3F(0, -1000, 0), 1000, new Lambertian(pertext));
    d_list[1] = new Sphere(Vec3F(0, 2, 0), 2, new Lambertian(pertext));

    const auto difflight = new DiffuseLight(new SolidColor(4, 4, 4));
   d_list[2] = new Sphere(Vec3F(0, 7, 0), 2, difflight);
    d_list[3] = new Quad(Vec3F(3, 1, -2), Vec3F(2, 0, 0), Vec3F(0, 2, 0),
                                difflight);

    *d_world = new HittableList(d_list, 4);
    *d_camera = new Camera();
    (*d_camera)->Initialize();
  }
}

__global__ void SimpleCornellBox(Camera** d_camera, Hittable** d_list,
                            Hittable** d_world, curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;

    const auto red = new Lambertian(new SolidColor(0.65f, 0.05f, 0.05f));
    const auto white = new Lambertian(new SolidColor(0.73f, 0.73f, 0.73f));
    const auto green = new Lambertian(new SolidColor(0.12f, 0.45f, 0.15f));
    const auto light = new DiffuseLight(new SolidColor(15, 15, 15));

    d_list[0] = new Quad(Vec3F(555, 0, 0), Vec3F(0, 555, 0),
                                Vec3F(0, 0, 555), green);
    d_list[1] = new Quad(Vec3F(0, 0, 0), Vec3F(0, 555, 0),
                                Vec3F(0, 0, 555), red);
    d_list[2] = new Quad(Vec3F(343, 554, 332), Vec3F(-130, 0, 0),
                                Vec3F(0, 0, -105), light);
    d_list[3] = new Quad(Vec3F(0, 0, 0), Vec3F(555, 0, 0),
                                Vec3F(0, 0, 555), white);
    d_list[4] = new Quad(Vec3F(555, 555, 555), Vec3F(-555, 0, 0),
                                Vec3F(0, 0, -555), white);
    d_list[5] = new Quad(Vec3F(0, 0, 555), Vec3F(555, 0, 0),
                                Vec3F(0, 555, 0), white);

    Hittable* box1 = new HittableList(
        CreateBox(Vec3F(0, 0, 0), Vec3F(165, 330, 165), white), 6);
    box1 = new RotateY(box1, 15);
    box1 = new Translate(box1, Vec3F(265, 0, 295));
    d_list[6] = box1;

    Hittable* box2 = new HittableList(
        CreateBox(Vec3F(0, 0, 0), Vec3F(165, 165, 165), white), 6);
    box2 = new RotateY(box2, -18);
    box2 = new Translate(box2, Vec3F(130, 0, 65));
    d_list[7] = box2;

    *rand_state = local_rand_state;
    *d_world = new HittableList(d_list, 8);
    *d_camera = new Camera();
    (*d_camera)->Initialize();
  }
}


__global__ void CornellSmokeScene(Camera** d_camera, Hittable** d_list,
                                 Hittable** d_world, curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;
    *rand_state = local_rand_state;

    const auto red = new Lambertian(new SolidColor(0.65f, 0.05f, 0.05f));
    const auto white = new Lambertian(new SolidColor(0.73f, 0.73f, 0.73f));
    const auto green = new Lambertian(new SolidColor(0.12f, 0.45f, 0.15f));
    const auto light = new DiffuseLight(new SolidColor(7, 7, 7));

    d_list[0] =
        new Quad(Vec3F(555, 0, 0), Vec3F(0, 555, 0), Vec3F(0, 0, 555), green);
    d_list[1] =
        new Quad(Vec3F(0, 0, 0), Vec3F(0, 555, 0), Vec3F(0, 0, 555), red);
    d_list[2] = new Quad(Vec3F(113, 554, 127), Vec3F(330, 0, 0),
                         Vec3F(0, 0, 305), light);
    d_list[3] =
        new Quad(Vec3F(0, 555, 0), Vec3F(555, 0, 0), Vec3F(0, 0, 555), white);
    d_list[4] = new Quad(Vec3F(0, 0, 0), Vec3F(555, 0, 0),
                         Vec3F(0, 0, 555), white);
    d_list[5] =
        new Quad(Vec3F(0, 0, 555), Vec3F(555, 0, 0), Vec3F(0, 555, 0), white);

 /*   Sphere* sphere = new Sphere(Vec3F(430, 150, 360), 165, white);
    d_list[6] = new ConstantMedium(sphere, 0.01f, 
        new Isotropic(new SolidColor(0, 0, 0)), rand_state);*/

    Hittable* box1 = new HittableList(
        CreateBox(Vec3F(0, 0, 0), Vec3F(165, 330, 165), white), 6);
    box1 = new RotateY(box1, 15);
    box1 = new Translate(box1, Vec3F(265, 0, 295));
    d_list[6] = new ConstantMedium(box1, 0.01f, 
        new Isotropic(new SolidColor(0, 0, 0)), rand_state);

    *rand_state = local_rand_state;

    Hittable* box2 = new HittableList(
        CreateBox(Vec3F(0, 0, 0), Vec3F(165, 165, 165), white), 6);
    box2 = new RotateY(box2, -18);
    box2 = new Translate(box2, Vec3F(130, 0, 65));
    d_list[7] = new ConstantMedium(box2, 0.01f, 
        new Isotropic(new SolidColor(1, 1, 1)), rand_state);

    *d_world = new HittableList(d_list, 8);
    *d_camera = new Camera();
    (*d_camera)->Initialize();
  }
}

__global__ void FinalScene(Camera** d_camera, Hittable** d_list,
                           Hittable** d_world, curandState* rand_state,
                           unsigned char* d_image_data,
                           ImageAttributes* img_attrib, ImageTexture** tex) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;

    *rand_state = local_rand_state;

   // hittable_list boxes1;
    auto ground = new Lambertian(new SolidColor(0.48f, 0.83f, 0.53f));

    int object_idx = 0;

    int boxes_per_side = 15; //20
    for (int i = 0; i < boxes_per_side; i++) {
      for (int j = 0; j < boxes_per_side; j++) {
        const float w = 100.0f;
        const float x0 = -1000.0f + i * w;
        const float z0 = -1000.0f + j * w;
        const float y0 = 0.0f;
        const float x1 = x0 + w;
        const float y1 = GetRandomFloat(rand_state) * 100; // rnd in range ]0 ; 100]
        const float z1 = z0 + w;

        d_list[object_idx] = new HittableList(
            CreateBox(Vec3F(x0, y0, z0), Vec3F(x1, y1, z1), ground), 6);

        object_idx++;

        //boxes1.add(box(point3(x0, y0, z0), point3(x1, y1, z1), ground));
      }
    }

    //world.add(make_shared<bvh_node>(boxes1));

    auto light = new DiffuseLight(new SolidColor(7, 7, 7));
    d_list[object_idx] = new Quad(Vec3F(123, 554, 147), Vec3F(300, 0, 0),
                                Vec3F(0, 0, 265), light);
    object_idx++;

    auto center1 = Vec3F(400, 400, 200);
    auto center2 = center1 + Vec3F(30, 0, 0);
    auto sphere_material = new Lambertian(new SolidColor(0.7f, 0.3f, 0.1f));
    d_list[object_idx] = new Sphere(center1, center2, 50, sphere_material);
    object_idx++;

    d_list[object_idx] = new Sphere(Vec3F(260, 150, 45), 50,
                                  new Dielectric(1.5f));
    object_idx++;

    d_list[object_idx] = new Sphere(Vec3F(0, 150, 145), 50,
                                    new Metal(Color(0.8f, 0.8f, 0.9f), 1.0f));
    object_idx++;

    auto boundary = new Sphere(Vec3F(360, 150, 145), 70,
                                        new Dielectric(1.5f));
    d_list[object_idx] = boundary;
    object_idx++;

    d_list[object_idx] = new ConstantMedium(boundary, 0.2f, 
        new Isotropic(new SolidColor(0.2f, 0.4f, 0.9f)), rand_state);
    object_idx++;


    boundary = new Sphere(Vec3F(0, 0, 0), 5000,
                                   new Dielectric(1.5f));
    d_list[object_idx] = new ConstantMedium(boundary, .0001f, 
        new Isotropic(new SolidColor(1, 1, 1)), rand_state);
    object_idx++;

    
    *tex = new ImageTexture(d_image_data, *img_attrib);
    d_list[object_idx] = new Sphere(Vec3F(400, 200, 400), 100, new Lambertian(*tex));
    object_idx++;

    /*auto emat =
        make_shared<lambertian>(make_shared<image_texture>("earthmap.jpg"));
    world.add(make_shared<sphere>(point3(400, 200, 400), 100, emat));*/

    auto pertext = new NoiseTexture(0.2f, rand_state);
    d_list[object_idx] = new Sphere(Vec3F(220, 280, 300), 80,
                                  new Lambertian(pertext));
    object_idx++;

    //hittable_list boxes2;
    auto white = new Lambertian(new SolidColor(.73f, .73f, .73f));
    int ns = 100; // 1000
    for (int j = 0; j < ns; j++) {
      //boxes2.add(make_shared<sphere>(point3::random(0, 165), 10, white));
      Hittable* sphere =
          new Sphere(GetRandomUnitVector(rand_state) * 165, 10, white);
      sphere = new Translate(new RotateY(sphere, 15), Vec3F(-100, 270, 395));
      d_list[object_idx] = sphere;
      object_idx++;
    }

  /*  world.add(make_shared<translate>(
        make_shared<rotate_y>(make_shared<bvh_node>(boxes2), 15),
        vec3(-100, 270, 395)));*/

    *d_world = new HittableList(d_list, object_idx - 1);
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
  //cudaDeviceSetLimit(cudaLimitStackSize, 65536);

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
  CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&d_camera), sizeof(Camera*)));

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

  switch(9)
  {
    case 1:
      BouncingSpheres<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2);
      break;
    case 2: {
      CheckeredSpheres<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2);
      break;
    }
    case 3: {
      const auto earth_image = ImageFileBuffer("../../images/earthmap.jpg");

      // Allocate Unified Memory so that both the CPU and GPU can access it
      unsigned char* d_image_data;
      const std::size_t image_size = earth_image.size();
      CHECK_CUDA_ERRORS(cudaMallocManaged(
          reinterpret_cast<void**>(&d_image_data), image_size));

      // Copy the image data to Unified Memory
      memcpy(d_image_data, earth_image.b_data_, image_size);

      ImageAttributes* d_img_attrib = nullptr;
      CHECK_CUDA_ERRORS(cudaMallocManaged(
          reinterpret_cast<void**>(&d_img_attrib), sizeof(ImageAttributes)));

      *d_img_attrib = earth_image.attributes;

      // Allocate the texture on the GPU.
      ImageTexture** d_texture = nullptr;
      CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&d_texture),
                                          sizeof(ImageTexture*)));

      Earth<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2, d_image_data,
                      d_img_attrib, d_texture);
      break;
    }
    case 4: {
      PerlinScene<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2);
      break;
    }
    case 5: {
      QuadScene<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2);
      break;
    }
    case 6: {
      SimpleLightScene<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2);
      break;
    }
    case 7: {
      SimpleCornellBox<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2);
      break;
    }
    case 8: {
      CornellSmokeScene<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2);
      break;
    }
    case 9: {
      const auto earth_image = ImageFileBuffer("../../images/earthmap.jpg");

      // Allocate Unified Memory so that both the CPU and GPU can access it
      unsigned char* d_image_data;
      const std::size_t image_size = earth_image.size();
      CHECK_CUDA_ERRORS(cudaMallocManaged(
          reinterpret_cast<void**>(&d_image_data), image_size));

      // Copy the image data to Unified Memory
      memcpy(d_image_data, earth_image.b_data_, image_size);

      ImageAttributes* d_img_attrib = nullptr;
      CHECK_CUDA_ERRORS(cudaMallocManaged(
          reinterpret_cast<void**>(&d_img_attrib), sizeof(ImageAttributes)));

      *d_img_attrib = earth_image.attributes;

      // Allocate the texture on the GPU.
      ImageTexture** d_texture = nullptr;
      CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&d_texture),
                                          sizeof(ImageTexture*)));

      FinalScene<<<1, 1>>>(d_camera, d_list, d_world, d_rand_state2, d_image_data, d_img_attrib, d_texture);
      break;
    }
    default:
      break;
  }

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
