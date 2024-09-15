#pragma once

#include "color.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

struct ImageAttributes {
  int width_ = 0;
  int height_ = 0; 
  int bytes_per_pixel_ = 0;
  int bytes_per_scanline_ = 0;
};

class ImageFileBuffer {
 public:
  explicit ImageFileBuffer(const char* const path) noexcept {
    // Loads the linear (gamma=1) image data from the given file name.

    int n = bytes_per_pixel_;  // Dummy out parameter: original components per pixel
    f_data_ = stbi_loadf(path, &image_width_, &image_height_, &n, bytes_per_pixel_);
    if (f_data_ == nullptr) {
      std::cerr << "Couldn't load the file image at path: " << path << '\n';
      return;
    }

    bytes_per_scanline_ = image_width_ * bytes_per_pixel_;
    convert_to_bytes();

    attributes.width_ = image_width_;
    attributes.height_ = image_height_;
    attributes.bytes_per_pixel_ = bytes_per_pixel_;
    attributes.bytes_per_scanline_ = bytes_per_scanline_;
  }

  // Move constructor and move assignment
  ImageFileBuffer(ImageFileBuffer&& other) noexcept = default;
  ImageFileBuffer& operator=(ImageFileBuffer&& other) noexcept = default;

  // Copy constructor and copy assignment
  ImageFileBuffer(const ImageFileBuffer& other) noexcept = default;
  ImageFileBuffer& operator=(const ImageFileBuffer& other) noexcept = default;

  ~ImageFileBuffer() noexcept {
    delete[] b_data_;
    stbi_image_free(f_data_);  // Free the floating-point image data
  }

  [[nodiscard]] std::size_t size() const noexcept {
    static auto size = image_width_ * image_height_ * 
        bytes_per_pixel_ * sizeof(unsigned char);

    return size;
  }

  [[nodiscard]] const unsigned char* GetPixelData(
      int x, int y) const noexcept {
    static constexpr unsigned char magenta[] = {255, 0, 255};
    if (b_data_ == nullptr) 
      return magenta;

    x = clamp(x, 0, image_width_);
    y = clamp(y, 0, image_height_);

    return b_data_ + y * bytes_per_scanline_ + x * bytes_per_pixel_;
  }


  ImageAttributes attributes{0, 0, 3, 0};

  int bytes_per_pixel_ = 3;
  float* f_data_ = nullptr;          // Linear floating point pixel data
  unsigned char* b_data_ = nullptr;  // Linear byte pixel data
  int image_width_ = 0;              // Loaded image width
  int image_height_ = 0;             // Loaded image height
  int bytes_per_scanline_ = 0;
  unsigned char* unified_data_ = nullptr;

  static int clamp(int x, int low, int high) {
    if (x < low) return low;
    if (x < high) return x;
    return high - 1;
  }

  static unsigned char float_to_byte(float value) {
    if (value <= 0.0f) return 0;
    if (value >= 1.0f) return 255;
    return static_cast<unsigned char>(256.0f * value);
  }

  void convert_to_bytes() {
    const int total_bytes = image_width_ * image_height_ * bytes_per_pixel_;

    b_data_ = new unsigned char[total_bytes];

    // Convert the linear floating point pixel data to bytes
    auto* bptr = b_data_;
    auto* fptr = f_data_;
    for (int i = 0; i < total_bytes; i++, fptr++, bptr++)
      *bptr = float_to_byte(*fptr);
  }
};


class Texture {
public:
  __host__ __device__ constexpr Texture() noexcept = default;
  __host__ __device__ Texture(Texture&& other) noexcept = default;
  __host__ __device__ Texture& operator=(Texture&& other) noexcept = default;
  __host__ __device__ Texture(const Texture& other) noexcept = default;
  __host__ __device__ Texture& operator=(const Texture& other) noexcept = default;
  __host__ __device__ virtual ~Texture() noexcept = default;

  __host__ __device__ [[nodiscard]] virtual Color ComputeColor(
      float u, float v, const Vec3F& p) const noexcept = 0;
};

class SolidColor final : public Texture {
public:
  __host__ __device__ constexpr SolidColor(const Color& albedo) noexcept
      : albedo_(albedo) {}
  __host__ __device__ constexpr SolidColor(const float red, const float green,
                                         const float blue)
      : SolidColor(Color(red, green, blue)) {}

  __host__ __device__ [[nodiscard]] Color ComputeColor(
      float u, float v, const Vec3F& p) const noexcept override {
    return albedo_;
  }

private:
  Color albedo_{};
};

class CheckerTexture final : public Texture {
 public:
  __host__ __device__ CheckerTexture(const float scale, Texture* even,
                                    Texture* odd)
      : inv_scale_(1.f / scale), even_(even), odd_(odd) {}

  __host__ __device__ CheckerTexture(float scale, const Color& c1,
                                    const Color& c2)
      : inv_scale_(1.f / scale),
        even_(new SolidColor(c1)),
        odd_(new SolidColor(c2)) {}

  __host__ __device__ [[nodiscard]] Color ComputeColor(
      float u, float v, const Vec3F& p)
      const noexcept override {
    const auto x_integer = static_cast<int>(std::floor(inv_scale_ * p.x));
    const auto y_integer = static_cast<int>(std::floor(inv_scale_ * p.y));
    const auto z_integer = static_cast<int>(std::floor(inv_scale_ * p.z));

    const bool is_even = (x_integer + y_integer + z_integer) % 2 == 0;

    return is_even ? even_->ComputeColor(u, v, p) : odd_->ComputeColor(u, v, p);
  }

 private:
  float inv_scale_ = 0;
  Texture* even_ = nullptr;
  Texture* odd_ = nullptr;
};


class ImageTexture final : public Texture {
public:
  __host__ __device__ ImageTexture(unsigned char* image_data) {
   unified_image_data_ = image_data;
    image_width_ = 1024;
    image_height_ = 512;
    bytes_per_scanline_ = 1024 * 3;
    bytes_per_pixel_ = 3;
  }

  __host__ __device__ ImageTexture(unsigned char* unified_image_data, 
      const ImageAttributes& image_attrib) {
    unified_image_data_ = unified_image_data;
    image_width_ = image_attrib.width_;
    image_height_ = image_attrib.height_;
    bytes_per_scanline_ = image_attrib.bytes_per_scanline_;
    bytes_per_pixel_ = image_attrib.bytes_per_pixel_;
  }
  __host__ __device__ ImageTexture(ImageTexture&& other) noexcept = default;
  __host__ __device__ ImageTexture& operator=(ImageTexture&& other) noexcept = delete;
  __host__ __device__ ImageTexture(const ImageTexture& other) noexcept = default;
  __host__ __device__ ImageTexture& operator=(const ImageTexture& other) noexcept = delete;
  __host__ __device__ ~ImageTexture() noexcept override = default;

  __host__ __device__ static int clamp(int x, int low, int high) {
    if (x < low) return low;
    if (x < high) return x;
    return high - 1;
  }

  __host__ __device__ [[nodiscard]] Color ComputeColor(
      float u, float v, const Vec3F& p) const noexcept override {
    // If we have no texture data, then return solid cyan as a debugging aid.
    if (image_height_ <= 0) 
        return Color{0, 1, 1};

    // Clamp input texture coordinates to [0,1] x [1,0]
    u = IntervalF{0, 1}.Clamp(u);
    v = 1.f - IntervalF{0, 1}.Clamp(v);  // Flip V to image coordinates

    const auto i = static_cast<int>(u * image_width_);
    const auto j = static_cast<int>(v * image_height_);
    //const auto pixel = image_.GetPixelData(i, j);

    const unsigned char* pixel = nullptr;

    static constexpr unsigned char magenta[] = {255, 0, 255};
    if (unified_image_data_ == nullptr)
    {
      pixel = magenta;
    }
    else
    {
      auto x = i;
      auto y = j;

      x = clamp(x, 0, image_width_);
      y = clamp(y, 0, image_height_);

      pixel = unified_image_data_ + y * bytes_per_scanline_ + x * bytes_per_pixel_;
    }

    constexpr float color_scale = 1.f / 255.f;
    return Color{color_scale * pixel[0], color_scale * pixel[1],
                 color_scale * pixel[2]};
  }

private:
  //ImageFileBuffer image_;
  unsigned char* unified_image_data_ = nullptr;
  int image_width_ = 0;
  int image_height_ = 0;
  int bytes_per_scanline_ = 0;
  int bytes_per_pixel_ = 3;
};