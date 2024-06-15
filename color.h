#pragma once

#include "vec3.h"
#include "interval.h"

#include <iostream>

using Color = Vec3F;

inline void write_color(std::ostream& out, const Color& pixel_color) {
  const auto r = pixel_color.x;
  const auto g = pixel_color.y;
  const auto b = pixel_color.z;

  // Translate the [0,1] component values to the byte range [0,255].
  static constexpr IntervalF intensity(0.000f, 0.999f);
  const int r_byte = static_cast<int>(256 * intensity.Clamp(r));
  const int g_byte = static_cast<int>(256 * intensity.Clamp(g));
  const int b_byte = static_cast<int>(256 * intensity.Clamp(b));

  // Write out the pixel color components.
  out << r_byte << ' ' << g_byte << ' ' << b_byte << '\n';
}