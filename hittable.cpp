//#include "hittable.h"
//
//__device__ HitResult Sphere::DetectHit(const RayF& r, 
//	const IntervalF& ray_interval) const noexcept {
//  const Vec3F oc = center_ - r.origin();
//  const auto a = r.direction().LengthSquared();
//  const auto h = r.direction().DotProduct(oc);
//  const auto c = oc.LengthSquared() - radius_ * radius_;
//  const auto discriminant = h * h - a * c;
//
//  HitResult hit_result{};
//
//  // If the ray didn't hit the sphere, assume the normal is -1.
//  if (discriminant < 0) {
//    hit_result.has_hit = false;
//    return hit_result;
//  }
//
//  const auto sqrt_d = std::sqrt(discriminant);
//
//  // Find the nearest root that lies in the acceptable range.
//  auto root = (h - sqrt_d) / a;
//  if (!ray_interval.Surrounds(root)) {
//    root = (h + sqrt_d) / a;
//    if (!ray_interval.Surrounds(root)) {
//      hit_result.has_hit = false;
//      return hit_result;
//    }
//  }
//
//  hit_result.t = root;
//  hit_result.point = r.GetPointAt(hit_result.t);
//  const Vec3F outward_normal = (hit_result.point - center_) / radius_;
//  hit_result.SetFaceNormal(r, outward_normal);
//  hit_result.has_hit = true;
//  hit_result.material = material_;
//
//  return hit_result;
//}
