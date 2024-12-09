#pragma once

#include "device_random.h"
#include "hittable.h"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

class BVH_Node;

// Define a structure to hold information for processing nodes
struct BVHStackEntry {
  Hittable** objects;
  std::size_t start;
  std::size_t end;
  curandState* rand_state;
  BVH_Node* node;  // Pointer to the node to be created
};

class BVHStack {
 public:
  // Define the maximum stack size
  static const int kMaxStackSize = 1000;

  // Global stack for BVH node creation
  BVHStackEntry d_stack_[kMaxStackSize]{};
  int stack_top_ = -1;

 public:
  // Push a new entry onto the stack
  __device__ void Push(Hittable** objects, std::size_t start, std::size_t end,
                       curandState* rand_state, BVH_Node* node) {
    if (stack_top_ < kMaxStackSize - 1) {
      stack_top_++;
      d_stack_[stack_top_] = BVHStackEntry{objects, start, end, rand_state, node};
    }
    else
    {
      printf("stack top > kMaxStackSize\n");
    }
  }

  // Pop an entry from the stack
  __device__ void Pop(Hittable**& objects, std::size_t& start, std::size_t& end,
                      curandState*& rand_state, BVH_Node*& node) {
    if (stack_top_ >= 0) {
      auto entry = d_stack_[stack_top_];
      objects = entry.objects;
      start = entry.start;
      end = entry.end;
      rand_state = entry.rand_state;
      node = entry.node;
      stack_top_--;
    } else
    {
      printf("stack_top < 0 \n");
    }
  }

  // Check if the stack is empty
  __device__ [[nodiscard]] bool IsEmpty() const noexcept { return stack_top_ < 0; }
};

class BVH_Node final : public Hittable {
 public:
  static const int MAX_DEPTH = 8;

  __device__ BVH_Node() noexcept = default;

  __device__ BVH_Node(Hittable** objects, const std::size_t start, const std::size_t end,
                      curandState* local_rand_state, int depth = 0) {

    // Build the bounding box of the span of source objects.
    aabb_ = AABB::empty();
    for (size_t object_index = start; object_index < end; object_index++) {
      aabb_ = AABB(aabb_, objects[object_index]->GetBoundingBox());
    }

;    //printf("aabb min max %f %f \n", aabb_.x.min, aabb_.x.max);

    // If we reached max depth or only one object left, create a leaf node
    if (depth >= MAX_DEPTH) {
      is_leaf_ = true;
      objects_ = objects;
      object_start_idx_ = start;
      object_end_idx_ = end;

      return;
    }
    else
    {
      const int axis = aabb_.LongestAxis();

      // const int axis = GetRandomInt(0, 2, rand_state);
      const auto comparator = (axis == 0)   ? BoxCompareX
                              : (axis == 1) ? BoxCompareY
                                            : BoxCompareZ;

      const size_t object_span = end - start;

      if (object_span == 1) {
        left_ = right_ = objects[start];
      }
      else if (object_span == 2) {
        left_ = objects[start];
        right_ = objects[start + 1];
      }
      else {
        // Sort the objects based on the selected axis
        thrust::device_ptr<Hittable*> d_obj_list(objects);
        thrust::sort(d_obj_list + start, d_obj_list + end, comparator);

        const auto mid = start + object_span / 2;

        // Recursively build left and right child nodes
        left_ = new BVH_Node(d_obj_list.get(), start, mid, local_rand_state,
                             depth + 1);

        right_ = new BVH_Node(d_obj_list.get(), mid, end, local_rand_state,
                              depth + 1);
        
        //// Dynamically allocate the BVHStack on the GPU heap
        // BVHStack* stack = new BVHStack();

        //// Push the initial parameters onto the stack with a temporary node
        //// pointer
        // BVH_Node* root_node =
        //     new BVH_Node();  // This will eventually point to the new node
        //     created

        // stack->Push(objects, start, end, local_rand_state, root_node);

        //// Iteratively process the stack
        // while (!stack->IsEmpty()) {
        //   Hittable** obj_list;
        //   std::size_t s, e;
        //   curandState* rand_state;
        //   BVH_Node* current_node;

        //  stack->Pop(obj_list, s, e, rand_state, current_node);

        //  if (current_node == nullptr) {
        //    printf("NULL\n");
        //  }

        //  // Build the bounding box of the span of source objects.
        //  current_node->aabb_ = AABB::empty();
        //  for (size_t object_index = start; object_index < end;
        //  object_index++) {
        //    current_node->aabb_ = AABB(current_node->aabb_,
        //                               obj_list[object_index]->GetBoundingBox());
        //  }

        //  const int axis = current_node->aabb_.LongestAxis();

        //  // const int axis = GetRandomInt(0, 2, rand_state);
        //  const auto comparator = (axis == 0)   ? BoxCompareX
        //                          : (axis == 1) ? BoxCompareY
        //                                        : BoxCompareZ;

        //  const size_t object_span = e - s;

        //  if (object_span == 1) {
        //    *current_node = new BVH_Node(obj_list[s]);
        //    continue;  // No need to split further
        //  } else if (object_span == 2) {
        //    *current_node =
        //        new BVH_Node(obj_list[s], obj_list[s + 1]);  // Assign two
        //        objects
        //    continue;  // No need to split further
        //  }

        //  // Sort the objects based on the selected axis
        //  thrust::device_ptr<Hittable*> d_obj_list(obj_list);
        //  thrust::sort(d_obj_list + s, d_obj_list + e, comparator);

        //  const auto mid = s + object_span / 2;

        //  current_node->left_ = new BVH_Node();
        //  current_node->right_ = new BVH_Node();

        //  stack->Push(obj_list, s, mid, rand_state,
        //              reinterpret_cast<BVH_Node*>(current_node->left_));
        //  stack->Push(obj_list, mid, e, rand_state,
        //              reinterpret_cast<BVH_Node*>(current_node->right_));
        //}

        // left_ = root_node->left_;
        // right_ = root_node->right_;
        // aabb_ = root_node->aabb_;

        //// Free dynamically allocated memory
        // delete stack;
      }
    }
  }

  // Additional constructor for leaf nodes
  __device__ BVH_Node(Hittable* object) {
    left_ = right_ = object;
  }

  // Constructor to link two nodes
  __device__ BVH_Node(Hittable* left, Hittable* right, bool compute_aabb = true) {
    left_ = left;
    right_ = right;
  }

   __device__ [[nodiscard]] HitResult DetectHit(
      const RayF& r, const IntervalF& ray_interval) const noexcept override {

       HitResult re{};

      if (!aabb_.Hit(r, ray_interval))
      {
        re.has_hit = false;
        return re;
      }

      // If the node is a leaf, check for intersection with all objects
      if (is_leaf_) {
        for (auto i = object_start_idx_; i < object_end_idx_; i++) {
          //printf("i: %i\n", i);
          const HitResult object_hit_result = objects_[i]->DetectHit(r, ray_interval);
          if (object_hit_result.has_hit &&
              (!re.has_hit || object_hit_result.t < re.t)) {
            re = object_hit_result;
          }
        }
          //printf("%i \n", hit_result.has_hit);
          return re;
      }

      bool left_has_hit = false;
      bool right_has_hit = false;
      HitResult tmp_res{};

      tmp_res = left_->DetectHit(r, ray_interval);
      if (tmp_res.has_hit) {
        re = tmp_res;  // Preserve the hit result from the left
        left_has_hit = tmp_res.has_hit;
      }

      tmp_res = right_->DetectHit(
          r, IntervalF(ray_interval.min,
                       re.has_hit ? re.t : ray_interval.max));
      if (tmp_res.has_hit) {
        re = tmp_res;  // Update hit result with right node hit
                                        // if applicable
        right_has_hit = tmp_res.has_hit;
      }

      // Finalize hit detection state
      re.has_hit = left_has_hit || right_has_hit;

      return re;

    //HitResult hit_result{};

    //// Check bounding box intersection
    //if (!aabb_.Hit(r, ray_interval)) {
    //  //printf("NO AABB HIT\n");
    //  hit_result.has_hit = false;
    //  return hit_result;
    //}

    ////printf("No leaf\n");

    //// If the node is a leaf, check for intersection with all objects
    //if (is_leaf_) {
    //  for (std::size_t i = object_start_idx_; i < object_end_idx_; i++) {
    //    //printf("i: %i\n", i);
    //    const HitResult object_hit_result = objects_[i]->DetectHit(r, ray_interval);
    //    if (object_hit_result.has_hit &&
    //        (!hit_result.has_hit || object_hit_result.t < hit_result.t)) {
    //      hit_result = object_hit_result;
    //    }
    //  }

    //  //printf("%i \n", hit_result.has_hit);
    //  return hit_result;
    //}

    ////printf("BEFORE LEFT\n");
    //HitResult left_hit_result = left_->DetectHit(r, ray_interval);
    //  if (left_hit_result.has_hit) {
    //    hit_result = left_hit_result;  // Preserve the hit result from the left
    //  //printf("left hit %i\n", left_hit_result.has_hit);
    //  }

    ////printf("BEFORE RIGHT\n");
    // // HitResult right_hit_result = right_->DetectHit(r, ray_interval);
    //HitResult right_hit_result = right_->DetectHit(
    //      r, IntervalF(ray_interval.min,
    //                   hit_result.has_hit ? hit_result.t : ray_interval.max));
    //  if (right_hit_result.has_hit) {
    //    hit_result = right_hit_result;  // Update hit result with right node hit
    //                                    // if applicable
    //  //printf("right hit %i\n", right_hit_result.has_hit);
    //  }

    //// Finalize hit detection state
    //hit_result.has_hit = left_hit_result.has_hit || right_hit_result.has_hit;

    //return hit_result;

    //HitResult hit_result{};
    //hit_result.has_hit = false;

    //static const int max_size = 12;

    //// Initialize stack with the root node
    //int stack_top = 0;  // Start with an empty stack

    //BVH_Node* stack[max_size];
    //stack[stack_top++] = const_cast<BVH_Node*>(this);

    //while (stack_top > 0) {
    //  // Pop a node from the stack

    //  if (stack_top <= 0)
    //  {
    //    printf("IDK WHY\n");
    //    break;
    //  }

    //  const auto current_node = stack[--stack_top];

    //  if (current_node == nullptr)
    //  {
    //    printf("Null node\n");
    //  }

    //  // Check bounding box intersection
    //  if (!current_node->aabb_.Hit(r, ray_interval)) {
    //    continue;  // Skip nodes that don't intersect the ray
    //  }

    //  // Check if the current node is a leaf
    //  if (current_node->left_ == nullptr && current_node->right_ == nullptr) {
    //    // Leaf node: test for hits against the actual object
    //    HitResult object_hit_result = current_node->DetectHit(r, ray_interval);

    //    if (object_hit_result.has_hit &&
    //        (!hit_result.has_hit || object_hit_result.t < hit_result.t)) {
    //      hit_result = object_hit_result;
    //    }
    //    continue;
    //  }

    //  // If not a leaf node, push children onto the stack
    //  if (current_node->left_ != nullptr) {
    //    stack[stack_top++] = static_cast<BVH_Node*>(current_node->left_);
    //  }

    //  // Safety check: ensure the stack doesn't overflow
    //  if (stack_top >= max_size) {
    //    printf("Error: BVH stack overflow 1 \n");
    //    break;
    //  }

    //  if (current_node->right_ != nullptr) {
    //    stack[stack_top++] = static_cast<BVH_Node*>(current_node->right_);
    //  }
   
    //  // Safety check: ensure the stack doesn't overflow
    //  if (stack_top >= max_size) {
    //    printf("Error: BVH stack overflow 2\n");
    //    break;
    //  }
    //}

    //return hit_result;
  }

  __host__ __device__ [[nodiscard]] AABB GetBoundingBox()
      const noexcept override {
    return aabb_;
  }

private:
  AABB aabb_{};
  Hittable* left_ = nullptr;
  Hittable* right_ = nullptr;
  Hittable** objects_; // List of objects if it's a leaf node.
  std::uint16_t object_start_idx_ = 0;
  std::uint16_t object_end_idx_ = 0;
  bool is_leaf_ = false;

  __host__ __device__ static bool BoxCompare(const Hittable* a,
                          const Hittable* b, int axis_index) {
    const auto a_axis_interval = a->GetBoundingBox().AxisInterval(axis_index);
    const auto b_axis_interval = b->GetBoundingBox().AxisInterval(axis_index);
    return a_axis_interval.min < b_axis_interval.min;
  }

  __host__ __device__ static bool BoxCompareX(const Hittable* a,
                            const Hittable* b) {
    return BoxCompare(a, b, 0);
  }

  __host__ __device__ static bool BoxCompareY(const Hittable* a,
                            const Hittable* b) {
    return BoxCompare(a, b, 1);
  }

  __host__ __device__ static bool BoxCompareZ(const Hittable* a,
                            const Hittable* b) {
    return BoxCompare(a, b, 2);
  }


};