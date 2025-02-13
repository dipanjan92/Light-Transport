import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, List
from jax import lax

# Import your existing BVHNode and AABB definitions.
# (Ensure that your AABB union functions expect a valid AABB rather than None.)
from accelerators.bvh import BVHNode  # This is the same BVHNode used in your original pipeline.
from primitives.aabb import AABB, get_surface_area, union, union_p

# ------------------------------------------------------------------
# Data structure for the linear BVH.
# ------------------------------------------------------------------
@dataclass
class LinearBVHNode:
    bounds: AABB = None
    primitives_offset: int = -1
    second_child_offset: int = -1
    n_primitives: int = 0
    axis: int = -1

# ------------------------------------------------------------------
# Utility: Create an empty AABB for accumulation.
# ------------------------------------------------------------------
def empty_aabb() -> AABB:
    INF = 1e10
    return AABB(jnp.array([INF, INF, INF]),
                jnp.array([-INF, -INF, -INF]),
                jnp.array([0.0, 0.0, 0.0]))

# ------------------------------------------------------------------
# Morton Code Utilities (vectorized)
# ------------------------------------------------------------------
def left_shift_3(x: jnp.ndarray) -> jnp.ndarray:
    # Clamp to 10-bit value.
    x = jnp.minimum(x, (1 << 10) - 1)
    x = jnp.bitwise_and(jnp.bitwise_or(x, jnp.left_shift(x, 16)),
                        0b00000011000000000000000011111111)
    x = jnp.bitwise_and(jnp.bitwise_or(x, jnp.left_shift(x, 8)),
                        0b00000011000000001111000000001111)
    x = jnp.bitwise_and(jnp.bitwise_or(x, jnp.left_shift(x, 4)),
                        0b00000011000011000011000011000011)
    x = jnp.bitwise_and(jnp.bitwise_or(x, jnp.left_shift(x, 2)),
                        0b00001001001001001001001001001001)
    return x

def encode_morton_3(v: jnp.ndarray) -> jnp.ndarray:
    x = left_shift_3(v[0])
    y = left_shift_3(v[1])
    z = left_shift_3(v[2])
    return (z << 2) | (y << 1) | x

def compute_morton_codes(centroids: jnp.ndarray) -> jnp.ndarray:
    overall_min = jnp.min(centroids, axis=0)
    overall_max = jnp.max(centroids, axis=0)
    extent = overall_max - overall_min
    extent = jnp.where(extent == 0, 1.0, extent)
    normalized = (centroids - overall_min) / extent
    morton_scale = 1 << 10
    scaled = jnp.floor(normalized * (morton_scale - 1)).astype(jnp.int32)
    codes = jax.vmap(encode_morton_3)(scaled)
    return codes

# ------------------------------------------------------------------
# Build Morton Primitives Arrays.
# Returns two arrays:
#   prim_nums: (N,) array of original primitive indices.
#   codes:    (N,) array of computed Morton codes.
# ------------------------------------------------------------------
def build_morton_primitives(bounded_boxes: List) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Each item in bounded_boxes is assumed to be a BVHPrimitive-like object.
    N = len(bounded_boxes)
    centroids = jnp.stack([bb.bounds.centroid for bb in bounded_boxes], axis=0)
    prim_nums = jnp.array([bb.prim_num for bb in bounded_boxes], dtype=jnp.int32)
    codes = compute_morton_codes(centroids)
    return prim_nums, codes

# ------------------------------------------------------------------
# Partitioning Helpers (functional versions)
# ------------------------------------------------------------------
def partition_pred(centroid: jnp.ndarray, centroid_bounds: AABB, n_buckets: int,
                   min_cost_split_bucket: int, dim: int) -> bool:
    mid = (centroid_bounds.min_point[dim] + centroid_bounds.max_point[dim]) * 0.5
    extent = centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim]
    extent = extent if extent != 0 else 1.0
    b = (centroid - centroid_bounds.min_point[dim]) / extent
    b = int(n_buckets * b)
    if b == n_buckets:
        b = n_buckets - 1
    return b <= min_cost_split_bucket

def interval_pred(morton_codes: jnp.ndarray, mask: int, i: int) -> bool:
    return (int(morton_codes[0]) & mask) == (int(morton_codes[i]) & mask)

# ------------------------------------------------------------------
# Upper-level BVH Construction using SAH (Functional Python loops)
# ------------------------------------------------------------------
def build_upper_sah(treelet_roots: List[BVHNode], start: int, end: int,
                    total_nodes: List[int]) -> BVHNode:
    assert start < end, "start should be less than end"
    n_nodes = end - start
    if n_nodes == 1:
        return treelet_roots[start]
    total_nodes[0] += 1
    bounds = empty_aabb()
    for i in range(start, end):
        bounds = union(bounds, treelet_roots[i].bounds)
    centroid_bounds = empty_aabb()
    for i in range(start, end):
        centroid = (treelet_roots[i].bounds.min_point + treelet_roots[i].bounds.max_point) * 0.5
        centroid_bounds = union_p(centroid_bounds, centroid)
    extent = bounds.max_point - bounds.min_point
    dim = int(jnp.argmax(extent))
    n_buckets = 12
    buckets_count = [0] * n_buckets
    buckets_bounds = [empty_aabb() for _ in range(n_buckets)]
    for i in range(start, end):
        centroid = (treelet_roots[i].bounds.min_point[dim] + treelet_roots[i].bounds.max_point[dim]) * 0.5
        denom = float(centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim])
        if denom == 0:
            bucket = 0
        else:
            bucket = int(n_buckets * ((float(centroid) - float(centroid_bounds.min_point[dim])) / denom))
        if bucket == n_buckets:
            bucket = n_buckets - 1
        buckets_count[bucket] += 1
        buckets_bounds[bucket] = union(buckets_bounds[bucket], treelet_roots[i].bounds)
    costs = []
    for i in range(n_buckets - 1):
        b0 = empty_aabb()
        count0 = 0
        for j in range(i + 1):
            b0 = union(b0, buckets_bounds[j])
            count0 += buckets_count[j]
        b1 = empty_aabb()
        count1 = 0
        for j in range(i + 1, n_buckets):
            b1 = union(b1, buckets_bounds[j])
            count1 += buckets_count[j]
        cost = 0.125 + (count0 * get_surface_area(b0) + count1 * get_surface_area(b1)) / get_surface_area(bounds)
        costs.append(cost)
    min_cost = costs[0]
    min_cost_split_bucket = 0
    for i in range(1, n_buckets - 1):
        if costs[i] < min_cost:
            min_cost = costs[i]
            min_cost_split_bucket = i
    left_list = []
    right_list = []
    for i in range(start, end):
        centroid = (treelet_roots[i].bounds.min_point[dim] + treelet_roots[i].bounds.max_point[dim]) * 0.5
        denom = float(centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim])
        if denom == 0:
            bucket = 0
        else:
            bucket = int(n_buckets * ((float(centroid) - float(centroid_bounds.min_point[dim])) / denom))
        if bucket == n_buckets:
            bucket = n_buckets - 1
        if bucket <= min_cost_split_bucket:
            left_list.append(treelet_roots[i])
        else:
            right_list.append(treelet_roots[i])
    if len(left_list) == 0 or len(right_list) == 0:
        mid = (start + end) // 2
        left_list = treelet_roots[start:mid]
        right_list = treelet_roots[mid:end]
    left_node = build_upper_sah(left_list, 0, len(left_list), total_nodes)
    right_node = build_upper_sah(right_list, 0, len(right_list), total_nodes)
    return BVHNode(bounds=bounds, child_0=left_node, child_1=right_node, split_axis=dim)

# ------------------------------------------------------------------
# emit_lbvh: Recursively build LBVH treelets.
#
# Now we pass two sorted arrays:
#    sorted_codes  : the sorted Morton codes (for bit comparisons)
#    sorted_indices: the corresponding primitive indices (to index into primitives and bounded_boxes)
# ------------------------------------------------------------------
def emit_lbvh(build_nodes: List[BVHNode],
              primitives: List[dict],
              bounded_boxes: List,
              sorted_codes: jnp.ndarray,
              sorted_indices: jnp.ndarray,
              n_primitives: int,
              total_nodes: List[int],
              ordered_prims: List[dict],
              ordered_prims_offset: List[int],
              bit_index: int) -> BVHNode:
    assert n_primitives > 0
    n_boxes = len(bounded_boxes)
    max_prims_in_node = max(4, int(0.1 * n_boxes))
    if bit_index == -1 or n_primitives <= max_prims_in_node:
        total_nodes[0] += 1
        node = build_nodes.pop(0)
        bnds = empty_aabb()
        first_prim_offset = ordered_prims_offset[0]
        ordered_prims_offset[0] += n_primitives
        for i in range(n_primitives):
            primitive_index = int(sorted_indices[i])
            ordered_prims.append(primitives[primitive_index])
            bnds = union(bnds, bounded_boxes[primitive_index].bounds)
        node.init_leaf(first_prim_offset, n_primitives, bnds)
        return node
    else:
        mask = 1 << bit_index
        first_code = int(sorted_codes[0]) & mask
        last_code = int(sorted_codes[n_primitives - 1]) & mask
        if first_code == last_code:
            return emit_lbvh(build_nodes, primitives, bounded_boxes,
                             sorted_codes,
                             sorted_indices,
                             n_primitives, total_nodes, ordered_prims, ordered_prims_offset, bit_index - 1)
        split_offset = 0
        for i in range(n_primitives):
            if (int(sorted_codes[0]) & mask) != (int(sorted_codes[i]) & mask):
                split_offset = i
                break
        if split_offset == 0 or split_offset == n_primitives:
            split_offset = n_primitives // 2
        total_nodes[0] += 1
        node = build_nodes.pop(0)
        left_child = emit_lbvh(build_nodes, primitives, bounded_boxes,
                               sorted_codes[:split_offset],
                               sorted_indices[:split_offset],
                               split_offset, total_nodes, ordered_prims, ordered_prims_offset, bit_index - 1)
        right_child = emit_lbvh(build_nodes, primitives, bounded_boxes,
                                sorted_codes[split_offset:],
                                sorted_indices[split_offset:],
                                n_primitives - split_offset, total_nodes, ordered_prims, ordered_prims_offset,
                                bit_index - 1)
        axis = bit_index % 3
        return BVHNode(bounds=union(left_child.bounds, right_child.bounds),
                       child_0=left_child, child_1=right_child, split_axis=axis)

# ------------------------------------------------------------------
# Top-level HLBVH Build
# ------------------------------------------------------------------
def build_hlbvh(primitives: List[dict],
                bounded_boxes: List,
                ordered_prims: List[dict],
                total_nodes: List[int]) -> BVHNode:
    overall_centroid_bounds = empty_aabb()
    for bb in bounded_boxes:
        overall_centroid_bounds = union_p(overall_centroid_bounds, bb.bounds.centroid)
    prim_nums, morton_codes = build_morton_primitives(bounded_boxes)
    sorted_order = jnp.argsort(morton_codes)
    sorted_codes = morton_codes[sorted_order]
    sorted_indices = prim_nums[sorted_order]
    treelets_to_build = []
    N = len(sorted_codes)
    start = 0
    end = 1
    mask = 0b00111111111111000000000000000000
    while end <= N:
        if end == N or ((int(sorted_codes[start]) & mask) != (int(sorted_codes[end]) & mask)):
            n_prims = end - start
            max_bvh_nodes = 2 * n_prims - 1
            build_nodes = [BVHNode(bounds=empty_aabb()) for _ in range(max_bvh_nodes)]
            treelets_to_build.append((start, n_prims, build_nodes))
            start = end
        end += 1
    finished_treelets = []
    ordered_prims_offset = [0]
    for (start_ix, n_prims, build_nodes) in treelets_to_build:
        first_bit_index = 29 - 12
        treelet = emit_lbvh(build_nodes, primitives, bounded_boxes,
                            sorted_codes[start_ix:start_ix+n_prims],
                            sorted_indices[start_ix:start_ix+n_prims],
                            n_prims, total_nodes, ordered_prims, ordered_prims_offset, first_bit_index)
        finished_treelets.append(treelet)
    return build_upper_sah(finished_treelets, 0, len(finished_treelets), total_nodes)

# ------------------------------------------------------------------
# Flattening the BVH tree.
#
# This routine traverses the HLBVH tree (whose child pointers are BVHNode objects,
# not integer indices) and produces a flat list of LinearBVHNode records.
# ------------------------------------------------------------------
def flatten_bvh_tree(node: BVHNode, linear_bvh: List[LinearBVHNode]) -> int:
    current_idx = len(linear_bvh)
    linear_node = LinearBVHNode()
    linear_node.bounds = node.bounds
    linear_node.primitives_offset = node.first_prim_offset
    linear_node.n_primitives = node.n_primitives
    linear_node.axis = node.split_axis
    # We will fill in second_child_offset if this node is interior.
    linear_node.second_child_offset = -1
    linear_bvh.append(linear_node)
    # If this is an interior node, recursively flatten its children.
    if node.n_primitives == 0:
        left_idx = flatten_bvh_tree(node.child_0, linear_bvh) if isinstance(node.child_0, BVHNode) else -1
        right_idx = flatten_bvh_tree(node.child_1, linear_bvh) if isinstance(node.child_1, BVHNode) else -1
        # In the linear representation, we assume the first child is stored at current_idx+1.
        # Set second_child_offset to the index of the right child.
        linear_bvh[current_idx].second_child_offset = right_idx
    return current_idx

def flatten_bvh(root: BVHNode, dummy: int = 0) -> List[LinearBVHNode]:
    linear_bvh: List[LinearBVHNode] = []
    flatten_bvh_tree(root, linear_bvh)
    return linear_bvh

# ------------------------------------------------------------------
# Packed BVH and Primitive Helpers
# ------------------------------------------------------------------
def pack_linear_bvh(linear_bvh: List[LinearBVHNode]) -> dict:
    n = len(linear_bvh)
    bounds_min = jnp.stack([node.bounds.min_point for node in linear_bvh], axis=0)
    bounds_max = jnp.stack([node.bounds.max_point for node in linear_bvh], axis=0)
    bounds_centroid = jnp.stack([node.bounds.centroid for node in linear_bvh], axis=0)
    primitives_offset = jnp.array([node.primitives_offset for node in linear_bvh], dtype=jnp.int32)
    n_primitives = jnp.array([node.n_primitives for node in linear_bvh], dtype=jnp.int32)
    second_child_offset = jnp.array([node.second_child_offset for node in linear_bvh], dtype=jnp.int32)
    axis = jnp.array([node.axis for node in linear_bvh], dtype=jnp.int32)
    # Assume that the first child is always stored as current_idx+1.
    child_0 = jnp.arange(n, dtype=jnp.int32) + 1
    return {
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "bounds_centroid": bounds_centroid,
        "primitives_offset": primitives_offset,
        "n_primitives": n_primitives,
        "second_child_offset": second_child_offset,
        "axis": axis,
        "child_0": child_0
    }

def pack_primitives(primitives: List[dict]) -> dict:
    keys = primitives[0].keys()
    packed = {}
    for key in keys:
        packed[key] = jnp.stack([prim[key] for prim in primitives], axis=0)
    return packed



# -------------------------------
# BVH Primitive Helper Classes and Creation
# -------------------------------
class BVHPrimitive:
    def __init__(self, prim_num, bounds):
        self.prim_num = prim_num
        self.bounds = bounds

def compute_triangle_bounds(v1, v2, v3):
    min_point = jnp.minimum(jnp.minimum(v1, v2), v3)
    max_point = jnp.maximum(jnp.maximum(v1, v2), v3)
    centroid = (min_point + max_point) / 2.0
    return AABB(min_point, max_point, centroid)

def create_bvh_primitives(triangles):
    num = int(triangles["vertex_1"].shape[0])
    bvh_prims = []
    for i in range(num):
        v1 = triangles["vertex_1"][i]
        v2 = triangles["vertex_2"][i]
        v3 = triangles["vertex_3"][i]
        bounds = compute_triangle_bounds(v1, v2, v3)
        bvh_prims.append(BVHPrimitive(i, bounds))
    return bvh_prims

def create_primitives(triangles):
    num = int(triangles["vertex_1"].shape[0])
    prims = []
    for i in range(num):
        prim = {
            "v0": triangles["vertex_1"][i],
            "v1": triangles["vertex_2"][i],
            "v2": triangles["vertex_3"][i],
            "centroid": triangles["centroid"][i],
            "normal": triangles["normal"][i],
            "edge_1": triangles["edge_1"][i],
            "edge_2": triangles["edge_2"][i]
        }
        prims.append(prim)
    return prims

# ------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    from io import load_obj, create_triangle_arrays

    file_path = "path/to/your.obj"
    vertices, faces = load_obj(file_path)
    triangles = create_triangle_arrays(vertices, faces)
    from accelerators.bvh import create_primitives, BVHNode
    primitives = create_primitives(triangles)
    from accelerators.hlbvh import BVHPrimitive
    bounded_boxes = []
    N = len(triangles["vertex_1"])
    def compute_triangle_bounds(v1, v2, v3):
        min_point = jnp.minimum(jnp.minimum(v1, v2), v3)
        max_point = jnp.maximum(jnp.maximum(v1, v2), v3)
        centroid = (min_point + max_point) / 2.0
        return AABB(min_point, max_point, centroid)
    for i in range(N):
        bnd = compute_triangle_bounds(triangles["vertex_1"][i],
                                      triangles["vertex_2"][i],
                                      triangles["vertex_3"][i])
        bounded_boxes.append(BVHPrimitive(i, bnd))
    total_nodes = [0]
    ordered_prims = []
    bvh_root = build_hlbvh(primitives, bounded_boxes, ordered_prims, total_nodes)
    print("Constructed HLBVH root:")
    print(bvh_root)
    print("Total nodes:", total_nodes[0])
    print("Ordered primitives count:", len(ordered_prims))
    # Now flatten the tree.
    from time import time
    start_t = time()
    linear_bvh_list = flatten_bvh(bvh_root)
    linear_bvh = pack_linear_bvh(linear_bvh_list)
    end_t = time()
    print("Flattening took:", end_t - start_t, "seconds")
