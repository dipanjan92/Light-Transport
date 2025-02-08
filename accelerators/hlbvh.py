import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import NamedTuple, Dict, Tuple

# -----------------------------------------------------------------------------
# Constants and Helper Functions
# -----------------------------------------------------------------------------
INF = 1e10

# (Assume that enclose_volumes, enclose_centroids, get_surface_area, offset_bounds,
#  partition, compute_bounding_box, and get_largest_dim have been implemented in JAX.)



# -----------------------------------------------------------------------------
# Morton Encoding Helpers
# -----------------------------------------------------------------------------
# We convert the left_shift_3 and encode_morton_3 functions to JAX.
def left_shift_3(x: int) -> int:
    # x must be less than 2^10.
    if x == (1 << 10):
        x = x - 1
    # We perform bit–manipulations using Python’s int.
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x

def encode_morton_3(v: jnp.ndarray) -> int:
    # Here v is a 3–element jnp.array of nonnegative values.
    # Convert components to Python ints after scaling.
    xs = int(v[0])
    ys = int(v[1])
    zs = int(v[2])
    return (left_shift_3(zs) << 2) | (left_shift_3(ys) << 1) | left_shift_3(xs)

# -----------------------------------------------------------------------------
# Data Structures for Morton Primitives and LBVH Treelets
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Radix Sort (Vectorized style)
# -----------------------------------------------------------------------------
def radix_sort(morton_prims: Tuple[MortonPrimitive, ...]) -> Tuple[MortonPrimitive, ...]:
    # Convert the tuple to a list of Python ints for morton_code and prim_ix.
    # In a fully vectorized version, you would store these in jnp.arrays.
    v = list(morton_prims)
    bits_per_pass = 6
    n_bits = 30
    n_passes = n_bits // bits_per_pass
    for _pass in range(n_passes):
        low_bit = _pass * bits_per_pass
        n_buckets = 1 << bits_per_pass
        bucket_count = [0] * n_buckets
        out = [None] * len(v)
        bit_mask = (1 << bits_per_pass) - 1
        for mp in v:
            bucket = (mp.morton_code >> low_bit) & bit_mask
            bucket_count[bucket] += 1
        out_ix = [0] * n_buckets
        for i in range(1, n_buckets):
            out_ix[i] = out_ix[i - 1] + bucket_count[i - 1]
        for mp in v:
            bucket = (mp.morton_code >> low_bit) & bit_mask
            out[out_ix[bucket]] = mp
            out_ix[bucket] += 1
        v = out
    return tuple(v)

# -----------------------------------------------------------------------------
# Build Upper SAH Function
# -----------------------------------------------------------------------------
def build_upper_sah(treelet_roots: Tuple[BVHNode, ...], start: int, end: int, total_nodes: int) -> BVHNode:
    # Recursively build an upper-level SAH node from a list of LBVH treelet roots.
    n_nodes = end - start
    if n_nodes == 1:
        return treelet_roots[start]
    total_nodes += 1
    # Compute the union of bounds for nodes in [start, end).
    bounds = None
    for i in range(start, end):
        bounds = union(bounds, treelet_roots[i].bounds) if bounds is not None else treelet_roots[i].bounds
    centroid_bounds = None
    for i in range(start, end):
        centroid = (treelet_roots[i].bounds.min_point + treelet_roots[i].bounds.max_point) * 0.5
        centroid_bounds = enclose_centroids(centroid_bounds, centroid)
    dim = jnp.argmax(centroid_bounds.max_point - centroid_bounds.min_point)
    n_buckets = 12
    buckets = [BucketInfo(0, AABB(jnp.array([INF, INF, INF]),
                                  jnp.array([-INF, -INF, -INF]),
                                  jnp.array([INF, INF, INF])))
               for _ in range(n_buckets)]
    for i in range(start, end):
        centroid = (treelet_roots[i].bounds.min_point[dim] + treelet_roots[i].bounds.max_point[dim]) * 0.5
        b = int(n_buckets * ((centroid - centroid_bounds.min_point[dim]) /
                             (centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim])))
        if b == n_buckets:
            b = n_buckets - 1
        buckets[b] = BucketInfo(buckets[b].count + 1,
                                union(buckets[b].bounds, treelet_roots[i].bounds))
    costs = []
    for i in range(n_buckets - 1):
        count0 = 0
        count1 = 0
        bound0 = None
        bound1 = None
        for j in range(i + 1):
            bound0 = union(bound0, buckets[j].bounds) if bound0 is not None else buckets[j].bounds
            count0 += buckets[j].count
        for j in range(i + 1, n_buckets):
            bound1 = union(bound1, buckets[j].bounds) if bound1 is not None else buckets[j].bounds
            count1 += buckets[j].count
        _cost = 0.125 + (count0 * bound0.get_surface_area() + count1 * bound1.get_surface_area()) / bounds.get_surface_area()
        costs.append(_cost)
    min_cost_split_bucket = min(range(len(costs)), key=lambda i: costs[i])
    # Partition treelet_roots using a partition predicate (implemented via PartitionWrapper)
    partitioner = PartitionWrapper(n_buckets, centroid_bounds, int(dim), min_cost_split_bucket)
    # (In JAX you would want a vectorized partition. Here we use Python’s sorted as a placeholder.)
    sorted_roots = sorted(treelet_roots[start:end],
                          key=lambda node: (node.bounds.min_point[dim] + node.bounds.max_point[dim]) * 0.5)
    mid = (start + end) // 2
    left = build_upper_sah(tuple(sorted_roots), 0, mid - start, total_nodes)
    right = build_upper_sah(tuple(sorted_roots), mid - start, end - start, total_nodes)
    node = BVHNode(bounds=bounds, child_0=-1, child_1=-1, split_axis=int(dim))
    node = node.init_interior(int(dim), left, right, bounds)
    return node

# -----------------------------------------------------------------------------
# Emit LBVH Function (Recursive)
# -----------------------------------------------------------------------------
def emit_lbvh(build_nodes: Tuple[BVHNode, ...],
              primitives: Tuple[Any, ...],
              bounded_boxes: Tuple[AABB, ...],
              morton_prims: Tuple[MortonPrimitive, ...],
              n_primitives: int,
              total_nodes: int,
              ordered_prims: jnp.ndarray,
              ordered_prims_offset: int,
              bit_index: int) -> BVHNode:
    # If bit_index == -1 or the number of primitives is less than a threshold,
    # create a leaf.
    max_prims_in_node = int(0.1 * len(bounded_boxes))
    if bit_index == -1 or n_primitives < max_prims_in_node:
        total_nodes += 1
        # Pop a node from build_nodes (here we assume build_nodes is a tuple and we simply take the first)
        node = build_nodes[0]
        first_prim_offset = ordered_prims_offset
        ordered_prims_offset += n_primitives
        bounds = None
        for i in range(n_primitives):
            primitive_index = morton_prims[i].prim_ix
            ordered_prims = ordered_prims.at[first_prim_offset + i].set(primitives[primitive_index])
            bounds = union(bounds, bounded_boxes[primitive_index]) if bounds is not None else bounded_boxes[primitive_index]
        return node.init_leaf(first_prim_offset, n_primitives, bounds)
    else:
        mask = 1 << bit_index
        # If all morton codes share the same bits at this level, descend.
        if (morton_prims[0].morton_code & mask) == (morton_prims[n_primitives - 1].morton_code & mask):
            return emit_lbvh(build_nodes, primitives, bounded_boxes, morton_prims,
                             n_primitives, total_nodes, ordered_prims, ordered_prims_offset, bit_index - 1)
        # Find split offset using an interval predicate.
        interval_wrapper = IntervalWrapper(morton_prims, mask)
        # (Here we use a simple loop to find the first index where the predicate fails.)
        split_offset = 0
        for i in range(n_primitives):
            if not interval_wrapper.interval_pred(i):
                split_offset = i
                break
        split_offset = split_offset + 1
        # Create interior LBVH node recursively.
        total_nodes += 1
        left = emit_lbvh(build_nodes, primitives, bounded_boxes, morton_prims[:split_offset],
                         split_offset, total_nodes, ordered_prims, ordered_prims_offset, bit_index - 1)
        right = emit_lbvh(build_nodes, primitives, bounded_boxes, morton_prims[split_offset:],
                          n_primitives - split_offset, total_nodes, ordered_prims, ordered_prims_offset, bit_index - 1)
        axis = bit_index % 3
        node = build_nodes[0].init_interior(axis, left, right, None)
        return node

# -----------------------------------------------------------------------------
# Build HLBVH
# -----------------------------------------------------------------------------
def build_hlbvh(primitives: Tuple[Any, ...],
                bounded_boxes: Tuple[AABB, ...],
                ordered_prims: jnp.ndarray) -> BVHNode:
    # Compute global bounds for centroids.
    bounds = None
    for box in bounded_boxes:
        bounds = enclose_centroids(bounds, box.centroid) if bounds is not None else box.centroid
    # Create Morton primitives.
    morton_prims_list = []
    morton_bits = 10
    morton_scale = 1 << morton_bits
    for i, box in enumerate(bounded_boxes):
        # Assume each bounded_box has a field prim_num.
        prim_ix = i  # or box.prim_num
        centroid_offset = offset_bounds(bounds, box.centroid)  # Must be implemented in JAX.
        scaled_offset = centroid_offset * morton_scale
        code = encode_morton_3(scaled_offset)
        morton_prims_list.append(MortonPrimitive(prim_ix, code))
    morton_prims = tuple(morton_prims_list)
    morton_prims = radix_sort(morton_prims)
    # Build LBVH treelets.
    treelets_to_build = []
    start = 0
    end = 1
    mask = 0x3FF << 22  # Example mask (adapt as needed)
    N = len(morton_prims)
    while end <= N:
        if end == N or ((morton_prims[start].morton_code & mask) != (morton_prims[end - 1].morton_code & mask)):
            n_primitives = end - start
            # Prepare a pool of BVHNodes for this treelet.
            max_bvh_nodes = 2 * n_primitives - 1
            build_nodes = tuple(BVHNode(bounds=AABB(jnp.array([INF, INF, INF]),
                                                    jnp.array([-INF, -INF, -INF]),
                                                    jnp.array([INF, INF, INF])))
                                for _ in range(max_bvh_nodes))
            treelets_to_build.append(LBVHTreelet(start, n_primitives, build_nodes))
            start = end
        end += 1
    # Now, for each treelet, build the LBVH (using emit_lbvh).
    ordered_prims_offset = 0
    finished_treelets = []
    total_nodes = 0
    first_bit_index = 29 - 12  # As in the original code.
    for treelet in treelets_to_build:
        lbvh_node = emit_lbvh(treelet.build_nodes, primitives, bounded_boxes,
                              morton_prims[treelet.start_ix:], treelet.n_primitives,
                              total_nodes, ordered_prims, ordered_prims_offset, first_bit_index)
        finished_treelets.append(lbvh_node)
    # total_nodes can be computed from the sizes of finished_treelets.
    total_nodes = sum(1 for _ in finished_treelets)
    # Finally, build the upper-level SAH tree.
    upper_bvh = build_upper_sah(tuple(finished_treelets), 0, len(finished_treelets), total_nodes)
    return upper_bvh

# -----------------------------------------------------------------------------
# End of HLBVH Conversion
# -----------------------------------------------------------------------------

