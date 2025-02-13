# bvh.py
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, List, Tuple
from jax import tree_util

# Import our AABB functions and types.
from primitives.aabb import AABB, union, get_largest_dim, aabb_intersect, get_surface_area
# Import triangle intersection (the watertight method).
from primitives.triangle import triangle_intersect

###############################################################################
# BVH Node (Tree Representation)
###############################################################################
@dataclass
class BVHNode:
    bounds: AABB
    split_axis: int = -1
    first_prim_offset: int = -1  # valid if leaf
    n_primitives: int = 0        # > 0 if leaf
    child_left: Optional['BVHNode'] = None
    child_right: Optional['BVHNode'] = None

def _bvhnode_flatten(node: BVHNode):
    children = (node.bounds, node.split_axis, node.first_prim_offset, node.n_primitives,
                node.child_left, node.child_right)
    aux = None
    return children, aux

def _bvhnode_unflatten(aux, children):
    return BVHNode(*children)

tree_util.register_pytree_node(BVHNode, _bvhnode_flatten, _bvhnode_unflatten)

###############################################################################
# Linear BVH Node (for efficient traversal in jitted code)
###############################################################################
@dataclass
class LinearBVHNode:
    bounds_min: jnp.ndarray  # shape (3,)
    bounds_max: jnp.ndarray  # shape (3,)
    split_axis: int
    first_prim_offset: int
    n_primitives: int
    child_left: int         # index in the linear array; -1 if leaf
    child_right: int        # index in the linear array; -1 if leaf

def _linear_bvh_flatten(node: LinearBVHNode):
    children = (node.bounds_min, node.bounds_max, node.split_axis,
                node.first_prim_offset, node.n_primitives, node.child_left, node.child_right)
    aux = None
    return children, aux

def _linear_bvh_unflatten(aux, children):
    return LinearBVHNode(*children)

tree_util.register_pytree_node(LinearBVHNode, _linear_bvh_flatten, _linear_bvh_unflatten)

###############################################################################
# BucketInfo (used for SAH partitioning)
###############################################################################
@dataclass
class BucketInfo:
    count: int = 0
    bounds: Optional[AABB] = None

    def add(self, b: AABB):
        self.count += 1
        if self.bounds is None:
            self.bounds = b
        else:
            self.bounds = union(self.bounds, b)

###############################################################################
# BVH Builder
###############################################################################
def compute_bounds_for_primitive(primitive_idx: int, primitives: dict) -> AABB:
    """
    Compute the AABB for a triangle and use the true geometric centroid
    (the arithmetic mean of the three vertices) for splitting.
    """
    v0 = primitives["vertex_1"][primitive_idx]
    v1 = primitives["vertex_2"][primitive_idx]
    v2 = primitives["vertex_3"][primitive_idx]
    # Compute the true triangle centroid.
    tri_centroid = (v0 + v1 + v2) / 3.0
    min_point = jnp.minimum(jnp.minimum(v0, v1), v2)
    max_point = jnp.maximum(jnp.maximum(v0, v1), v2)
    return AABB(min_point, max_point, tri_centroid)

def build_bvh(primitives: dict, prim_indices: List[int],
              max_prims_in_node: int = 4, split_method: int = 0) -> Tuple[BVHNode, List[int]]:
    """
    Build a BVH for the primitives indexed by prim_indices.
    If split_method==0, a median‐split is used.
    If split_method==1, SAH bucket partitioning is used (per PBRT).
    Returns:
      - BVH tree (a BVHNode)
      - Ordered list of primitive indices matching the BVH ordering.
    IMPORTANT: After building, call reorder_primitives() with the returned indices
    to reorder your triangle arrays.
    """
    # Precompute bounding boxes.
    prim_bounds = []
    for i in prim_indices:
        aabb = compute_bounds_for_primitive(i, primitives)
        prim_bounds.append(aabb)
    # Maintain an ordered list of primitive indices.
    ordered_indices = list(prim_indices)

    def recursive_build(start: int, end: int) -> BVHNode:
        # Compute union of bounding boxes.
        node_bounds = None
        for i in range(start, end):
            if node_bounds is None:
                node_bounds = prim_bounds[i]
            else:
                node_bounds = union(node_bounds, prim_bounds[i])
        n_prims = end - start
        node = BVHNode(bounds=node_bounds)
        if n_prims <= max_prims_in_node:
            node.first_prim_offset = start
            node.n_primitives = n_prims
            return node
        else:
            # Compute centroid bounds (using true centroids).
            centroid_bounds = None
            for i in range(start, end):
                centroid = prim_bounds[i].centroid
                aabb_centroid = AABB(centroid, centroid, centroid)
                if centroid_bounds is None:
                    centroid_bounds = aabb_centroid
                else:
                    centroid_bounds = union(centroid_bounds, aabb_centroid)
            # Choose splitting method.
            if split_method == 1:
                # SAH bucket partitioning (using 12 buckets).
                n_buckets = 12
                buckets = [BucketInfo() for _ in range(n_buckets)]
                dim = int(get_largest_dim(centroid_bounds))
                for i in range(start, end):
                    extent = centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim]
                    if extent == 0:
                        b = 0
                    else:
                        b = int(n_buckets * ((prim_bounds[i].centroid[dim] - centroid_bounds.min_point[dim]) / extent))
                        if b == n_buckets:
                            b = n_buckets - 1
                    buckets[b].add(prim_bounds[i])
                costs = []
                for i in range(n_buckets - 1):
                    b0 = None
                    b1 = None
                    count0 = 0
                    count1 = 0
                    for j in range(0, i + 1):
                        if buckets[j].bounds is not None:
                            b0 = buckets[j].bounds if b0 is None else union(b0, buckets[j].bounds)
                            count0 += buckets[j].count
                    for j in range(i + 1, n_buckets):
                        if buckets[j].bounds is not None:
                            b1 = buckets[j].bounds if b1 is None else union(b1, buckets[j].bounds)
                            count1 += buckets[j].count
                    if b0 is None or b1 is None:
                        costs.append(1e10)
                    else:
                        cost = (count0 * get_surface_area(b0) + count1 * get_surface_area(b1)) / get_surface_area(node_bounds)
                        costs.append(cost)
                min_cost = min(costs)
                min_cost_split_bucket = costs.index(min_cost)
                leaf_cost = n_prims
                if n_prims > max_prims_in_node or min_cost < leaf_cost:
                    # Partition based on bucket predicate.
                    dim = int(get_largest_dim(centroid_bounds))
                    def predicate(i):
                        extent = centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim]
                        if extent == 0:
                            b = 0
                        else:
                            b = int(n_buckets * ((prim_bounds[i].centroid[dim] - centroid_bounds.min_point[dim]) / extent))
                            if b == n_buckets:
                                b = n_buckets - 1
                        return b <= min_cost_split_bucket
                    left_indices = [i for i in range(start, end) if predicate(i)]
                    right_indices = [i for i in range(start, end) if not predicate(i)]
                    if len(left_indices) == 0 or len(right_indices) == 0:
                        mid = start + n_prims // 2
                    else:
                        mid = start + len(left_indices)
                        new_order = left_indices + right_indices
                        prim_bounds[start:end] = [prim_bounds[i] for i in new_order]
                        ordered_indices[start:end] = [ordered_indices[i] for i in new_order]
                else:
                    mid = start + n_prims // 2
            else:
                # Median split.
                dim = int(get_largest_dim(centroid_bounds))
                indices = list(range(start, end))
                indices.sort(key=lambda j: float(prim_bounds[j].centroid[dim]))
                mid = start + (n_prims // 2)
                prim_bounds[start:end] = [prim_bounds[j] for j in indices]
                ordered_indices[start:end] = [ordered_indices[j] for j in indices]
            if mid == start or mid == end:
                node.first_prim_offset = start
                node.n_primitives = n_prims
                return node
            left = recursive_build(start, mid)
            right = recursive_build(mid, end)
            node.child_left = left
            node.child_right = right
            return node

    root = recursive_build(0, len(prim_indices))
    return root, ordered_indices

###############################################################################
# Flatten BVH tree into a linear array for fast traversal in jitted code.
###############################################################################
def flatten_bvh(root: BVHNode) -> List[LinearBVHNode]:
    nodes: List[LinearBVHNode] = []

    def recursive_flatten(node: BVHNode) -> int:
        current_idx = len(nodes)
        nodes.append(None)  # placeholder
        if node.child_left is None and node.child_right is None:
            linear = LinearBVHNode(
                bounds_min=node.bounds.min_point,
                bounds_max=node.bounds.max_point,
                split_axis=node.split_axis,
                first_prim_offset=node.first_prim_offset,
                n_primitives=node.n_primitives,
                child_left=-1,
                child_right=-1
            )
        else:
            left_idx = recursive_flatten(node.child_left)
            right_idx = recursive_flatten(node.child_right)
            linear = LinearBVHNode(
                bounds_min=node.bounds.min_point,
                bounds_max=node.bounds.max_point,
                split_axis=node.split_axis,
                first_prim_offset=-1,
                n_primitives=0,
                child_left=left_idx,
                child_right=right_idx
            )
        nodes[current_idx] = linear
        return current_idx

    recursive_flatten(root)
    return nodes

###############################################################################
# BVH Intersection (Traversing a flattened BVH using a stack)
###############################################################################
@jax.jit
def bvh_intersect(ray_origin: jnp.ndarray,
                  ray_direction: jnp.ndarray,
                  linear_bvh: dict,
                  primitives: dict,
                  t_max: float) -> float:
    """
    Traverse the flattened BVH (as a dictionary of arrays) and test for
    ray intersections with the primitives (triangles). Returns the closest intersection
    distance (t) found; if no hit is found, returns t_max.
    """
    num_nodes = linear_bvh["bounds_min"].shape[0]
    max_stack = num_nodes

    stack = -jnp.ones((max_stack,), dtype=jnp.int32)
    stack = stack.at[0].set(0)
    stack_ptr = jnp.array(1, dtype=jnp.int32)
    best_t = t_max

    def cond_fn(state):
        _stack, sp, _best_t = state
        return sp > 0

    def body_fn(state):
        _stack, sp, _best_t = state
        node_idx = _stack[sp - 1]
        sp = sp - 1

        node_min = linear_bvh["bounds_min"][node_idx]
        node_max = linear_bvh["bounds_max"][node_idx]
        node_centroid = (node_min + node_max) * 0.5
        node_aabb = AABB(node_min, node_max, node_centroid)
        hit_node = aabb_intersect(node_aabb, ray_origin, ray_direction)

        def if_hit(_):
            n_prims = linear_bvh["n_primitives"][node_idx]
            def leaf_fn(_):
                first = linear_bvh["first_prim_offset"][node_idx]
                n = n_prims
                def prim_body(i, best_t_in):
                    prim_idx = first + i
                    hit, t_candidate = triangle_intersect(
                        ray_origin, ray_direction,
                        primitives["vertex_1"][prim_idx],
                        primitives["vertex_2"][prim_idx],
                        primitives["vertex_3"][prim_idx],
                        best_t_in
                    )
                    best_t_in = jax.lax.select(hit & (t_candidate < best_t_in),
                                               t_candidate, best_t_in)
                    return best_t_in
                best_after = jax.lax.fori_loop(0, n, prim_body, _best_t)
                return best_after, _stack, sp
            def internal_fn(_):
                left = linear_bvh["child_left"][node_idx]
                right = linear_bvh["child_right"][node_idx]
                new_stack = _stack.at[sp].set(left)
                new_stack = new_stack.at[sp + 1].set(right)
                sp_new = sp + 2
                return _best_t, new_stack, sp_new
            return jax.lax.cond(n_prims > 0, leaf_fn, internal_fn, operand=None)
        def if_miss(_):
            return _best_t, _stack, sp
        best_t_new, stack_new, sp_new = jax.lax.cond(hit_node, if_hit, if_miss, operand=None)
        return (stack_new, sp_new, best_t_new)

    final_stack, final_sp, best_t_final = jax.lax.while_loop(cond_fn, body_fn, (stack, stack_ptr, best_t))
    return best_t_final

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def reorder_primitives(primitives: dict, ordered_indices: List[int]) -> dict:
    """
    Reorder the triangle arrays (all keys) using the ordered_indices.
    """
    ordered_indices_arr = jnp.array(ordered_indices, dtype=jnp.int32)
    new_primitives = {}
    for key in primitives:
        new_primitives[key] = primitives[key][ordered_indices_arr]
    return new_primitives

def linear_bvh_to_dict(linear_nodes: List[LinearBVHNode]) -> dict:
    return {
        "bounds_min": jnp.stack([node.bounds_min for node in linear_nodes], axis=0),
        "bounds_max": jnp.stack([node.bounds_max for node in linear_nodes], axis=0),
        "split_axis": jnp.array([node.split_axis for node in linear_nodes], dtype=jnp.int32),
        "first_prim_offset": jnp.array([node.first_prim_offset for node in linear_nodes], dtype=jnp.int32),
        "n_primitives": jnp.array([node.n_primitives for node in linear_nodes], dtype=jnp.int32),
        "child_left": jnp.array([node.child_left for node in linear_nodes], dtype=jnp.int32),
        "child_right": jnp.array([node.child_right for node in linear_nodes], dtype=jnp.int32),
    }

###############################################################################
# DEBUG FUNCTIONS
###############################################################################
def debug_print_bvh_tree(node: BVHNode, depth: int = 0):
    """
    Recursively prints the BVH tree structure.
    """
    indent = "  " * depth
    bmin = node.bounds.min_point.tolist()
    bmax = node.bounds.max_point.tolist()
    centroid = node.bounds.centroid.tolist()
    if node.child_left is None and node.child_right is None:
        print(f"{indent}Leaf: offset={node.first_prim_offset}, n_primitives={node.n_primitives}, "
              f"bounds_min={bmin}, bounds_max={bmax}, centroid={centroid}")
    else:
        print(f"{indent}Internal: split_axis={node.split_axis}, "
              f"bounds_min={bmin}, bounds_max={bmax}, centroid={centroid}")
        if node.child_left is not None:
            debug_print_bvh_tree(node.child_left, depth + 1)
        if node.child_right is not None:
            debug_print_bvh_tree(node.child_right, depth + 1)

def debug_print_flattened_bvh(flat_bvh: dict):
    """
    Prints the flattened BVH (dictionary of arrays) node-by-node.
    """
    n_nodes = flat_bvh["bounds_min"].shape[0]
    for i in range(n_nodes):
        bmin = flat_bvh["bounds_min"][i].tolist()
        bmax = flat_bvh["bounds_max"][i].tolist()
        split_axis = int(flat_bvh["split_axis"][i])
        first_offset = int(flat_bvh["first_prim_offset"][i])
        n_primitives = int(flat_bvh["n_primitives"][i])
        child_left = int(flat_bvh["child_left"][i])
        child_right = int(flat_bvh["child_right"][i])
        print(f"Node {i}: split_axis={split_axis}, first_offset={first_offset}, "
              f"n_primitives={n_primitives}, child_left={child_left}, child_right={child_right}, "
              f"bounds_min={bmin}, bounds_max={bmax}")
