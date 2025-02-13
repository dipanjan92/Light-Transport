# bvh.py
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Tuple, Any
from jax import lax

# Global constants
INF = 1e10
MAX_DEPTH = 64  # maximum traversal stack depth
MAX_LEAF_PRIMS = 16  # maximum number of primitives we loop over in a leaf

# Import your other modules (make sure these modules are jittable)
from primitives.aabb import AABB, union, union_p, aabb_intersect, aabb_hit_distance, aabb_intersect_p
from primitives.intersects import Intersection, set_intersection
from primitives.ray import Ray, spawn_ray
from primitives.triangle import intersect_triangle  # Must be jittable

# -------------------------------
# Data Structures
# -------------------------------

@dataclass
class BVHNode:
    bounds: AABB = None
    first_prim_offset: int = -1
    n_primitives: int = 0
    child_0: int = -1
    child_1: int = -1
    split_axis: int = -1

    def init_leaf(self, first: int, n: int, box: AABB):
        self.first_prim_offset = first
        self.n_primitives = n
        self.bounds = box

    def init_interior(self, axis: int, c0: int, c1: int, box: AABB):
        self.child_0 = c0
        self.child_1 = c1
        self.split_axis = axis
        self.bounds = box
        self.n_primitives = 0

@dataclass
class LinearBVHNode:
    bounds: AABB = None
    primitives_offset: int = -1
    second_child_offset: int = -1
    n_primitives: int = 0
    axis: int = -1

@dataclass
class BucketInfo:
    count: int = 0
    bounds: AABB = None

@dataclass
class BuildParams:
    n_triangles: int = 0
    n_ordered_prims: int = 0
    total_nodes: int = 0
    split_method: int = 0

# -------------------------------
# Helper functions for the BVH build stack
# -------------------------------

def push(stack: List[Tuple[int, int, int, int]], start: int, end: int, parent_idx: int, is_second_child: int):
    stack.append((start, end, parent_idx, is_second_child))

def pop(stack: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    return stack.pop()

# -------------------------------
# Primitive Partitioning Helpers
# -------------------------------

def partition_equal_counts(bvh_primitives: List[Any], start: int, end: int, dim: int) -> int:
    sublist = bvh_primitives[start:end]
    sublist.sort(key=lambda prim: float(prim.bounds.centroid[dim]))
    bvh_primitives[start:end] = sublist
    mid = (start + end) // 2
    return mid

def partition_middle(bvh_primitives: List[Any], start: int, end: int, dim: int, centroid_bounds: AABB) -> int:
    pmid = (centroid_bounds.min_point[dim] + centroid_bounds.max_point[dim]) / 2.0
    left = start
    right = end - 1
    while left <= right:
        while left <= right and float(bvh_primitives[left].bounds.centroid[dim]) < pmid:
            left += 1
        while left <= right and float(bvh_primitives[right].bounds.centroid[dim]) >= pmid:
            right -= 1
        if left < right:
            bvh_primitives[left], bvh_primitives[right] = bvh_primitives[right], bvh_primitives[left]
            left += 1
            right -= 1
    mid = left
    if mid == start or mid == end:
        mid = (start + end) // 2
    return mid

def partition_sah(bvh_primitives: List[Any],
                  start: int,
                  end: int,
                  dim: int,
                  centroid_bounds: AABB,
                  costs: List[float],
                  buckets: List[BucketInfo],
                  bounds: AABB,
                  max_prims_in_node: int) -> int:
    nBuckets = 12
    nSplits = nBuckets - 1
    minCostSplitBucket = -1
    minCost = INF
    leafCost = (end - start)
    for b in range(nBuckets):
        buckets[b].count = 0
        buckets[b].bounds = AABB(jnp.array([INF, INF, INF]),
                                  jnp.array([-INF, -INF, -INF]),
                                  jnp.array([0.0, 0.0, 0.0]))
    for i in range(start, end):
        centroid = bvh_primitives[i].bounds.centroid
        denom = float(centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim])
        b_idx = 0 if denom == 0 else int(nBuckets * (float(centroid[dim]) - float(centroid_bounds.min_point[dim])) / denom)
        if b_idx == nBuckets:
            b_idx = nBuckets - 1
        buckets[b_idx].count += 1
        buckets[b_idx].bounds = union(buckets[b_idx].bounds, bvh_primitives[i].bounds)
    countBelow = 0
    countAbove = 0
    boundBelow = AABB(jnp.array([INF, INF, INF]),
                      jnp.array([-INF, -INF, -INF]),
                      jnp.array([0.0, 0.0, 0.0]))
    boundAbove = AABB(jnp.array([INF, INF, INF]),
                      jnp.array([-INF, -INF, -INF]),
                      jnp.array([0.0, 0.0, 0.0]))
    for j in range(nSplits):
        boundBelow = union(boundBelow, buckets[j].bounds)
        countBelow += buckets[j].count
        costs[j] = countBelow * boundBelow.get_surface_area()
    for k in range(nSplits - 1, -1, -1):
        boundAbove = union(boundAbove, buckets[k + 1].bounds)
        countAbove += buckets[k + 1].count
        costs[k] += countAbove * boundAbove.get_surface_area()
    for m in range(nSplits):
        if costs[m] < minCost:
            minCost = costs[m]
            minCostSplitBucket = m
    minCost = minCost / bounds.get_surface_area() if bounds.get_surface_area() != 0 else INF
    mid = start
    if (end - start) > max_prims_in_node or minCost < leafCost:
        mid = start
        for n in range(start, end):
            centroid = bvh_primitives[n].bounds.centroid
            denom = float(centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim])
            b_idx = 0 if denom == 0 else int(nBuckets * (float(centroid[dim]) - float(centroid_bounds.min_point[dim])) / denom)
            if b_idx == nBuckets:
                b_idx = nBuckets - 1
            if b_idx <= minCostSplitBucket:
                bvh_primitives[mid], bvh_primitives[n] = bvh_primitives[n], bvh_primitives[mid]
                mid += 1
    else:
        mid = start
    return mid

# -------------------------------
# BVH Build and Flatten Routines
# -------------------------------

def build_bvh(primitives: List[Any],
              bvh_primitives: List[Any],
              _start: int,
              _end: int,
              ordered_prims: List[Any],
              split_method: int) -> Tuple[List[BVHNode], List[Any]]:
    nodes: List[BVHNode] = []
    total_nodes = 0
    ordered_prims_idx = 0
    stack: List[Tuple[int, int, int, int]] = []
    push(stack, _start, _end, -1, 0)
    max_prims_in_node = max(4, int(0.1 * len(bvh_primitives)))
    costs = [0.0] * 12
    buckets = [BucketInfo(0, AABB(jnp.array([INF, INF, INF]),
                                  jnp.array([-INF, -INF, -INF]),
                                  jnp.array([0.0, 0.0, 0.0])))
               for _ in range(12)]
    while stack:
        start, end, parent_idx, is_second_child = pop(stack)
        current_node_idx = total_nodes
        total_nodes += 1
        node = BVHNode()
        nodes.append(node)
        if parent_idx != -1:
            if is_second_child:
                nodes[parent_idx].child_1 = current_node_idx
            else:
                nodes[parent_idx].child_0 = current_node_idx
        bounds = AABB(jnp.array([INF, INF, INF]),
                      jnp.array([-INF, -INF, -INF]),
                      jnp.array([0.0, 0.0, 0.0]))
        for i in range(start, end):
            bounds = union(bounds, bvh_primitives[i].bounds)
        if bounds.get_surface_area() == 0 or (end - start) == 1:
            first_prim_offset = ordered_prims_idx
            for i in range(start, end):
                prim_num = bvh_primitives[i].prim_num
                ordered_prims.append(primitives[prim_num])
                ordered_prims_idx += 1
            node.init_leaf(first_prim_offset, end - start, bounds)
        else:
            centroid_bounds = AABB(jnp.array([INF, INF, INF]),
                                   jnp.array([-INF, -INF, -INF]),
                                   jnp.array([0.0, 0.0, 0.0]))
            for i in range(start, end):
                centroid_bounds = union_p(centroid_bounds, bvh_primitives[i].bounds.centroid)
            diff = centroid_bounds.max_point - centroid_bounds.min_point
            dim = int(jnp.argmax(diff))
            if float(centroid_bounds.max_point[dim]) == float(centroid_bounds.min_point[dim]):
                first_prim_offset = ordered_prims_idx
                for i in range(start, end):
                    prim_num = bvh_primitives[i].prim_num
                    ordered_prims.append(primitives[prim_num])
                    ordered_prims_idx += 1
                node.init_leaf(first_prim_offset, end - start, bounds)
            else:
                mid = (start + end) // 2
                if split_method == 2:
                    mid = partition_equal_counts(bvh_primitives, start, end, dim)
                elif split_method == 1:
                    mid = partition_middle(bvh_primitives, start, end, dim, centroid_bounds)
                elif split_method == 0:
                    if (end - start) <= 2:
                        mid = partition_equal_counts(bvh_primitives, start, end, dim)
                    else:
                        mid = partition_sah(bvh_primitives, start, end, dim, centroid_bounds, costs, buckets, bounds, max_prims_in_node)
                node.split_axis = dim
                node.n_primitives = 0
                node.bounds = bounds
                push(stack, mid, end, current_node_idx, 1)
                push(stack, start, mid, current_node_idx, 0)
    return nodes, ordered_prims

def flatten_bvh(nodes: List[BVHNode], root: int) -> List[LinearBVHNode]:
    linear_bvh: List[LinearBVHNode] = []
    stack: List[Tuple[int, int, int]] = []
    stack.append((root, -1, 0))
    offset = 0
    while stack:
        node_idx, parent_idx, is_second_child = stack.pop()
        if node_idx == -1:
            continue
        current_idx = offset
        linear_node = LinearBVHNode()
        linear_node.bounds = nodes[node_idx].bounds
        linear_node.primitives_offset = nodes[node_idx].first_prim_offset
        linear_node.n_primitives = nodes[node_idx].n_primitives
        linear_node.axis = nodes[node_idx].split_axis
        linear_node.second_child_offset = nodes[node_idx].child_1  # child_1 stores second child index
        linear_bvh.append(linear_node)
        offset += 1
        if parent_idx != -1 and is_second_child:
            linear_bvh[parent_idx].second_child_offset = current_idx
        if nodes[node_idx].n_primitives == 0:
            if nodes[node_idx].child_1 != -1:
                stack.append((nodes[node_idx].child_1, current_idx, 1))
            if nodes[node_idx].child_0 != -1:
                stack.append((nodes[node_idx].child_0, current_idx, 0))
    return linear_bvh

# -------------------------------
# Packed BVH and Primitive Helpers
# -------------------------------

def pack_linear_bvh(linear_bvh: List[LinearBVHNode]) -> dict:
    n = len(linear_bvh)
    bounds_min = jnp.stack([node.bounds.min_point for node in linear_bvh], axis=0)
    bounds_max = jnp.stack([node.bounds.max_point for node in linear_bvh], axis=0)
    bounds_centroid = jnp.stack([node.bounds.centroid for node in linear_bvh], axis=0)
    primitives_offset = jnp.array([node.primitives_offset for node in linear_bvh], dtype=jnp.int32)
    n_primitives = jnp.array([node.n_primitives for node in linear_bvh], dtype=jnp.int32)
    second_child_offset = jnp.array([node.second_child_offset for node in linear_bvh], dtype=jnp.int32)
    axis = jnp.array([node.axis for node in linear_bvh], dtype=jnp.int32)
    # Assume that the first child is always stored as child_0 = current + 1
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

def pack_primitives(primitives: List[Any]) -> dict:
    keys = primitives[0].keys()
    packed = {}
    for key in keys:
        packed[key] = jnp.stack([prim[key] for prim in primitives], axis=0)
    return packed

# -------------------------------
# BVH Intersection Routines (Fully Jitted)
# -------------------------------

@jax.jit
def intersect_bvh(ray: Ray, primitives: dict, bvh: dict, t_max: float) -> Intersection:
    """
    Intersect a ray with a BVH (packed in a dictionary) following a PBRT–style
    iterative traversal. The BVH dictionary is assumed to have the following keys:
      - "bounds_min":   (N,3) array of node minimum bounds.
      - "bounds_max":   (N,3) array of node maximum bounds.
      - "n_primitives": (N,) array (int32) indicating the number of primitives in a leaf.
      - "primitives_offset": (N,) array (int32) indicating the starting index in the
                             primitives array for a leaf.
      - "axis":         (N,) array (int32) indicating the split axis (for interior nodes).
      - "second_child_offset": (N,) array (int32) for the second child of an interior node.

    The primitives dictionary is assumed to contain triangle data under keys "v0", "v1", "v2".
    (The triangle intersection function should return a tuple (hit, t_candidate).)

    The function returns an Intersection record (with intersected==0 if no hit was found).
    """
    # Precompute ray data.
    invDir = 1.0 / ray.direction
    # Create an array of booleans (or 0/1 ints) indicating if each component is negative.
    dirIsNeg = jnp.array([invDir[0] < 0, invDir[1] < 0, invDir[2] < 0])

    # Initialize the intersection record with no hit (t_max).
    init_hit = Intersection(
        min_distance=t_max,
        intersected_point=ray.origin,
        normal=jnp.array([0.0, 0.0, 0.0]),
        shading_normal=jnp.array([0.0, 0.0, 0.0]),
        dpdu=jnp.array([0.0, 0.0, 0.0]),
        dpdv=jnp.array([0.0, 0.0, 0.0]),
        dndu=jnp.array([0.0, 0.0, 0.0]),
        dndv=jnp.array([0.0, 0.0, 0.0]),
        nearest_object=-1,
        intersected=0
    )

    # Initialize the traversal state.
    # state = (currentNodeIndex, toVisitOffset, stack, tMax, intersection, nodesVisited)
    currentNodeIndex = jnp.int32(0)
    toVisitOffset = jnp.int32(0)
    # Allocate a fixed-size stack; fill with -1.
    stack = -jnp.ones((MAX_DEPTH,), dtype=jnp.int32)
    nodesVisited = jnp.int32(0)
    state = (currentNodeIndex, toVisitOffset, stack, t_max, init_hit, nodesVisited)

    def cond_fn(state):
        curr, toVisit, stack, tCurr, inter, nodesVis = state
        # Continue while there is a valid current node.
        return curr != -1

    def body_fn(state):
        curr, toVisit, stack, tCurr, inter, nodesVis = state
        nodesVis = nodesVis + 1

        # Load the current node’s data from the packed BVH.
        node_bounds_min = bvh["bounds_min"][curr]
        node_bounds_max = bvh["bounds_max"][curr]
        node_nPrims = bvh["n_primitives"][curr]
        node_axis = bvh["axis"][curr]  # valid for interior nodes
        node_secondChild = bvh["second_child_offset"][curr]

        # Test intersection of the ray with the node’s bounding box.
        hitAABB = aabb_intersect_p(ray.origin, ray.direction, tCurr,
                                   invDir, dirIsNeg,
                                   node_bounds_min, node_bounds_max)

        # If the ray hits the node’s bounds…
        def if_hit(_):
            # If this is a leaf node, test all primitives.
            def if_leaf(_):
                # For leaves, loop over the primitives in the node.
                prim_offset = bvh["primitives_offset"][curr]

                # We'll use a fori_loop to accumulate the best hit.
                def leaf_body(i, carry):
                    best_inter, best_t = carry
                    primIdx = prim_offset + i
                    v0 = primitives["v0"][primIdx]
                    v1 = primitives["v1"][primIdx]
                    v2 = primitives["v2"][primIdx]
                    hit_prim, t_candidate = intersect_triangle(
                        ray.origin, ray.direction, v0, v1, v2, best_t)

                    # If this primitive is hit, update the intersection record.
                    def update_hit(_):
                        return set_intersection(ray.origin, ray.direction, v0, v1, v2, t_candidate)

                    new_inter = lax.cond(hit_prim, update_hit, lambda _: best_inter, operand=None)
                    new_t = jnp.minimum(best_t, lax.select(hit_prim, t_candidate, best_t))
                    return (new_inter, new_t)

                best_inter, best_t = lax.fori_loop(0, node_nPrims, leaf_body, (inter, tCurr))

                # After testing the leaf, pop the next node from the stack (if any).
                def pop_stack(_):
                    new_toVisit = toVisit - 1
                    new_curr = stack[new_toVisit]
                    return new_curr, new_toVisit

                new_curr, new_toVisit = lax.cond(toVisit == 0,
                                                 lambda _: (jnp.int32(-1), toVisit),
                                                 pop_stack,
                                                 operand=None)
                return new_curr, new_toVisit, stack, best_t, best_inter, nodesVis

            # Otherwise, interior node.
            def if_interior(_):
                # For interior nodes, choose which child is near.
                # (In PBRT, the ordering depends on the sign of the ray direction along the split axis.)
                def if_neg(dummy):
                    # If the ray direction is negative along node_axis:
                    new_stack = stack.at[toVisit].set(curr + 1)
                    new_toVisit = toVisit + 1
                    new_curr = node_secondChild
                    return new_curr, new_toVisit, new_stack

                def if_not_neg(dummy):
                    new_stack = stack.at[toVisit].set(node_secondChild)
                    new_toVisit = toVisit + 1
                    new_curr = curr + 1
                    return new_curr, new_toVisit, new_stack

                new_curr, new_toVisit, new_stack = lax.cond(dirIsNeg[node_axis],
                                                            if_neg,
                                                            if_not_neg,
                                                            operand=None)
                return new_curr, new_toVisit, new_stack, tCurr, inter, nodesVis

            return lax.cond(node_nPrims > 0, if_leaf, if_interior, operand=None)

        # If the ray misses the node’s bounds, simply pop the next node.
        def if_miss(_):
            def pop_stack(_):
                new_toVisit = toVisit - 1
                new_curr = stack[new_toVisit]
                return new_curr, new_toVisit

            new_curr, new_toVisit = lax.cond(toVisit == 0,
                                             lambda _: (jnp.int32(-1), toVisit),
                                             pop_stack,
                                             operand=None)
            return new_curr, new_toVisit, stack, tCurr, inter, nodesVis

        return lax.cond(hitAABB, if_hit, if_miss, operand=None)

    final_state = lax.while_loop(cond_fn, body_fn, state)
    # final_state is (curr, toVisit, stack, tBest, intersection, nodesVisited)
    return final_state[4]


def unoccluded(isec_p: jnp.ndarray,
               isec_n: jnp.ndarray,
               target_p: jnp.ndarray,
               primitives: dict,
               bvh: dict,
               shadow_epsilon: float = 0.0001) -> bool:
    direction = jax.nn.normalize(target_p - isec_p)
    distance = jnp.linalg.norm(target_p - isec_p) * (1 - shadow_epsilon)
    ray = spawn_ray(isec_p, isec_n, direction)
    intersection = intersect_bvh(ray, primitives, bvh, 0, distance)
    return intersection.intersected == 1

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
