import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from primitives.aabb import AABB, union, union_p, intersect_bounds, update_centroid, get_surface_area, get_largest_dim
from primitives.ray import spawn_ray
from primitives.triangle import triangle_intersect
from utils.stdlib import push, pop

INF = 1e10
MAX_DEPTH = 64  # Maximum stack depth for BVH traversal

# -----------------------------------------------------------------------------
# Register custom types as pytrees so that JAX can trace them.
# -----------------------------------------------------------------------------
from jax import tree_util

@dataclass(frozen=True)
class BVHNode:
    bounds: AABB
    first_prim_offset: int = -1
    n_primitives: int = 0
    child_0: int = -1
    child_1: int = -1
    split_axis: int = -1

    def init_leaf(self, first: int, n: int, box: AABB) -> "BVHNode":
        return BVHNode(bounds=box, first_prim_offset=first, n_primitives=n)

    def init_interior(self, axis: int, c0: int, c1: int, box: AABB) -> "BVHNode":
        return BVHNode(bounds=box, child_0=c0, child_1=c1, split_axis=axis)

def _bvhnode_flatten(node: BVHNode):
    children = (node.bounds, node.first_prim_offset, node.n_primitives,
                node.child_0, node.child_1, node.split_axis)
    return children, None

def _bvhnode_unflatten(aux, children):
    return BVHNode(children[0], children[1], children[2],
                   children[3], children[4], children[5])

tree_util.register_pytree_node(BVHNode, _bvhnode_flatten, _bvhnode_unflatten)

@dataclass(frozen=True)
class LinearBVHNode:
    bounds: AABB
    primitives_offset: int
    second_child_offset: int
    n_primitives: int
    axis: int

def _lin_bvhnode_flatten(node: LinearBVHNode):
    children = (node.bounds, node.primitives_offset,
                node.second_child_offset, node.n_primitives, node.axis)
    return children, None

def _lin_bvhnode_unflatten(aux, children):
    return LinearBVHNode(children[0], children[1], children[2], children[3], children[4])

tree_util.register_pytree_node(LinearBVHNode, _lin_bvhnode_flatten, _lin_bvhnode_unflatten)

# -----------------------------------------------------------------------------
# Ordered Indices Helpers
# -----------------------------------------------------------------------------
def update_ordered_indices(ordered_indices: jnp.ndarray, idx: int, prim_idx: int) -> jnp.ndarray:
    return ordered_indices.at[idx].set(prim_idx)

def get_ordered_primitives(primitives: Dict[str, jnp.ndarray],
                           ordered_indices: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    return {key: primitives[key][ordered_indices] for key in primitives}

# -----------------------------------------------------------------------------
# Helper to update the nodes dictionary.
# We assume state["nodes"] is a dictionary of JAX arrays.
# -----------------------------------------------------------------------------
def update_nodes_dict(nodes: Dict[str, jnp.ndarray], idx: int, new_node: BVHNode) -> Dict[str, jnp.ndarray]:
    idx_static = jax.lax.stop_gradient(idx)
    new_nodes = {}
    new_nodes["bounds_min"] = jax.lax.dynamic_update_index_in_dim(
        nodes["bounds_min"], new_node.bounds.min_point[None, :], idx_static, axis=0)
    new_nodes["bounds_max"] = jax.lax.dynamic_update_index_in_dim(
        nodes["bounds_max"], new_node.bounds.max_point[None, :], idx_static, axis=0)
    new_nodes["bounds_centroid"] = jax.lax.dynamic_update_index_in_dim(
        nodes["bounds_centroid"], new_node.bounds.centroid[None, :], idx_static, axis=0)
    new_nodes["first_prim_offset"] = jax.lax.dynamic_update_index_in_dim(
        nodes["first_prim_offset"], jnp.array([new_node.first_prim_offset], dtype=jnp.int32), idx_static, axis=0)
    new_nodes["n_primitives"] = jax.lax.dynamic_update_index_in_dim(
        nodes["n_primitives"], jnp.array([new_node.n_primitives], dtype=jnp.int32), idx_static, axis=0)
    new_nodes["child_0"] = jax.lax.dynamic_update_index_in_dim(
        nodes["child_0"], jnp.array([new_node.child_0], dtype=jnp.int32), idx_static, axis=0)
    new_nodes["child_1"] = jax.lax.dynamic_update_index_in_dim(
        nodes["child_1"], jnp.array([new_node.child_1], dtype=jnp.int32), idx_static, axis=0)
    new_nodes["split_axis"] = jax.lax.dynamic_update_index_in_dim(
        nodes["split_axis"], jnp.array([new_node.split_axis], dtype=jnp.int32), idx_static, axis=0)
    for key in nodes:
        if key not in new_nodes:
            new_nodes[key] = nodes[key]
    return new_nodes

# -----------------------------------------------------------------------------
# Helper: Update parent's child pointer using dynamic indexing.
# -----------------------------------------------------------------------------
def update_parent(state: Dict[str, Any],
                  parent_idx: int,
                  current_node_idx: int,
                  is_second_child: int) -> Dict[str, Any]:
    parent_idx_static = jax.lax.stop_gradient(parent_idx)
    parent = {
        key: jax.lax.dynamic_slice_in_dim(state["nodes"][key],
                                          parent_idx_static,
                                          1,
                                          axis=0)
        for key in state["nodes"]
    }
    # Here current_node_idx is a concrete integer.
    current_idx_scalar = current_node_idx
    # For a 1D array, use single-index indexing.
    child1_scalar = state["nodes"]["child_1"][parent_idx_static]
    pred = jnp.equal(current_idx_scalar, child1_scalar)
    updated_parent = jax.lax.cond(
        pred,
        lambda _: {
            "bounds_min": parent["bounds_min"],
            "bounds_max": parent["bounds_max"],
            "bounds_centroid": parent["bounds_centroid"],
            "first_prim_offset": parent["first_prim_offset"],
            "n_primitives": parent["n_primitives"],
            "child_0": parent["child_0"],
            "child_1": jnp.array([current_node_idx], dtype=jnp.int32),
            "split_axis": parent["split_axis"]
        },
        lambda _: {
            "bounds_min": parent["bounds_min"],
            "bounds_max": parent["bounds_max"],
            "bounds_centroid": parent["bounds_centroid"],
            "first_prim_offset": parent["first_prim_offset"],
            "n_primitives": parent["n_primitives"],
            "child_0": jnp.array([current_node_idx], dtype=jnp.int32),
            "child_1": parent["child_1"],
            "split_axis": parent["split_axis"]
        },
        operand=None
    )
    new_nodes = {}
    for key in state["nodes"]:
        new_nodes[key] = jax.lax.dynamic_update_index_in_dim(
            state["nodes"][key],
            updated_parent[key],
            parent_idx_static,
            axis=0
        )
    return {**state, "nodes": new_nodes}

# -----------------------------------------------------------------------------
# Partitioning Functions and Wrappers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PartitionWrapper:
    n_buckets: int
    centroid_bounds: AABB
    dim: int
    min_cost_split_bucket: int

    def partition_pred(self, x: BVHNode) -> bool:
        centroid = (x.bounds.min_point[self.dim] + x.bounds.max_point[self.dim]) * 0.5
        b_val = (centroid - self.centroid_bounds.min_point[self.dim]) / (
            self.centroid_bounds.max_point[self.dim] - self.centroid_bounds.min_point[self.dim])
        b_int = jnp.array(self.n_buckets * b_val, dtype=jnp.int32)
        b_int = jax.lax.cond(jnp.equal(b_int, self.n_buckets),
                             lambda _: jnp.array(self.n_buckets - 1, dtype=jnp.int32),
                             lambda _: b_int,
                             operand=None)
        return b_int <= self.min_cost_split_bucket

@dataclass(frozen=True)
class IntervalWrapper:
    morton_prims: Tuple[Any, ...]
    mask: int

    def interval_pred(self, i: int) -> bool:
        return ((self.morton_prims[0].morton_code & self.mask) ==
                (self.morton_prims[i].morton_code & self.mask))

def partition_equal_counts(bvh_primitives: Dict[str, Any],
                           start: int, end: int, dim: int) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    prim_centroids = bvh_primitives["bounds_centroid"]
    indices = jnp.arange(start, end)
    def body_fun(i, val):
        inds = val
        current = inds[i]
        sub = inds[i:]
        j = i + jnp.array(jax.lax.stop_gradient(jnp.argmin(prim_centroids[sub, dim])), dtype=jnp.int32)
        inds = inds.at[i].set(inds[j])
        inds = inds.at[j].set(current)
        return inds
    indices = jax.lax.fori_loop(start, end, body_fun, indices)
    mid = jnp.floor_divide(start + end, 2)
    return mid, bvh_primitives

def partition_middle(bvh_primitives: Dict[str, Any],
                     start: int, end: int, dim: int,
                     centroid_bounds: AABB) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    pmid = (centroid_bounds.min_point[dim] + centroid_bounds.max_point[dim]) / 2.0
    n = end - start
    def union_centroid_loop(i, cb):
        idx = start + i
        centroid = bvh_primitives["bounds_centroid"][idx]
        return union_p(cb, centroid)
    centroid_bounds = jax.lax.fori_loop(0, n, union_centroid_loop, centroid_bounds)
    return jnp.array(pmid, dtype=jnp.int32), bvh_primitives

def partition_sah(bvh_primitives: Dict[str, Any],
                  start: int, end: int, dim: int,
                  centroid_bounds: AABB, costs: jnp.ndarray,
                  buckets: Dict[str, Any], bounds: AABB,
                  max_prims_in_node: int) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    nBuckets = 12
    nSplits = nBuckets - 1
    for b in range(nBuckets):
        buckets["count"] = buckets["count"].at[b].set(0)
        buckets["bounds_min"] = buckets["bounds_min"].at[b].set(jnp.array([INF, INF, INF]))
        buckets["bounds_max"] = buckets["bounds_max"].at[b].set(jnp.array([-INF, -INF, -INF]))
    n = end - start
    def bucket_loop(i, buck):
        idx = start + i
        centroid = bvh_primitives["bounds_centroid"][idx]
        bucket_index = jnp.array(nBuckets * (centroid[dim] - centroid_bounds.min_point[dim]) /
                                   (centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim]),
                                   dtype=jnp.int32)
        bucket_index = jnp.minimum(bucket_index, nBuckets - 1)
        buck["count"] = buck["count"].at[bucket_index].set(buck["count"][bucket_index] + 1)
        prim_bounds = AABB(
            bvh_primitives["bounds_min"][idx],
            bvh_primitives["bounds_max"][idx],
            bvh_primitives["bounds_centroid"][idx]
        )
        curr_bounds = AABB(buck["bounds_min"][bucket_index],
                           buck["bounds_max"][bucket_index],
                           (buck["bounds_min"][bucket_index] + buck["bounds_max"][bucket_index]) / 2.0)
        new_bounds = union(curr_bounds, prim_bounds)
        buck["bounds_min"] = buck["bounds_min"].at[bucket_index].set(new_bounds.min_point)
        buck["bounds_max"] = buck["bounds_max"].at[bucket_index].set(new_bounds.max_point)
        return buck
    buckets = jax.lax.fori_loop(0, n, bucket_loop, buckets)
    def cost_loop(j, vals):
        countBelow, bBelow, costs_ = vals
        curr_bounds = AABB(
            buckets["bounds_min"][j],
            buckets["bounds_max"][j],
            (buckets["bounds_min"][j] + buckets["bounds_max"][j]) / 2.0
        )
        bBelow = union(bBelow, curr_bounds)
        countBelow += jnp.array(buckets["count"][j], dtype=jnp.int32)
        costs_ = costs_.at[j].set(countBelow * get_surface_area(bBelow))
        return (countBelow, bBelow, costs_)
    bBelow_init = AABB(jnp.array([INF, INF, INF]),
                       jnp.array([-INF, -INF, -INF]),
                       jnp.array([INF, INF, INF]))
    _, _, costs = jax.lax.fori_loop(0, nSplits, cost_loop, (0, bBelow_init, costs))
    return jnp.array(start, dtype=jnp.int32), bvh_primitives

# -----------------------------------------------------------------------------
# BVH Build, Flatten, and Intersection (High-level Routines)
# -----------------------------------------------------------------------------
def build_bvh(primitives: Dict[str, jnp.ndarray],
              bvh_primitives: Dict[str, jnp.ndarray],
              ordered_indices: jnp.ndarray,
              split_method: int) -> Dict[str, Any]:
    """
    Build the BVH hierarchically. The state includes:
      - an index array (ordered_indices) that will hold the final order of primitives
      - a dictionary "nodes" of BVH node arrays.
    """
    n_boxes = bvh_primitives["prim_num"].shape[0]
    max_prims_in_node = jnp.maximum(4, int(0.1 * n_boxes))
    nodes = {
        "bounds_min": jnp.full((n_boxes, 3), INF, dtype=jnp.float32),
        "bounds_max": jnp.full((n_boxes, 3), -INF, dtype=jnp.float32),
        "bounds_centroid": jnp.full((n_boxes, 3), INF, dtype=jnp.float32),
        "first_prim_offset": -jnp.ones((n_boxes,), dtype=jnp.int32),
        "n_primitives": jnp.zeros((n_boxes,), dtype=jnp.int32),
        "child_0": -jnp.ones((n_boxes,), dtype=jnp.int32),
        "child_1": -jnp.ones((n_boxes,), dtype=jnp.int32),
        "split_axis": -jnp.ones((n_boxes,), dtype=jnp.int32)
    }
    state = {
        "stack": jnp.zeros((n_boxes, 4), dtype=jnp.int32),
        "stack_ptr": 0,
        "total_nodes": 0,
        "nodes": nodes,
        "ordered_indices": ordered_indices,
        "ordered_index_offset": 0,
        "primitives": primitives,
        "bvh_primitives": bvh_primitives,
        "split_method": split_method,
        "costs": jnp.zeros((12 - 1,), dtype=jnp.float32),
        "buckets": {
            "count": jnp.zeros((12,), dtype=jnp.int32),
            "bounds_min": jnp.tile(jnp.array([INF, INF, INF]), (12, 1)),
            "bounds_max": jnp.tile(jnp.array([-INF, -INF, -INF]), (12, 1))
        }
    }
    state = push(state, jnp.array([0, n_boxes, -1, 0], dtype=jnp.int32))

    def cond_fn(s):
        return s["stack_ptr"] > 0

    def body_fn(s):
        (start, end, parent_idx, is_second_child), s = pop(s)
        current_node_idx = jax.lax.stop_gradient(s["total_nodes"])
        s = {**s, "total_nodes": s["total_nodes"] + 1}
        s = jax.lax.cond(
            parent_idx != -1,
            lambda s_: update_parent(s_, parent_idx, current_node_idx, is_second_child),
            lambda s_: s_,
            operand=s
        )
        # Compute union bounds over [start, end) using a fori_loop.
        init_bounds = AABB(jnp.array([INF, INF, INF]),
                           jnp.array([-INF, -INF, -INF]),
                           jnp.array([INF, INF, INF]))
        n = end - start
        def union_loop(i, b):
            idx = start + i
            prim_bounds = AABB(
                s["bvh_primitives"]["bounds_min"][idx],
                s["bvh_primitives"]["bounds_max"][idx],
                s["bvh_primitives"]["bounds_centroid"][idx]
            )
            return union(b, prim_bounds)
        bounds = jax.lax.fori_loop(0, n, union_loop, init_bounds)
        # Leaf branch: update ordered indices and create a new leaf node.
        def leaf_branch(s_):
            first_index_offset = s_["ordered_index_offset"]
            def update_order_loop(i, oi):
                return update_ordered_indices(
                    oi,
                    first_index_offset + i,
                    s_["bvh_primitives"]["prim_num"][start + i]
                )
            new_ordered_indices = jax.lax.fori_loop(0, n, update_order_loop, s_["ordered_indices"])
            s_["ordered_indices"] = new_ordered_indices
            new_leaf_node = BVHNode(
                bounds=AABB(bounds.min_point, bounds.max_point, bounds.centroid),
                first_prim_offset=first_index_offset,
                n_primitives=end - start,
                child_0=-1,
                child_1=-1,
                split_axis=-1
            )
            s_["nodes"] = update_nodes_dict(s_["nodes"], current_node_idx, new_leaf_node)
            s_["ordered_index_offset"] += (end - start)
            return s_
        # Interior branch (using a simple midpoint partition).
        def interior_branch(s_):
            init_cb = AABB(jnp.array([INF, INF, INF]),
                           jnp.array([-INF, -INF, -INF]),
                           jnp.array([INF, INF, INF]))
            def union_centroid_loop(i, cb):
                idx = start + i
                centroid = s_["bvh_primitives"]["bounds_centroid"][idx]
                return union_p(cb, centroid)
            centroid_bounds = jax.lax.fori_loop(0, n, union_centroid_loop, init_cb)
            # Get the dimension of maximum extent.
            dim = get_largest_dim(centroid_bounds)
            # Use jnp.take for dynamic indexing and then use switch.
            cond_pred = jnp.equal(jnp.take(centroid_bounds.max_point, dim),
                                  jnp.take(centroid_bounds.min_point, dim))
            branch_index = jnp.where(cond_pred, 0, 1)
            s_ = jax.lax.switch(branch_index, [leaf_branch, lambda s: s], operand=s_)
            mid = start + jnp.floor_divide(n, 2)
            new_interior_node = BVHNode(
                bounds=AABB(bounds.min_point, bounds.max_point, bounds.centroid),
                first_prim_offset=-1,
                n_primitives=0,
                child_0=-1,
                child_1=-1,
                split_axis=dim
            )
            s_["nodes"] = update_nodes_dict(s_["nodes"], current_node_idx, new_interior_node)
            s_ = push(s_, jnp.array([mid, end, current_node_idx, 1], dtype=jnp.int32))
            s_ = push(s_, jnp.array([start, mid, current_node_idx, 0], dtype=jnp.int32))
            return s_
        leaf_cond = jnp.logical_or(jnp.equal(get_surface_area(bounds), 0.0),
                                   jnp.equal(end - start, 1))
        s = jax.lax.cond(leaf_cond, leaf_branch, interior_branch, operand=s)
        return s

    state = jax.lax.while_loop(cond_fn, body_fn, state)
    return state

def flatten_bvh(nodes: Dict[str, Any]) -> Tuple[Dict[int, LinearBVHNode], int]:
    # Plain Python loops are acceptable here.
    linear_bvh = {}
    stack = [(0, -1, 0)]
    offset = 0
    while stack:
        node_idx, parent_idx, is_second_child = stack.pop()
        if node_idx == -1:
            continue
        current_idx = offset
        node = {key: nodes[key][node_idx] for key in nodes}
        if node["n_primitives"] > 0:
            lnode = LinearBVHNode(
                bounds=AABB(node["bounds_min"], node["bounds_max"], node["bounds_centroid"]),
                primitives_offset=node["first_prim_offset"],
                second_child_offset=-1,
                n_primitives=node["n_primitives"],
                axis=-1
            )
        else:
            lnode = LinearBVHNode(
                bounds=AABB(node["bounds_min"], node["bounds_max"], node["bounds_centroid"]),
                primitives_offset=0,
                second_child_offset=-1,
                n_primitives=0,
                axis=node["split_axis"]
            )
        linear_bvh[current_idx] = lnode
        offset += 1
        if parent_idx != -1 and is_second_child:
            parent = linear_bvh[parent_idx]
            parent = LinearBVHNode(
                bounds=parent.bounds,
                primitives_offset=parent.primitives_offset,
                second_child_offset=current_idx,
                n_primitives=parent.n_primitives,
                axis=parent.axis
            )
            linear_bvh[parent_idx] = parent
        if node["n_primitives"] == 0:
            if node["child_1"] != -1:
                stack.append((node["child_1"], current_idx, 1))
            if node["child_0"] != -1:
                stack.append((node["child_0"], current_idx, 0))
    return linear_bvh, offset

def intersect_bvh(ray: Any,
                  primitives: Dict[str, jnp.ndarray],
                  linear_bvh: Dict[int, LinearBVHNode],
                  t_min: float = 0.0,
                  t_max: float = INF) -> Any:
    nodes_to_visit = [0] * MAX_DEPTH
    to_visit_offset = 0
    current_node_index = 0  # using a plain Python int here
    tMax = t_max

    inv_dir = 1.0 / ray.direction
    dir_is_neg = (inv_dir < 0)
    intersection = None

    while True:
        if current_node_index == -1:
            break

        node = linear_bvh[current_node_index]

        if intersect_bounds(node.bounds, ray, inv_dir):
            if node.n_primitives > 0:
                for i in range(node.n_primitives):
                    # Get the primitive index (using .item() to convert from a JAX scalar)
                    prim_index = node.primitives_offset.item() + i
                    # Extract the triangle vertices from the primitives dictionary.
                    v0 = primitives["vertex_1"][prim_index]
                    v1 = primitives["vertex_2"][prim_index]
                    v2 = primitives["vertex_3"][prim_index]
                    hit, t_candidate = triangle_intersect(ray.origin, ray.direction, v0, v1, v2, tMax)
                    if hit and (t_min < t_candidate) and (t_candidate < tMax):
                        tMax = t_candidate
                        intersection = (ray, prim_index, t_candidate)
                if to_visit_offset == 0:
                    break
                to_visit_offset -= 1
                current_node_index = int(nodes_to_visit[to_visit_offset])
            else:
                if dir_is_neg[node.axis]:
                    if to_visit_offset < MAX_DEPTH:
                        nodes_to_visit[to_visit_offset] = current_node_index + 1
                        to_visit_offset += 1
                    current_node_index = int(node.second_child_offset)
                else:
                    if to_visit_offset < MAX_DEPTH:
                        nodes_to_visit[to_visit_offset] = int(node.second_child_offset)
                        to_visit_offset += 1
                    current_node_index = current_node_index + 1
        else:
            if to_visit_offset == 0:
                break
            to_visit_offset -= 1
            current_node_index = int(nodes_to_visit[to_visit_offset])
    return intersection


def unoccluded(isec_p: jnp.ndarray,
               isec_n: jnp.ndarray,
               target_p: jnp.ndarray,
               primitives: Dict[str, Any],
               bvh: Dict[int, LinearBVHNode],
               shadow_epsilon: float = 0.0001) -> bool:
    direction = jax.nn.normalize(target_p - isec_p)
    distance = jnp.linalg.norm(target_p - isec_p) * (1.0 - shadow_epsilon)
    ray = spawn_ray(isec_p, isec_n, direction)
    intersection = intersect_bvh(ray, primitives, bvh, 0.0, distance)
    return intersection is None

def swap_bvh_primitives(bvh_primitives: Dict[str, Any],
                        i: int, j: int) -> Dict[str, Any]:
    new_bp = {}
    for key, arr in bvh_primitives.items():
        temp_i = arr[i]
        temp_j = arr[j]
        arr = arr.at[i].set(temp_j)
        arr = arr.at[j].set(temp_i)
        new_bp[key] = arr
    return new_bp

def init_bvh_primitives(triangle_arrays: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    v1 = triangle_arrays["vertex_1"]
    v2 = triangle_arrays["vertex_2"]
    v3 = triangle_arrays["vertex_3"]

    bounds_min = jnp.minimum(jnp.minimum(v1, v2), v3)
    bounds_max = jnp.maximum(jnp.maximum(v1, v2), v3)
    bounds_centroid = (bounds_min + bounds_max) / 2.0

    n_primitives = v1.shape[0]
    prim_num = jnp.arange(n_primitives)

    return {
        "prim": triangle_arrays,
        "prim_num": prim_num,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "bounds_centroid": bounds_centroid,
    }
