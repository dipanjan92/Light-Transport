import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Sequence, Dict, Tuple

from primitives.aabb import AABB  # Assumed implemented with JAX (using jnp)


# -----------------------------------------------------------------------------
# BVH Primitive & Node Data Structures
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BVHPrimitive:
    prim: Any  # The actual primitive; must be a JAX-compatible pytree or immutable object.
    prim_num: int
    bounds: AABB

@dataclass(frozen=True)
class BucketInfo:
    count: int
    bounds: AABB


@dataclass(frozen=True)
class BuildParams:
    n_triangles: int
    n_ordered_prims: int
    total_nodes: int
    split_method: int


@dataclass(frozen=True)
class MortonPrimitive:
    prim_ix: int
    morton_code: int


@dataclass(frozen=True)
class LBVHTreelet:
    start_ix: int
    n_primitives: int
    build_nodes: Dict  # For example, a dict of BVHNodes


# In JAX we represent a BVH node (used during LBVH build) as:
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

@dataclass(frozen=True)
class LinearBVHNode:
    bounds: AABB
    primitives_offset: int
    second_child_offset: int
    n_primitives: int
    axis: int


# -----------------------------------------------------------------------------
# Partitioning Helpers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PartitionWrapper:
    n_buckets: int
    centroid_bounds: AABB
    dim: int
    min_cost_split_bucket: int

    def partition_pred(self, x: BVHNode) -> bool:
        # Compute the center of x.bounds along the chosen dimension.
        centroid = (x.bounds.min_point[self.dim] + x.bounds.max_point[self.dim]) * 0.5
        # Normalize and scale to get a bucket index.
        b_val = (centroid - self.centroid_bounds.min_point[self.dim]) / (
                self.centroid_bounds.max_point[self.dim] - self.centroid_bounds.min_point[self.dim])
        b_int = int(self.n_buckets * b_val)
        # If b_int equals n_buckets, adjust it using jax.lax.cond.
        b_int = jax.lax.cond(b_int == self.n_buckets,
                             lambda _: self.n_buckets - 1,
                             lambda _: b_int,
                             operand=None)
        return b_int <= self.min_cost_split_bucket


@dataclass(frozen=True)
class IntervalWrapper:
    morton_prims: Tuple[MortonPrimitive, ...]
    mask: int

    def interval_pred(self, i: int) -> bool:
        return ((self.morton_prims[0].morton_code & self.mask) ==
                (self.morton_prims[i].morton_code & self.mask))


# -----------------------------------------------------------------------------
# BVH Primitive Initialization
# -----------------------------------------------------------------------------
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



# -----------------------------------------------------------------------------
# Ordered Indices Helpers
# -----------------------------------------------------------------------------
def update_ordered_indices(ordered_indices: jnp.ndarray, idx: int, prim_idx: int) -> jnp.ndarray:
    return ordered_indices.at[idx].set(prim_idx)

def get_ordered_primitives(primitives: Dict[str, jnp.ndarray],
                           ordered_indices: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    return {key: primitives[key][ordered_indices] for key in primitives}

# -----------------------------------------------------------------------------
# Helper to update nodes stored in a tuple
# -----------------------------------------------------------------------------
def update_nodes(nodes: Tuple[BVHNode, ...], idx: int, new_node: BVHNode) -> Tuple[BVHNode, ...]:
    nodes_list = list(nodes)
    nodes_list[idx] = new_node
    return tuple(nodes_list)