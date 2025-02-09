# aabb.py
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Tuple

# Import your Ray type as needed. (Ensure that your Ray is a pytree as well.)
from primitives.ray import Ray
from jax import tree_util

# Define a constant for infinity.
INF = jnp.inf


@dataclass(frozen=True)
class AABB:
    min_point: jnp.ndarray  #: shape (3,)
    max_point: jnp.ndarray  #: shape (3,)
    centroid: jnp.ndarray  #: shape (3,)


@jax.jit
def update_centroid(aabb: AABB) -> AABB:
    """Update the centroid of the AABB."""
    new_centroid = (aabb.min_point + aabb.max_point) * 0.5
    # (The third argument is redundant since we recompute the centroid.)
    return AABB(aabb.min_point, aabb.max_point, new_centroid)


@jax.jit
def union(aabb1: AABB, aabb2: AABB) -> AABB:
    """Compute the union of two AABBs."""
    new_min = jnp.minimum(aabb1.min_point, aabb2.min_point)
    new_max = jnp.maximum(aabb1.max_point, aabb2.max_point)
    # Note: We could compute the centroid directly as (new_min+new_max)/2.
    return AABB(new_min, new_max, (new_min + new_max) * 0.5)


@jax.jit
def union_p(aabb: AABB, point: jnp.ndarray) -> AABB:
    """Compute the union of an AABB and a point."""
    new_min = jnp.minimum(aabb.min_point, point)
    new_max = jnp.maximum(aabb.max_point, point)
    return AABB(new_min, new_max, (new_min + new_max) * 0.5)


@jax.jit
def get_surface_area(aabb: AABB) -> float:
    """Compute the surface area of the AABB."""
    diag = aabb.max_point - aabb.min_point
    # Surface area = 2*(xy + xz + yz)
    return 2.0 * (diag[0] * diag[1] + diag[0] * diag[2] + diag[1] * diag[2])


@jax.jit
def get_largest_dim(aabb: AABB) -> int:
    """Return the dimension (0,1,2) with the largest extent."""
    extents = aabb.max_point - aabb.min_point
    return jnp.argmax(extents)


@jax.jit
def aabb_intersect(aabb: AABB, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray) -> bool:
    """
    Check for ray-AABB intersection using the slab method.

    Returns True if the ray intersects the box.
    """
    ray_inv_dir = 1.0 / ray_direction
    t1 = (aabb.min_point - ray_origin) * ray_inv_dir
    t2 = (aabb.max_point - ray_origin) * ray_inv_dir
    tmin = jnp.max(jnp.minimum(t1, t2))
    tmax = jnp.min(jnp.maximum(t1, t2))
    return tmin <= tmax


@jax.jit
def offset(aabb: AABB, point: jnp.ndarray) -> jnp.ndarray:
    """
    Return the offset of a point relative to the AABB, normalized by the box size.

    For dimensions where the box has zero extent the raw difference is returned.
    """
    o = point - aabb.min_point
    diff = aabb.max_point - aabb.min_point
    # Avoid division by zero.
    normalized = jnp.where(diff > 0, o / diff, o)
    return normalized


@jax.jit
def is_empty_box(aabb: AABB) -> bool:
    """
    Return True if the AABB is empty, i.e. if any min > max.
    """
    return jnp.any(aabb.min_point > aabb.max_point)


# Note:
# The following function contains a Python branch on a None value.
# In practice you should avoid calling this function within a jitted context
# with a = None. If needed, you can mark the branch as static.
def enclose_centroids(a: Any, centroid: jnp.ndarray) -> AABB:
    """
    Enclose the given centroid within the current AABB `a`. If `a` is None,
    then create a new AABB with both min and max equal to centroid.
    """
    if a is None:
        return AABB(centroid, centroid, centroid)
    new_min = jnp.minimum(a.min_point, centroid)
    new_max = jnp.maximum(a.max_point, centroid)
    return AABB(new_min, new_max, (new_min + new_max) * 0.5)


@jax.jit
def contains(aabb1: AABB, aabb2: AABB) -> bool:
    """
    Return True if aabb1 fully contains aabb2.
    """
    cond_min = jnp.all(aabb1.min_point <= aabb2.min_point)
    cond_max = jnp.all(aabb1.max_point >= aabb2.max_point)
    return cond_min & cond_max


@jax.jit
def equal_bounds(aabb: AABB) -> bool:
    """
    Return True if the AABB has zero volume (min equals max in all dimensions).
    """
    return jnp.all(aabb.max_point == aabb.min_point)


@jax.jit
def intersect_bounds(aabb: AABB, ray: Ray, inv_dir: jnp.ndarray) -> bool:
    """
    Check for ray-AABB intersection using a variant of the slab method.

    `ray` is assumed to have an attribute `origin` (a jnp.ndarray).
    `inv_dir` should be precomputed as 1.0 / ray.direction.
    """
    t1 = (aabb.min_point - ray.origin) * inv_dir
    t2 = (aabb.max_point - ray.origin) * inv_dir
    tmin = jnp.minimum(t1, t2)
    tmax = jnp.maximum(t1, t2)
    t_enter = jnp.max(tmin)
    t_exit = jnp.min(tmax)
    return t_exit >= jnp.maximum(t_enter, 0.0)


# === PyTree registration ===

def _aabb_flatten(aabb: AABB) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
    """Flatten the AABB for JAX pytree compatibility."""
    children = (aabb.min_point, aabb.max_point, aabb.centroid)
    aux = None
    return children, aux


def _aabb_unflatten(aux: None, children: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> AABB:
    """Unflatten the AABB for JAX pytree compatibility."""
    return AABB(*children)


tree_util.register_pytree_node(AABB, _aabb_flatten, _aabb_unflatten)
