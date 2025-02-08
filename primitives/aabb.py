import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import NamedTuple

from jax import tree_util

from primitives.ray import Ray

# Define a constant for infinity.
INF = jnp.inf


@dataclass(frozen=True)
class AABB:
    min_point: jnp.ndarray  # shape (3,)
    max_point: jnp.ndarray  # shape (3,)
    centroid: jnp.ndarray   # shape (3,)

def update_centroid(aabb: AABB) -> AABB:
    new_centroid = (aabb.min_point + aabb.max_point) * 0.5
    return AABB(aabb.min_point, aabb.max_point, new_centroid)

def union(aabb1: AABB, aabb2: AABB) -> AABB:
    new_min = jnp.minimum(aabb1.min_point, aabb2.min_point)
    new_max = jnp.maximum(aabb1.max_point, aabb2.max_point)
    return update_centroid(AABB(new_min, new_max, (new_min + new_max) * 0.5))

def union_p(aabb: AABB, point: jnp.ndarray) -> AABB:
    new_min = jnp.minimum(aabb.min_point, point)
    new_max = jnp.maximum(aabb.max_point, point)
    return update_centroid(AABB(new_min, new_max, (new_min + new_max) * 0.5))

def get_surface_area(aabb: AABB) -> float:
    diag = aabb.max_point - aabb.min_point
    return 2 * (diag[0]*diag[1] + diag[0]*diag[2] + diag[1]*diag[2])

def get_largest_dim(aabb: AABB) -> jnp.ndarray:
    # Compute extents along each axis.
    dx = jnp.abs(aabb.max_point[0] - aabb.min_point[0])
    dy = jnp.abs(aabb.max_point[1] - aabb.min_point[1])
    dz = jnp.abs(aabb.max_point[2] - aabb.min_point[2])
    # Return a JAX integer array (scalar) computed via jnp.where.
    return jnp.where((dx > dy) & (dx > dz), 0,
                     jnp.where(dy > dz, 1, 2))


def aabb_intersect(aabb: AABB, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray) -> bool:
    """Check for ray-AABB intersection.

    This is a functional equivalent to the Taichi version.
    """
    # Set initial t_min and t_max.
    t_min = 0.0
    t_max = INF
    # Compute the inverse of the ray direction.
    ray_inv_dir = 1.0 / ray_direction

    # Loop over the three axes.
    # (Here we “unroll” the loop manually.)
    for i in range(3):
        t1 = (aabb.min_point[i] - ray_origin[i]) * ray_inv_dir[i]
        t2 = (aabb.max_point[i] - ray_origin[i]) * ray_inv_dir[i]
        # Update t_min and t_max.
        t_min = jnp.minimum(jnp.maximum(t1, t_min), jnp.maximum(t2, t_min))
        t_max = jnp.maximum(jnp.minimum(t1, t_max), jnp.minimum(t2, t_max))
    return t_min <= t_max


def offset(aabb: AABB, point: jnp.ndarray) -> jnp.ndarray:
    """Return the offset of a point relative to the AABB (normalized by the box size)."""
    o = point - aabb.min_point
    diff = aabb.max_point - aabb.min_point
    # To avoid division by zero, we use jnp.where.
    o = jnp.where(diff > 0, o / diff, o)
    return o

def is_empty_box(aabb: AABB) -> bool:
    """
    Return True if the box is “empty.”
    (For example, if the minimum is INF and maximum is -INF.)
    """
    # In our case we define an empty box as one that never got updated.
    return (aabb.min_point[0] == INF) and (aabb.max_point[0] == -INF) and (aabb.min_point[0] > aabb.max_point[0])


def enclose_centroids(a: AABB, centroid: jnp.ndarray) -> AABB:
    # If a is None, return an AABB that is a degenerate box at centroid.
    if a is None:
        return AABB(centroid, centroid, centroid)
    new_min = jnp.minimum(a.min_point, centroid)
    new_max = jnp.maximum(a.max_point, centroid)
    return AABB(new_min, new_max, (new_min + new_max) / 2.0)


def contains(aabb1: AABB, aabb2: AABB) -> bool:
    """Return True if aabb1 fully contains aabb2."""
    return ((aabb1.min_point[0] <= aabb2.min_point[0]) and
            (aabb1.min_point[1] <= aabb2.min_point[1]) and
            (aabb1.min_point[2] <= aabb2.min_point[2]) and
            (aabb1.max_point[0] >= aabb2.max_point[0]) and
            (aabb1.max_point[1] >= aabb2.max_point[1]) and
            (aabb1.max_point[2] >= aabb2.max_point[2]))


def equal_bounds(aabb: AABB) -> bool:
    """Return True if the AABB has zero volume (min equals max in all dimensions)."""
    return jnp.all(aabb.max_point == aabb.min_point)


# -------------------------------------
# Ray and intersect_bounds
# -------------------------------------


def intersect_bounds(aabb: AABB, ray: Ray, inv_dir: jnp.ndarray) -> bool:
    """
    Return True if the ray intersects the AABB.
    This version uses jnp.where to compute tmin and tmax per axis and then returns (tmin <= tmax).
    Note: The returned value is a traced JAX array (of shape bool[]).
    """
    tmin_x = jnp.where(inv_dir[0] < 0,
                       (aabb.max_point[0] - ray.origin[0]) * inv_dir[0],
                       (aabb.min_point[0] - ray.origin[0]) * inv_dir[0])
    tmax_x = jnp.where(inv_dir[0] < 0,
                       (aabb.min_point[0] - ray.origin[0]) * inv_dir[0],
                       (aabb.max_point[0] - ray.origin[0]) * inv_dir[0])
    tmin_y = jnp.where(inv_dir[1] < 0,
                       (aabb.max_point[1] - ray.origin[1]) * inv_dir[1],
                       (aabb.min_point[1] - ray.origin[1]) * inv_dir[1])
    tmax_y = jnp.where(inv_dir[1] < 0,
                       (aabb.min_point[1] - ray.origin[1]) * inv_dir[1],
                       (aabb.max_point[1] - ray.origin[1]) * inv_dir[1])
    tmin_z = jnp.where(inv_dir[2] < 0,
                       (aabb.max_point[2] - ray.origin[2]) * inv_dir[2],
                       (aabb.min_point[2] - ray.origin[2]) * inv_dir[2])
    tmax_z = jnp.where(inv_dir[2] < 0,
                       (aabb.min_point[2] - ray.origin[2]) * inv_dir[2],
                       (aabb.max_point[2] - ray.origin[2]) * inv_dir[2])
    tmin = jnp.maximum(jnp.maximum(tmin_x, tmin_y), tmin_z)
    tmax = jnp.minimum(jnp.minimum(tmax_x, tmax_y), tmax_z)
    return tmin <= tmax

def _aabb_flatten(aabb: AABB):
    # Children that are JAX arrays
    children = (aabb.min_point, aabb.max_point, aabb.centroid)
    aux = None
    return children, aux

def _aabb_unflatten(aux, children):
    return AABB(*children)

tree_util.register_pytree_node(AABB, _aabb_flatten, _aabb_unflatten)