# aabb.py

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Tuple
from jax import tree_util
from primitives.ray import Ray  # Ensure your Ray type is defined as a PyTree

INF = jnp.inf

@dataclass(frozen=True)
class AABB:
    min_point: jnp.ndarray  # shape (3,)
    max_point: jnp.ndarray  # shape (3,)
    centroid: jnp.ndarray   # shape (3,)

    # --- Instance methods (to be called like bounds.get_surface_area()) ---

    def update_centroid(self) -> "AABB":
        new_centroid = (self.min_point + self.max_point) * 0.5
        return AABB(self.min_point, self.max_point, new_centroid)

    def union(self, other: "AABB") -> "AABB":
        new_min = jnp.minimum(self.min_point, other.min_point)
        new_max = jnp.maximum(self.max_point, other.max_point)
        return AABB(new_min, new_max, (new_min + new_max) * 0.5)

    def union_point(self, point: jnp.ndarray) -> "AABB":
        new_min = jnp.minimum(self.min_point, point)
        new_max = jnp.maximum(self.max_point, point)
        return AABB(new_min, new_max, (new_min + new_max) * 0.5)

    def get_surface_area(self) -> float:
        diag = self.max_point - self.min_point
        return 2.0 * (diag[0] * diag[1] + diag[0] * diag[2] + diag[1] * diag[2])

    def get_largest_dim(self) -> int:
        extents = self.max_point - self.min_point
        return int(jnp.argmax(extents))

    def aabb_intersect(self, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray) -> bool:
        ray_inv_dir = 1.0 / ray_direction
        t1 = (self.min_point - ray_origin) * ray_inv_dir
        t2 = (self.max_point - ray_origin) * ray_inv_dir
        tmin = jnp.max(jnp.minimum(t1, t2))
        tmax = jnp.min(jnp.maximum(t1, t2))
        return tmin <= tmax

    def offset(self, point: jnp.ndarray) -> jnp.ndarray:
        o = point - self.min_point
        diff = self.max_point - self.min_point
        normalized = jnp.where(diff > 0, o / diff, o)
        return normalized

    def is_empty_box(self) -> bool:
        return bool(jnp.any(self.min_point > self.max_point))

    def contains(self, other: "AABB") -> bool:
        cond_min = jnp.all(self.min_point <= other.min_point)
        cond_max = jnp.all(self.max_point >= other.max_point)
        return bool(cond_min & cond_max)

    def equal_bounds(self) -> bool:
        return bool(jnp.all(self.max_point == self.min_point))


# --- Standalone functions (functional style with JIT) ---
@jax.jit
def update_centroid(aabb: AABB) -> AABB:
    new_centroid = (aabb.min_point + aabb.max_point) * 0.5
    return AABB(aabb.min_point, aabb.max_point, new_centroid)

@jax.jit
def union(aabb1: AABB, aabb2: AABB) -> AABB:
    new_min = jnp.minimum(aabb1.min_point, aabb2.min_point)
    new_max = jnp.maximum(aabb1.max_point, aabb2.max_point)
    return AABB(new_min, new_max, (new_min + new_max) * 0.5)

@jax.jit
def union_p(aabb: AABB, point: jnp.ndarray) -> AABB:
    new_min = jnp.minimum(aabb.min_point, point)
    new_max = jnp.maximum(aabb.max_point, point)
    return AABB(new_min, new_max, (new_min + new_max) * 0.5)

@jax.jit
def get_surface_area(aabb: AABB) -> float:
    diag = aabb.max_point - aabb.min_point
    return 2.0 * (diag[0] * diag[1] + diag[0] * diag[2] + diag[1] * diag[2])

@jax.jit
def get_largest_dim(aabb: AABB) -> int:
    extents = aabb.max_point - aabb.min_point
    return int(jnp.argmax(extents))

@jax.jit
def aabb_intersect(aabb: AABB, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray) -> bool:
    ray_inv_dir = 1.0 / ray_direction
    t1 = (aabb.min_point - ray_origin) * ray_inv_dir
    t2 = (aabb.max_point - ray_origin) * ray_inv_dir
    tmin = jnp.max(jnp.minimum(t1, t2))
    tmax = jnp.min(jnp.maximum(t1, t2))
    return (tmin <= tmax)

@jax.jit
def aabb_intersect_p(ray_origin: jnp.ndarray,
                     ray_direction: jnp.ndarray,
                     tMax: float,
                     invDir: jnp.ndarray,
                     dirIsNeg: jnp.ndarray,
                     bounds_min: jnp.ndarray,
                     bounds_max: jnp.ndarray) -> bool:
    # Compute slab intersections.
    tmin = (bounds_min - ray_origin) * invDir
    tmax = (bounds_max - ray_origin) * invDir
    # For each axis, if the ray direction is negative, swap tmin and tmax.
    tmin_final = jnp.where(dirIsNeg, tmax, tmin)
    tmax_final = jnp.where(dirIsNeg, tmin, tmax)
    # The ray enters when the maximum of the tmin values and exits at the minimum
    # of the tmax values.
    t_enter = jnp.maximum(jnp.maximum(tmin_final[0], tmin_final[1]), tmin_final[2])
    t_exit  = jnp.minimum(jnp.minimum(tmax_final[0], tmax_final[1]), tmax_final[2])
    return (t_enter <= t_exit) & (t_enter < tMax)

@jax.jit
def offset(aabb: AABB, point: jnp.ndarray) -> jnp.ndarray:
    o = point - aabb.min_point
    diff = aabb.max_point - aabb.min_point
    normalized = jnp.where(diff > 0, o / diff, o)
    return normalized

@jax.jit
def is_empty_box(aabb: AABB) -> bool:
    return jnp.any(aabb.min_point > aabb.max_point)

def enclose_centroids(a: Any, centroid: jnp.ndarray) -> AABB:
    if a is None:
        return AABB(centroid, centroid, centroid)
    new_min = jnp.minimum(a.min_point, centroid)
    new_max = jnp.maximum(a.max_point, centroid)
    return AABB(new_min, new_max, (new_min + new_max) * 0.5)

@jax.jit
def contains(aabb1: AABB, aabb2: AABB) -> bool:
    cond_min = jnp.all(aabb1.min_point <= aabb2.min_point)
    cond_max = jnp.all(aabb1.max_point >= aabb2.max_point)
    return cond_min & cond_max

@jax.jit
def equal_bounds(aabb: AABB) -> bool:
    return jnp.all(aabb.max_point == aabb.min_point)

@jax.jit
def aabb_hit_distance(bmin: jnp.ndarray,
                      bmax: jnp.ndarray,
                      ray_origin: jnp.ndarray,
                      ray_direction: jnp.ndarray,
                      epsilon: float = 1e-6) -> float:
    """Return the t value at which the ray hits the AABB or INF if no hit."""
    inv_dir = 1.0 / ray_direction
    t1 = (bmin - ray_origin) * inv_dir
    t2 = (bmax - ray_origin) * inv_dir
    tmin = jnp.max(jnp.minimum(t1, t2))
    tmax = jnp.min(jnp.maximum(t1, t2))
    hit = tmin <= tmax
    return jax.lax.select(hit, tmin, INF)

# --- PyTree registration for AABB ---
def _aabb_flatten(aabb: AABB) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
    children = (aabb.min_point, aabb.max_point, aabb.centroid)
    aux = None
    return children, aux

def _aabb_unflatten(aux: None, children: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> AABB:
    return AABB(*children)

tree_util.register_pytree_node(AABB, _aabb_flatten, _aabb_unflatten)
