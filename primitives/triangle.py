# triangle.py
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class Triangle(NamedTuple):
    vertex_1: jnp.ndarray  # shape (3,)
    vertex_2: jnp.ndarray  # shape (3,)
    vertex_3: jnp.ndarray  # shape (3,)
    centroid: jnp.ndarray  # shape (3,)
    normal: jnp.ndarray  # shape (3,)
    edge_1: jnp.ndarray  # shape (3,)
    edge_2: jnp.ndarray  # shape (3,)


@jax.jit
def triangle_intersect(ray_origin: jnp.ndarray,
                       ray_direction: jnp.ndarray,
                       v0: jnp.ndarray,
                       v1: jnp.ndarray,
                       v2: jnp.ndarray,
                       t_max: float,
                       epsilon: float = 1e-6) -> Tuple[bool, float]:
    """
    Intersect a ray with a triangle using a watertight method.

    Parameters:
      ray_origin: The origin of the ray (shape: (3,))
      ray_direction: The direction of the ray (shape: (3,))
      v0, v1, v2: The triangle vertices (each of shape: (3,))
      t_max: Maximum allowed intersection distance.
      epsilon: A small value to prevent numerical issues.

    Returns:
      A tuple (hit: bool, t: float) where hit indicates if the ray intersects
      the triangle, and t is the intersection distance (or t_max if no hit).
    """
    # Compute the vectors from the ray origin to the triangle vertices.
    p0 = v0 - ray_origin
    p1 = v1 - ray_origin
    p2 = v2 - ray_origin

    # Choose the projection axis (using a “watertight” method).
    abs_d = jnp.abs(ray_direction)
    kz = jnp.where(abs_d[1] > abs_d[0], 1, 0)
    kz = jnp.where(abs_d[2] > abs_d[kz], 2, kz)
    kx = (kz + 1) % 3
    ky = (kx + 1) % 3

    # Project the ray direction and the vertex vectors.
    d = jnp.array([ray_direction[kx], ray_direction[ky], ray_direction[kz]])
    p0t = jnp.array([p0[kx], p0[ky], p0[kz]])
    p1t = jnp.array([p1[kx], p1[ky], p1[kz]])
    p2t = jnp.array([p2[kx], p2[ky], p2[kz]])

    # If the ray's dominant component (d[2]) is negative, flip the sign of d and p*t.
    d, p0t, p1t, p2t = jax.lax.cond(
        d[2] < 0,
        lambda _: (-d, -p0t, -p1t, -p2t),
        lambda _: (d, p0t, p1t, p2t),
        operand=None
    )

    # Compute shear constants.
    d2 = jnp.where(jnp.abs(d[2]) < epsilon, epsilon, d[2])
    Sx = -d[0] / d2
    Sy = -d[1] / d2

    # Apply shear to the projected vertex coordinates.
    p0t = p0t.at[0].set(p0t[0] + Sx * p0t[2])
    p0t = p0t.at[1].set(p0t[1] + Sy * p0t[2])
    p1t = p1t.at[0].set(p1t[0] + Sx * p1t[2])
    p1t = p1t.at[1].set(p1t[1] + Sy * p1t[2])
    p2t = p2t.at[0].set(p2t[0] + Sx * p2t[2])
    p2t = p2t.at[1].set(p2t[1] + Sy * p2t[2])

    # Compute edge functions.
    e0 = p1t[0] * p2t[1] - p1t[1] * p2t[0]
    e1 = p2t[0] * p0t[1] - p2t[1] * p0t[0]
    e2 = p0t[0] * p1t[1] - p0t[1] * p1t[0]
    inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))

    def no_hit(_):
        return False, t_max

    def potential_hit(_):
        det = e0 + e1 + e2

        def reject_det(_):
            return False, t_max

        def accept_det(_):
            invDet = 1.0 / det
            t_candidate = (e0 * p0t[2] + e1 * p1t[2] + e2 * p2t[2]) * invDet
            hit = (t_candidate > epsilon) & (t_candidate < t_max)

            def hit_true(_):
                return True, t_candidate

            def hit_false(_):
                return False, t_max

            return jax.lax.cond(hit, hit_true, hit_false, operand=None)

        return jax.lax.cond(jnp.abs(det) < epsilon, reject_det, accept_det, operand=None)

    return jax.lax.cond(inside, potential_hit, no_hit, operand=None)
