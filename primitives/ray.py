import jax
import jax.numpy as jnp
from dataclasses import dataclass

from jax import tree_util


@dataclass(frozen=True)
class Ray:
    origin: jnp.ndarray  # Expected shape: (3,)
    direction: jnp.ndarray  # Expected shape: (3,)

    def at(self, t: float) -> jnp.ndarray:
        """Return the point along the ray at distance t."""
        return self.origin + t * self.direction

def offset_ray_origin(p: jnp.ndarray, n: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """
    Compute an offset for the ray origin to avoid self-intersection.

    p: the original point (shape (3,))
    n: the normal at point p (shape (3,))
    w: the direction for which the ray is being spawned (shape (3,))
    """
    epsilon = 1e-4
    base_offset = n * epsilon
    dot_val = jnp.dot(w, n)
    offset = jax.lax.cond(dot_val < 0,
                          lambda _: -base_offset,
                          lambda _: base_offset,
                          operand=None)
    return p + offset

def spawn_ray(p: jnp.ndarray, n: jnp.ndarray, d: jnp.ndarray) -> Ray:
    """
    Spawn a new ray from point p with direction d, offsetting the origin
    along the normal n to avoid self-intersections.
    """
    origin = offset_ray_origin(p, n, d)
    return Ray(origin, d)


def _ray_flatten(ray: Ray):
    children = (ray.origin, ray.direction)
    aux = None
    return children, aux

def _ray_unflatten(aux, children):
    return Ray(children[0], children[1])

tree_util.register_pytree_node(Ray, _ray_flatten, _ray_unflatten)
