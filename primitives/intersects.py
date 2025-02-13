# intersects.py
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from jax import tree_util

INF = 1e10

@dataclass(frozen=True)
class Intersection:
    min_distance: float = INF
    intersected_point: jnp.ndarray = field(default_factory=lambda: jnp.zeros(3, dtype=jnp.float32))
    normal: jnp.ndarray = field(default_factory=lambda: jnp.zeros(3, dtype=jnp.float32))
    shading_normal: jnp.ndarray = field(default_factory=lambda: jnp.zeros(3, dtype=jnp.float32))
    dpdu: jnp.ndarray = field(default_factory=lambda: jnp.zeros(3, dtype=jnp.float32))
    dpdv: jnp.ndarray = field(default_factory=lambda: jnp.zeros(3, dtype=jnp.float32))
    dndu: jnp.ndarray = field(default_factory=lambda: jnp.zeros(3, dtype=jnp.float32))
    dndv: jnp.ndarray = field(default_factory=lambda: jnp.zeros(3, dtype=jnp.float32))
    nearest_object: int = -1
    intersected: int = 0

@jax.jit
def set_intersection(ray_origin: jnp.ndarray,
                     ray_direction: jnp.ndarray,
                     v0: jnp.ndarray,
                     v1: jnp.ndarray,
                     v2: jnp.ndarray,
                     t: float) -> Intersection:
    new_intersected_point = ray_origin + t * ray_direction
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = jnp.cross(edge1, edge2)
    norm = jnp.linalg.norm(normal)
    normal = jax.lax.cond(
        norm > 0,
        lambda n: normal / norm,
        lambda n: normal,
        operand=norm
    )
    return Intersection(
        min_distance=t,
        intersected_point=new_intersected_point,
        normal=normal,
        shading_normal=normal,
        dpdu=edge1,
        dpdv=edge2,
        dndu=jnp.zeros(3, dtype=jnp.float32),
        dndv=jnp.zeros(3, dtype=jnp.float32),
        nearest_object=-1,
        intersected=1,
    )

def _intersection_flatten(isec: Intersection):
    children = (isec.min_distance,
                isec.intersected_point,
                isec.normal,
                isec.shading_normal,
                isec.dpdu,
                isec.dpdv,
                isec.dndu,
                isec.dndv,
                isec.nearest_object,
                isec.intersected)
    aux = None
    return children, aux

def _intersection_unflatten(aux, children):
    return Intersection(*children)

tree_util.register_pytree_node(Intersection, _intersection_flatten, _intersection_unflatten)
