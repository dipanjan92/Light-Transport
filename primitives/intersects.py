import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any

# We assume that a type Primitive is defined elsewhere and that it contains, for example:
# - shape_type: an integer (0 for triangle, 1 for sphere)
# - triangle: an object with attribute 'normal', 'edge_1', 'edge_2'
# - sphere: an object with attribute 'center'
# - is_light: a boolean flag
# - material: an object with attribute 'emission'
# - bsdf: an object with a method init_frame(shading_normal, dpdu)
from primitives.primitives import Primitive


@dataclass(frozen=True)
class Intersection:
    min_distance: float  # Distance along the ray
    intersected_point: jnp.ndarray  # (3,) point of intersection
    normal: jnp.ndarray  # (3,) geometric normal at the intersection
    shading_normal: jnp.ndarray  # (3,) normal used for shading
    dpdu: jnp.ndarray  # (3,) partial derivative of the surface with respect to u
    dpdv: jnp.ndarray  # (3,) partial derivative of the surface with respect to v
    dndu: jnp.ndarray  # (3,) derivative of the normal with respect to u
    dndv: jnp.ndarray  # (3,) derivative of the normal with respect to v
    nearest_object: Any  # The primitive that was intersected
    intersected: int  # 1 if an intersection was found, 0 otherwise


def set_intersection(ray: Any, prim: Primitive, tMax: float) -> Intersection:
    """
    Given a ray, a primitive, and a hit distance tMax, return a new Intersection
    with the appropriate fields set.

    Assumes ray has attributes 'origin' and 'direction'.
    """
    new_point = ray.origin + tMax * ray.direction
    if prim.shape_type == 0:
        # For a triangle primitive.
        new_normal = prim.triangle.normal
        # Compute triangle interaction derivatives.
        tri_inter = calculate_triangle_interaction(prim.triangle)
    elif prim.shape_type == 1:
        # For a sphere primitive.
        vec = new_point - prim.sphere.center
        new_normal = vec / jnp.linalg.norm(vec)
        tri_inter = calculate_sphere_interaction(prim.sphere)
    else:
        new_normal = jnp.array([0.0, 0.0, 0.0])
        tri_inter = None

    shading_normal = new_normal  # flat shading for now

    # Build and return a new Intersection.
    return Intersection(
        min_distance=tMax,
        intersected_point=new_point,
        normal=new_normal,
        shading_normal=shading_normal,
        dpdu=tri_inter.dpdu if tri_inter is not None else jnp.array([0.0, 0.0, 0.0]),
        dpdv=tri_inter.dpdv if tri_inter is not None else jnp.array([0.0, 0.0, 0.0]),
        dndu=tri_inter.dndu if tri_inter is not None else jnp.array([0.0, 0.0, 0.0]),
        dndv=tri_inter.dndv if tri_inter is not None else jnp.array([0.0, 0.0, 0.0]),
        nearest_object=prim,
        intersected=1
    )


def Le(inter: Intersection, d: jnp.ndarray) -> jnp.ndarray:
    """
    Return the emitted radiance (Le) from the intersected primitive.
    For an area light, if the dot product of the normal and direction d is nonnegative,
    the light emits its emission.
    """
    L = jnp.array([0.0, 0.0, 0.0])
    if (inter.nearest_object.shape_type == 0) and (inter.nearest_object.is_light):
        # For an area light, only emit from the front side.
        if jnp.dot(inter.normal, d) >= 0:
            L = inter.nearest_object.material.emission
    return L


def calculate_triangle_interaction(triangle: Any) -> Intersection:
    """
    For a triangle, return an Intersection (with dummy values for unrelated fields)
    in which dpdu and dpdv are set to the triangle's edges and dndu/dndv are zero.
    (This is for flat shading.)
    """
    dpdu = triangle.edge_1
    dpdv = triangle.edge_2
    zero = jnp.array([0.0, 0.0, 0.0])
    # Return a temporary Intersection object with only the derivative fields filled.
    return Intersection(
        min_distance=0.0,
        intersected_point=jnp.zeros(3),
        normal=triangle.normal,
        shading_normal=triangle.normal,
        dpdu=dpdu,
        dpdv=dpdv,
        dndu=zero,
        dndv=zero,
        nearest_object=None,
        intersected=1
    )


def calculate_sphere_interaction(sphere: Any) -> Intersection:
    """
    For a sphere, a simplified interaction. Here we return zero derivatives.
    In a full implementation, you might compute partial derivatives on the sphere.
    """
    zero = jnp.array([0.0, 0.0, 0.0])
    # Return a dummy Intersection with zero derivatives.
    return Intersection(
        min_distance=0.0,
        intersected_point=jnp.zeros(3),
        normal=jnp.array([0.0, 0.0, 1.0]),
        shading_normal=jnp.array([0.0, 0.0, 1.0]),
        dpdu=zero,
        dpdv=zero,
        dndu=zero,
        dndv=zero,
        nearest_object=None,
        intersected=1
    )


def get_bsdf(inter: Intersection) -> Any:
    """
    Return the BSDF from the intersected primitive.
    Assumes that the primitive’s bsdf has a method init_frame(shading_normal, dpdu)
    that returns a BSDF configured for shading.
    """
    bsdf = inter.nearest_object.bsdf
    # Here we assume that bsdf.init_frame returns a new (initialized) BSDF.
    return bsdf.init_frame(inter.shading_normal, inter.dpdu)
