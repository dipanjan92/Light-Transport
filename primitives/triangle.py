import jax
import jax.numpy as jnp
from typing import NamedTuple


class Triangle(NamedTuple):
    vertex_1: jnp.ndarray  # shape (3,)
    vertex_2: jnp.ndarray  # shape (3,)
    vertex_3: jnp.ndarray  # shape (3,)
    centroid: jnp.ndarray  # shape (3,)
    normal: jnp.ndarray    # shape (3,)
    edge_1: jnp.ndarray    # shape (3,)
    edge_2: jnp.ndarray    # shape (3,)

def compute_triangle(v1: jnp.ndarray, v2: jnp.ndarray, v3: jnp.ndarray) -> Triangle:
    """Given three vertices, compute the triangle's derived data."""
    centroid = (v1 + v2 + v3) / 3.0
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = jnp.cross(edge1, edge2)
    normal = normal / jnp.linalg.norm(normal)
    return Triangle(vertex_1=v1, vertex_2=v2, vertex_3=v3,
                    centroid=centroid, normal=normal,
                    edge_1=edge1, edge_2=edge2)

# A simple shape sample structure.
class ShapeSample(NamedTuple):
    p: jnp.ndarray  # sample point (3,)
    n: jnp.ndarray  # normal (3,)
    pdf: float

# --------------------------------------------------------------------
# Intersection Function
# --------------------------------------------------------------------

def triangle_intersect(ray_origin, ray_direction, v0, v1, v2, t_max, epsilon=1e-6):
    # Compute edges and determinant.
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = jnp.cross(ray_direction, edge2)
    a = jnp.dot(edge1, h)

    # If a is near zero, the ray is parallel to the triangle.
    cond_parallel = jnp.abs(a) < epsilon

    def parallel_case(_):
        return False, t_max

    def nonparallel_case(_):
        f = 1.0 / a
        s = ray_origin - v0
        u = f * jnp.dot(s, h)
        cond_u = (u < 0.0) | (u > 1.0)

        def reject_u(_):
            return False, t_max

        def accept_u(_):
            q = jnp.cross(s, edge1)
            v = f * jnp.dot(ray_direction, q)
            cond_v = (v < 0.0) | ((u + v) > 1.0)

            def reject_v(_):
                return False, t_max

            def accept_v(_):
                t_candidate = f * jnp.dot(edge2, q)
                cond_t = (t_candidate > epsilon) & (t_candidate < t_max)

                def hit_true(_):
                    return True, t_candidate

                def hit_false(_):
                    return False, t_max

                return jax.lax.cond(cond_t, hit_true, hit_false, operand=None)

            return jax.lax.cond(cond_v, reject_v, accept_v, operand=None)

        return jax.lax.cond(cond_u, reject_u, accept_u, operand=None)

    return jax.lax.cond(cond_parallel, parallel_case, nonparallel_case, operand=None)


def __intersect_triangle(tri: Triangle, ray_o: jnp.ndarray, ray_d: jnp.ndarray, tMax: float):
    """
    Determine if the ray (origin, direction) intersects the triangle.
    Returns a tuple (hit: bool, t: float) where t is the distance along the ray.
    """
    # Unpack vertices
    p0 = tri.vertex_1
    p1 = tri.vertex_2
    p2 = tri.vertex_3

    # Check if triangle is degenerate
    cp = jnp.cross(p2 - p0, p1 - p0)
    norm_sqr = jnp.dot(cp, cp)
    if norm_sqr == 0:
        return False, 0.0

    # Transform triangle vertices to ray coordinate space
    p0t = p0 - ray_o
    p1t = p1 - ray_o
    p2t = p2 - ray_o

    # Permute components of triangle vertices and ray direction.
    # Instead of the explicit Ti-code, we choose the axis with maximum abs component.
    abs_rd = jnp.abs(ray_d)
    kz = int(jnp.argmax(abs_rd))
    kx = (kz + 1) % 3
    ky = (kx + 1) % 3

    d = jnp.array([ray_d[kx], ray_d[ky], ray_d[kz]])
    p0t = jnp.array([p0t[kx], p0t[ky], p0t[kz]])
    p1t = jnp.array([p1t[kx], p1t[ky], p1t[kz]])
    p2t = jnp.array([p2t[kx], p2t[ky], p2t[kz]])

    # Apply shear transformation to translated vertex positions
    Sx = -d[0] / d[2]
    Sy = -d[1] / d[2]
    Sz = 1.0 / d[2]

    # Update the x and y components
    p0t = jnp.array([p0t[0] + Sx * p0t[2], p0t[1] + Sy * p0t[2], p0t[2]])
    p1t = jnp.array([p1t[0] + Sx * p1t[2], p1t[1] + Sy * p1t[2], p1t[2]])
    p2t = jnp.array([p2t[0] + Sx * p2t[2], p2t[1] + Sy * p2t[2], p2t[2]])

    # Compute edge function coefficients
    e0 = p1t[0] * p2t[1] - p1t[1] * p2t[0]
    e1 = p2t[0] * p0t[1] - p2t[1] * p0t[0]
    e2 = p0t[0] * p1t[1] - p0t[1] * p1t[0]

    # Check if the edge functions have the same sign
    cond = ((e0 >= 0) and (e1 >= 0) and (e2 >= 0)) or ((e0 <= 0) and (e1 <= 0) and (e2 <= 0))
    if not cond:
        return False, 0.0

    det = e0 + e1 + e2
    if det == 0:
        return False, 0.0

    # Scale the z-components by Sz
    p0t = jnp.array([p0t[0], p0t[1], p0t[2] * Sz])
    p1t = jnp.array([p1t[0], p1t[1], p1t[2] * Sz])
    p2t = jnp.array([p2t[0], p2t[1], p2t[2] * Sz])

    tScaled = e0 * p0t[2] + e1 * p1t[2] + e2 * p2t[2]

    cond2 = ((det < 0) and (tScaled < 0) and (tScaled >= tMax * det)) or \
            ((det > 0) and (tScaled > 0) and (tScaled <= tMax * det))
    if not cond2:
        return False, 0.0

    invDet = 1.0 / det
    # Barycentrics (if needed, here computed but not returned separately)
    b0 = e0 * invDet
    b1 = e1 * invDet
    b2 = e2 * invDet
    t = tScaled * invDet

    if t <= 0:
        return False, 0.0

    return True, t

# --------------------------------------------------------------------
# Area, Bounds, and PDF Functions
# --------------------------------------------------------------------

def get_area(tri: Triangle) -> float:
    """Return the area of the triangle."""
    return 0.5 * jnp.linalg.norm(jnp.cross(tri.edge_1, tri.edge_2))

def get_bounds(tri: Triangle):
    """Return (min_point, max_point) for the triangle."""
    min_p = jnp.minimum(jnp.minimum(tri.vertex_1, tri.vertex_2), tri.vertex_3)
    max_p = jnp.maximum(jnp.maximum(tri.vertex_1, tri.vertex_2), tri.vertex_3)
    return min_p, max_p

def get_pdf(tri: Triangle) -> float:
    """Return the PDF value for uniformly sampling the triangle."""
    area = get_area(tri)
    return 1.0 / area

# --------------------------------------------------------------------
# Sampling Functions
# --------------------------------------------------------------------

def sample_uniform_triangle(u: jnp.ndarray) -> jnp.ndarray:
    """
    Given a 2D sample u (with two components in [0,1]), return barycentric
    coordinates (b0, b1, b2) for a uniformly sampled point on the triangle.
    """
    def branch1(_):
        b0 = u[0] / 2.0
        b1 = u[1] - b0
        return b0, b1
    def branch2(_):
        b1 = u[1] / 2.0
        b0 = u[0] - b1
        return b0, b1
    b0, b1 = jax.lax.cond(u[0] < u[1], branch1, branch2, operand=None)
    return jnp.array([b0, b1, 1.0 - b0 - b1])

def sample(tri: Triangle, u: jnp.ndarray) -> ShapeSample:
    """
    Sample a point uniformly on the triangle using a 2D sample u.
    Returns a ShapeSample with the sampled point, the triangle's normal, and the PDF.
    """
    b = sample_uniform_triangle(u)
    p = b[0] * tri.vertex_1 + b[1] * tri.vertex_2 + b[2] * tri.vertex_3
    pdf_val = get_pdf(tri)
    return ShapeSample(p=p, n=tri.normal, pdf=pdf_val)

def sample_p(tri: Triangle, p: jnp.ndarray, u: jnp.ndarray) -> ShapeSample:
    """
    Given a reference point p and a 2D sample u, sample a point on the triangle
    and adjust the PDF based on the geometry.
    """
    ss = sample(tri, u)
    pdf_val = ss.pdf
    wi = ss.p - p
    if jnp.dot(wi, wi) == 0:
        pdf_val = 0.0
    else:
        wi_norm = wi / jnp.linalg.norm(wi)
        pdf_val = pdf_val * (jnp.sum((p - ss.p) ** 2)) / jnp.abs(jnp.dot(ss.n, -wi_norm))
        pdf_val = jnp.where(jnp.isinf(pdf_val), 0.0, pdf_val)
    return ShapeSample(p=ss.p, n=ss.n, pdf=pdf_val)

# def PDF(tri: Triangle, p: jnp.ndarray, wi: jnp.ndarray, tMax: float = INF) -> float:
#     """
#     Compute the probability density function value for sampling the triangle
#     from a given point p in the direction wi.
#     """
#     hit, t = intersect_triangle(tri, p, wi, tMax)
#     pdf_val = 0.0
#     if hit:
#         isec_p = p + t * wi
#         cos_theta = jnp.abs(jnp.dot(tri.normal, -wi))
#         if cos_theta > 0:
#             pdf_val = get_pdf(tri) / (cos_theta / jnp.sum((p - isec_p) ** 2))
#     return pdf_val

# --------------------------------------------------------------------
# Solid Angle Function
# --------------------------------------------------------------------

def solid_angle(tri: Triangle, p: jnp.ndarray) -> float:
    """
    Compute the solid angle subtended by the triangle as seen from point p.
    This implementation uses the spherical excess method.
    """
    # Compute vectors from p to the triangle vertices.
    v0 = tri.vertex_1 - p
    v1 = tri.vertex_2 - p
    v2 = tri.vertex_3 - p
    # Normalize the vectors.
    v0 = v0 / jnp.linalg.norm(v0)
    v1 = v1 / jnp.linalg.norm(v1)
    v2 = v2 / jnp.linalg.norm(v2)
    # Compute the angles between the vectors.
    def angle(u, v):
        return jnp.arccos(jnp.clip(jnp.dot(u, v), -1.0, 1.0))
    a0 = angle(v0, v1)
    a1 = angle(v1, v2)
    a2 = angle(v2, v0)
    # The spherical excess is (a0 + a1 + a2 - pi)
    E = a0 + a1 + a2 - jnp.pi
    return E
