import jax
import jax.numpy as jnp
import math
from dataclasses import dataclass
from typing import Tuple, Any

# Assume these functions and types are implemented in JAX elsewhere:
# - unoccluded, intersect_bvh (from your BVH module)
# - Frame and frame_from_z from frame.py
# - Sampler functions: cosine_hemisphere_pdf, sample_cosine_hemisphere, sample_uniform_sphere, uniform_sphere_pdf, sample_uniform_disk_concentric
# - BSDF evaluation: bsdf.f(wo, wi)
# - Primitive: each primitive (e.g. Triangle) has methods sample(u) and sample_p(p, u) and get_pdf(), PDF(), etc.
# - Ray: a dataclass with fields origin and direction.
# - Constants:
INF = 1e10
# A placeholder for a specular BSDF flag:
BXDF_SPECULAR = 1

# -----------------------------------------------------------------------------
# Data Structures for Light Samples
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LightLiSample:
    L: jnp.ndarray      # Emitted radiance (3,)
    wi: jnp.ndarray     # Incident direction (3,)
    pdf: float          # PDF of the light sample
    intr_p: jnp.ndarray # Intersection point on the light (3,)
    intr_n: jnp.ndarray # Normal at the intersection (3,)

@dataclass(frozen=True)
class LightLeSample:
    L: jnp.ndarray         # Emitted radiance (3,)
    ray_origin: jnp.ndarray  # Origin of the emitted ray (3,)
    ray_dir: jnp.ndarray     # Direction of the emitted ray (3,)
    intr_p: jnp.ndarray      # Surface point (3,)
    intr_n: jnp.ndarray      # Surface normal (3,)
    pdf_pos: float         # PDF for position sampling
    pdf_dir: float         # PDF for directional sampling

@dataclass(frozen=True)
class SampledLight:
    light_idx: int
    pdf: float

# -----------------------------------------------------------------------------
# Diffuse Area Light
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DiffuseAreaLight:
    shape_idx: int      # Index of the shape emitting light
    Le: jnp.ndarray     # Emission color (3,)
    two_sided: bool     # Whether the light emits from both sides

    def L(self, p: jnp.ndarray, n: jnp.ndarray, w: jnp.ndarray, scale: float = 1.0) -> jnp.ndarray:
        """
        Return the emitted radiance at point p with normal n in direction w.
        Emission only occurs on the front side.
        """
        # Return Le * scale if dot(n, w) >= 0; otherwise, zero.
        return jnp.where(jnp.dot(n, w) >= 0, self.Le * scale, jnp.array([0.0, 0.0, 0.0]))

    def sample_Li(self, p: jnp.ndarray, u: jnp.ndarray, shape: Any) -> LightLiSample:
        """
        Sample the incident radiance arriving at point p from the light.
        `u` is a 2D random sample (e.g. a jnp.array of shape (2,)).
        The `shape` object is assumed to have a method sample_p(p, u)
        that returns an object with attributes:
           - p: the sampled position,
           - n: the surface normal at that point,
           - pdf: the PDF for sampling that point.
        """
        ss = shape.sample_p(p, u)
        # Only proceed if pdf is nonzero and the distance is nonzero.
        cond = jnp.logical_and(ss.pdf != 0, jnp.linalg.norm(ss.p - p)**2 != 0)
        def sample_fn(_):
            wi = (ss.p - p) / jnp.linalg.norm(ss.p - p)
            Le = self.L(ss.p, ss.n, -wi)
            return LightLiSample(L=Le, wi=wi, pdf=ss.pdf, intr_p=ss.p, intr_n=ss.n)
        default_sample = LightLiSample(L=jnp.array([0.0, 0.0, 0.0]),
                                       wi=jnp.array([0.0, 0.0, 0.0]),
                                       pdf=0.0,
                                       intr_p=jnp.array([0.0, 0.0, 0.0]),
                                       intr_n=jnp.array([0.0, 0.0, 0.0]))
        return jax.lax.cond(cond, sample_fn, lambda _: default_sample, operand=None)

    def pdf_Li(self, isect: Any, wi: jnp.ndarray, shape: Any) -> float:
        """
        Return the PDF for sampling the incident radiance from the light.
        Here, isect is assumed to have an attribute 'intersected' (a boolean)
        and 'intersected_point'.
        """
        pdf = 0.0
        if isect.intersected:
            pdf = shape.PDF(isect.intersected_point, wi)
        return pdf

    def sample_Le(self, u1: float, u2: jnp.ndarray, shape: Any) -> LightLeSample:
        """
        Sample an emitted ray from the light.
        u1: a scalar sample for position;
        u2: a 2D sample for direction.
        The `shape` object is assumed to have a method sample(u) returning
        a sample with attributes: p (position), n (normal), pdf.
        """
        ss = shape.sample(u1)
        # For two-sided lights, we sample using a cosine–hemisphere strategy with a flip.
        def side1(_):
            # Remap u2[0] to [0,1] by multiplying by 2.
            u0 = jnp.minimum(u2[0] * 2, 0.99999)
            w = sample_cosine_hemisphere(jnp.array([u0, u2[1]]))
            return w, cosine_hemisphere_pdf(jnp.abs(w[2])) / 2
        def side2(_):
            u0 = jnp.minimum((u2[0] - 0.5) * 2, 0.99999)
            w = sample_cosine_hemisphere(jnp.array([u0, u2[1]]))
            # Flip the z–component.
            w = jnp.array([w[0], w[1], -w[2]])
            return w, cosine_hemisphere_pdf(jnp.abs(w[2])) / 2
        if self.two_sided:
            w, pdf_dir = jax.lax.cond(u2[0] < 0.5, side1, side2, operand=None)
        else:
            w = sample_cosine_hemisphere(u2)
            pdf_dir = cosine_hemisphere_pdf(w[2])
        # If pdf_dir is nonzero, transform the direction to world space.
        if pdf_dir != 0:
            n_frame = frame_from_z(ss.n)  # using our JAX frame_from_z from frame.py
            w_world = n_frame.from_local(w)
            Le = self.L(ss.p, ss.n, w_world)
            return LightLeSample(L=Le,
                                 ray_origin=ss.p,
                                 ray_dir=w_world,
                                 intr_p=ss.p,
                                 intr_n=ss.n,
                                 pdf_pos=ss.pdf,
                                 pdf_dir=pdf_dir)
        else:
            return LightLeSample(L=jnp.array([0.0, 0.0, 0.0]),
                                 ray_origin=ss.p,
                                 ray_dir=jnp.array([0.0, 0.0, 0.0]),
                                 intr_p=ss.p,
                                 intr_n=ss.n,
                                 pdf_pos=ss.pdf,
                                 pdf_dir=pdf_dir)

    def pdf_Le(self, n: jnp.ndarray, w: jnp.ndarray, shape: Any) -> Tuple[float, float]:
        """
        Return the PDF for sampling an emitted ray from the light.
        """
        pdf_pos = shape.get_pdf()
        if self.two_sided:
            pdf_dir = cosine_hemisphere_pdf(jnp.abs(jnp.dot(n, w))) / 2
        else:
            pdf_dir = cosine_hemisphere_pdf(jnp.dot(n, w))
        return pdf_pos, pdf_dir

# -----------------------------------------------------------------------------
# Uniform Light Sampler
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class UniformLightSampler:
    num_lights: int

    def sample(self, u: float) -> SampledLight:
        if self.num_lights != 0:
            light_idx = min(int(u * self.num_lights), self.num_lights - 1)
            pdf = 1.0 / self.num_lights if self.num_lights > 0 else 0.0
            return SampledLight(light_idx=light_idx, pdf=pdf)
        else:
            return SampledLight(light_idx=0, pdf=0.0)

    def pmf(self) -> float:
        return 1.0 / self.num_lights if self.num_lights > 0 else 0.0

# -----------------------------------------------------------------------------
# Utility Functions for Direct Lighting
# -----------------------------------------------------------------------------
def uniform_sample_one_light(isect: Any, wo: jnp.ndarray, bsdf: Any,
                             lights: Tuple[Any, ...],
                             light_sampler: UniformLightSampler,
                             primitives: Tuple[Any, ...],
                             bvh: Any) -> jnp.ndarray:
    # In production, you would pass and update a PRNG key.
    key = jax.random.PRNGKey(0)
    s_l = light_sampler.sample(jax.random.uniform(key))
    light = lights[s_l.light_idx]
    key1, key2, key3, key4 = jax.random.split(key, 4)
    u_light = jnp.array([jax.random.uniform(key1), jax.random.uniform(key2)])
    u_scattering = jnp.array([jax.random.uniform(key3), jax.random.uniform(key4)])
    L = estimate_direct(isect, wo, bsdf, u_scattering, lights, light, s_l.light_idx, u_light, primitives, bvh)
    return L / s_l.pdf if s_l.pdf != 0 else L

def estimate_direct(isect: Any, wo: jnp.ndarray, bsdf: Any, u_scattering: jnp.ndarray,
                    lights: Tuple[Any, ...], ls: DiffuseAreaLight, light_idx: int,
                    u_light: jnp.ndarray, primitives: Tuple[Any, ...], bvh: Any) -> jnp.ndarray:
    Ld = jnp.array([0.0, 0.0, 0.0])
    # Sample light source using the area light's method.
    # We assume that isect has an attribute 'intersected_point' and that the primitive
    # associated with the light is stored at primitives[ls.shape_idx].triangle.
    li_sample = ls.sample_Li(isect.intersected_point, u_light, primitives[ls.shape_idx].triangle)
    if li_sample.pdf > 0 and not is_black(li_sample.L):
        f = bsdf.f(wo, li_sample.wi) * jnp.abs(jnp.dot(li_sample.wi, isect.normal))
        if not is_black(f):
            if unoccluded(isect.intersected_point, isect.normal, li_sample.intr_p, primitives, bvh, 1e-4):
                Ld += f * li_sample.L / li_sample.pdf
    return Ld

def is_black(v: jnp.ndarray) -> bool:
    return jnp.all(jnp.abs(v) < 1e-6)

def is_delta_light(light: Any) -> bool:
    # Implement this based on your light types. For now, assume no delta lights.
    return False

def power_heuristic(nf: float, f_pdf: float, ng: float, g_pdf: float) -> float:
    f = nf * f_pdf
    g = ng * g_pdf
    f_sq = f * f
    g_sq = g * g
    # Avoid division by zero or inf values.
    return jax.lax.cond(jnp.isinf(f_sq),
                        lambda _: 1.0,
                        lambda _: f_sq / (f_sq + g_sq),
                        operand=None)
