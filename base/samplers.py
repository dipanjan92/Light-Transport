import math
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple

# Constant for inverse pi.
inv_pi = 1.0 / math.pi


# -----------------------------------------------------------------------------
# Data Structure
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ShapeSample:
    p: jnp.ndarray  # Sampled point (3,)
    n: jnp.ndarray  # Surface normal at sample (3,)
    pdf: float  # PDF for sampling that point


# -----------------------------------------------------------------------------
# Uniform Sphere Sampling
# -----------------------------------------------------------------------------
def sample_uniform_sphere(u: jnp.ndarray) -> jnp.ndarray:
    """
    Given a 2D sample u (shape (2,)), return a uniformly sampled point on the unit sphere.
    """
    # u[0] and u[1] are assumed to be in [0,1].
    z = 1.0 - 2.0 * u[0]
    r = jnp.sqrt(jnp.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * math.pi * u[1]
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), z])


def uniform_sphere_pdf() -> float:
    return 1.0 / (4.0 * math.pi)


# -----------------------------------------------------------------------------
# Uniform Hemisphere Sampling
# -----------------------------------------------------------------------------
def sample_uniform_hemisphere(u: jnp.ndarray) -> jnp.ndarray:
    """
    Given a 2D sample u (shape (2,)) return a point uniformly sampled on the unit hemisphere.
    """
    z = u[0]
    r = jnp.sqrt(jnp.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * math.pi * u[1]
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    return jnp.array([x, y, z])


def uniform_hemisphere_pdf() -> float:
    return 1.0 / (2.0 * math.pi)


# -----------------------------------------------------------------------------
# Uniform Disk Sampling (Concentric Mapping)
# -----------------------------------------------------------------------------
def sample_uniform_disk_concentric(u: jnp.ndarray) -> jnp.ndarray:
    """
    Given a 2D sample u (shape (2,)), return a point sampled on a unit disk using polar coordinates.
    This is a simple disk sampling (not the concentric mapping version).
    """
    r = jnp.sqrt(u[0])
    theta = 2.0 * math.pi * u[1]
    return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta)])


def sample_uniform_disk_polar(u: jnp.ndarray) -> jnp.ndarray:
    """
    Alternative uniform disk sampling in polar coordinates.
    """
    r = jnp.sqrt(u[0])
    theta = 2.0 * math.pi * u[1]
    return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta)])


# -----------------------------------------------------------------------------
# Concentric Disk Sampling
# -----------------------------------------------------------------------------
def concentric_sample_disk(u: jnp.ndarray) -> jnp.ndarray:
    """
    Maps a uniformly distributed sample (u) in [0,1]^2 to a uniformly distributed sample
    on a unit disk using the concentric mapping.
    """
    # Map to [-1, 1]^2.
    u_offset = 2.0 * u - jnp.array([1.0, 1.0])

    def nonzero_fn(_):
        # We have two branches depending on the relative magnitude of the components.
        def branch1(_):
            r = u_offset[0]
            theta = (math.pi / 4.0) * (u_offset[1] / u_offset[0])
            return r * jnp.array([jnp.cos(theta), jnp.sin(theta)])

        def branch2(_):
            r = u_offset[1]
            theta = (math.pi / 2.0) - (math.pi / 4.0) * (u_offset[0] / u_offset[1])
            return r * jnp.array([jnp.cos(theta), jnp.sin(theta)])

        return jax.lax.cond(jnp.abs(u_offset[0]) > jnp.abs(u_offset[1]),
                            branch1, branch2, operand=None)

    return jax.lax.cond(jnp.logical_and(u_offset[0] == 0.0, u_offset[1] == 0.0),
                        lambda _: jnp.array([0.0, 0.0]),
                        nonzero_fn,
                        operand=None)


# -----------------------------------------------------------------------------
# Cosine-weighted Hemisphere Sampling
# -----------------------------------------------------------------------------
def sample_cosine_hemisphere(u: jnp.ndarray) -> jnp.ndarray:
    """
    Sample a direction in the hemisphere with cosine–weighted distribution.
    u is a 2D sample in [0,1]^2.
    """
    d = concentric_sample_disk(u)
    # Compute z coordinate from disk sample.
    z = jnp.sqrt(jnp.maximum(0.0, 1.0 - d[0] ** 2 - d[1] ** 2))
    return jnp.array([d[0], d[1], z])


def cosine_hemisphere_pdf(cos_theta: float) -> float:
    """
    Return the PDF for cosine–weighted hemisphere sampling.
    """
    return cos_theta * inv_pi
