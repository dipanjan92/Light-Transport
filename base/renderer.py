import jax
import jax.numpy as jnp
from jax import random
from typing import Any, Tuple


# Assume the following are imported from your other modules (all converted to JAX):
#   - Ray (a dataclass with fields origin and direction)
#   - path_trace(ray, primitives, bvh, lights, light_sampler, sample_lights, sample_bsdf, max_depth)
#   - trace_mis(ray, primitives, bvh, lights, light_sampler, sample_lights, sample_bsdf, max_depth)
#   - UniformLightSampler (converted to a functional or dataclass version)
#
# Also assume that the scene structure has attributes:
#   scene.spp (samples per pixel, int)
#   scene.integrator (0 or 1)
#   scene.sample_lights, scene.sample_bsdf, scene.max_depth
#
# And the camera has attributes:
#   camera.width, camera.height, and a method generate_ray(u, v) -> (ray_origin, ray_direction)

def render(key: Any,
           scene: Any,
           image_shape: Tuple[int, int],
           lights: Any,
           camera: Any,
           primitives: Any,
           bvh: Any) -> jnp.ndarray:
    """
    Render an image given the scene, camera, lights, primitives and BVH.
    `key` is a PRNG key.
    `image_shape` is a tuple (height, width).
    """
    height, width = image_shape
    spp = scene.spp

    # Create a light sampler instance.
    light_sampler = UniformLightSampler(num_lights=lights.shape[0])

    # We'll define a function that renders one pixel.
    # To simplify key management, we assume that for each pixel we generate spp independent keys.
    def render_pixel(pixel_idx: Tuple[int, int], key: Any) -> jnp.ndarray:
        j, i = pixel_idx  # j is row (height index), i is column (width index)
        # Generate a separate PRNG key for this pixel and split into spp keys.
        pixel_key, subkey = random.split(key)
        sample_keys = random.split(subkey, spp)

        # Accumulate radiance for this pixel.
        def sample_body(acc, sample_key):
            # For each sample, generate two independent uniform random numbers for u and v.
            key1, key2 = random.split(sample_key)
            u = (i + random.uniform(key1, minval=0.0, maxval=1.0)) / width
            v = (j + random.uniform(key2, minval=0.0, maxval=1.0)) / height
            # Generate the camera ray.
            ray_origin, ray_direction = camera.generate_ray(u, v)
            ray = Ray(origin=ray_origin, direction=ray_direction)
            # Choose the integrator.
            sample_L = jax.lax.cond(
                scene.integrator == 0,
                lambda _: path_trace(ray, primitives, bvh, lights, light_sampler,
                                     sample_lights=scene.sample_lights,
                                     sample_bsdf=scene.sample_bsdf,
                                     max_depth=scene.max_depth),
                lambda _: trace_mis(ray, primitives, bvh, lights, light_sampler,
                                    sample_lights=scene.sample_lights,
                                    sample_bsdf=scene.sample_bsdf,
                                    max_depth=scene.max_depth),
                operand=None
            )
            return acc + sample_L, None

        # Use lax.scan to loop over spp samples.
        L_total, _ = jax.lax.scan(sample_body,
                                  jnp.array([0.0, 0.0, 0.0]),
                                  sample_keys)
        return L_total / spp

    # Create a grid of pixel indices.
    jj = jnp.arange(height)
    ii = jnp.arange(width)
    # Use jnp.meshgrid to create a (height, width, 2) array of pixel indices.
    grid_j, grid_i = jnp.meshgrid(jj, ii, indexing='ij')
    # Flatten the pixel index arrays.
    pixel_indices = jnp.stack([grid_j.flatten(), grid_i.flatten()], axis=-1)  # shape (height*width, 2)

    # Split the main key into one key per pixel.
    pixel_keys = random.split(key, pixel_indices.shape[0])

    # Vectorize render_pixel over the flattened pixel index and key.
    render_pixel_vmap = jax.vmap(render_pixel, in_axes=(0, 0))
    pixels_flat = render_pixel_vmap(pixel_indices, pixel_keys)  # shape (height*width, 3)
    # Reshape back to image shape.
    image = pixels_flat.reshape((height, width, 3))
    return image
