import jax
import jax.numpy as jnp
from jax import random
from typing import Any, Dict, Tuple

# Assume these are implemented elsewhere in JAX:
# - intersect_bvh(ray, primitives, bvh, t_min, t_max) -> Intersection
# - spawn_ray(p, n, wi) -> Ray
# - is_black(v: jnp.ndarray) -> bool
# - max_component(v: jnp.ndarray) -> float
# - path tracer BSDF functions: bsdf.f(wo, wi), bsdf.sample_f(wo, u, u2) returning an object with attributes f, pdf, wi, flags, etc.
# - light_sampler.sample(u) returning a SampledLight (with attributes light_idx and pdf, and for area lights a shape_idx)
# - Also, primitives, bvh, lights, etc., are assumed to be JAX–compatible.

# A constant for infinity.
INF = 1e10


def path_trace(key: jnp.ndarray,
               ray: Any,
               primitives: Any,
               bvh: Any,
               lights: Any,
               light_sampler: Any,
               sample_lights: int = 1,
               sample_bsdf: int = 1,
               max_depth: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform path tracing starting from the given ray.
    Returns a tuple (L, key_new), where L is the accumulated radiance (a 3-vector)
    and key_new is the updated PRNG key.

    Assumes all helper functions (intersect_bvh, spawn_ray, is_black, etc.) are implemented in JAX.
    """
    # Initialize state.
    state = {
        "L": jnp.array([0.0, 0.0, 0.0]),  # Accumulated radiance.
        "beta": jnp.array([1.0, 1.0, 1.0]),  # Path throughput.
        "ray": ray,  # Current ray.
        "depth": 0,  # Recursion depth.
        "done": False,  # Termination flag.
        "key": key
    }

    def cond_fn(state: Dict[str, Any]) -> bool:
        return jnp.logical_and(state["depth"] < max_depth, jnp.logical_not(state["done"]))

    def body_fn(state: Dict[str, Any]) -> Dict[str, Any]:
        key = state["key"]
        # Intersect the ray with the scene.
        isect = intersect_bvh(state["ray"], primitives, bvh, 0.0, INF)

        # If no intersection, mark done.
        state = jax.lax.cond(isect.intersected == 0,
                             lambda s: {**s, "done": True},
                             lambda s: s,
                             operand=state)

        # If done, no further work is needed.
        # Otherwise, accumulate emission from the intersected object.
        emission = isect.nearest_object.material.emission
        new_L = state["L"] + state["beta"] * emission

        # Russian roulette for termination after depth > 4.
        def rr_fn(s: Dict[str, Any]) -> Tuple[jnp.ndarray, bool, jnp.ndarray]:
            # r_r is the maximum component of the diffuse reflectance.
            r_r = max_component(isect.nearest_object.material.diffuse)
            key_rr, key_new = random.split(s["key"])
            rr_sample = random.uniform(key_rr, minval=0.0, maxval=1.0)
            new_beta = jax.lax.cond(rr_sample >= r_r,
                                    lambda _: jnp.array([0.0, 0.0, 0.0]),
                                    lambda _: s["beta"] / r_r,
                                    operand=None)
            done_flag = rr_sample >= r_r
            return new_beta, done_flag, key_new

        new_beta, done_flag, key = jax.lax.cond(state["depth"] > 4,
                                                lambda s: rr_fn(s),
                                                lambda s: (s["beta"], False, s["key"]),
                                                operand=state)
        state = {**state, "beta": new_beta, "key": key}
        state = jax.lax.cond(done_flag,
                             lambda s: {**s, "done": True},
                             lambda s: s,
                             operand=state)

        # Increment depth.
        state = {**state, "depth": state["depth"] + 1}

        # Get the BSDF from the intersected primitive and initialize its local frame.
        bsdf = isect.nearest_object.bsdf
        bsdf = bsdf.init_frame(isect.normal, isect.dpdu)

        wo = - state["ray"].direction  # Outgoing direction.

        # Direct lighting contribution.
        def direct_lighting(s: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
            key_dl, key_new = random.split(s["key"])
            s_l = light_sampler.sample(random.uniform(key_dl))
            light = lights[s_l.light_idx]
            key1, key2 = random.split(key_new)
            u_light = jnp.array([random.uniform(key1), random.uniform(key2)])
            # Assume that the shape for the light is stored in primitives[s_l.shape_idx].triangle.
            l_shape = primitives[s_l.shape_idx].triangle
            ls = light.sample_Li(isect.intersected_point, u_light, l_shape)
            f = bsdf.f(wo, ls.wi) * jnp.abs(jnp.dot(ls.wi, isect.normal))
            cond_direct = jnp.logical_and(jnp.logical_not(is_black(ls.L)),
                                          ls.pdf > 0)
            cond_vis = unoccluded(isect.intersected_point, isect.normal, ls.intr_p, primitives, bvh, 1e-4)
            added = jax.lax.cond(jnp.logical_and(cond_direct, cond_vis),
                                 lambda _: s["beta"] * (f * ls.L / ls.pdf) / s_l.pdf,
                                 lambda _: jnp.array([0.0, 0.0, 0.0]),
                                 operand=None)
            return added, key_new

        added_direct, key = jax.lax.cond(sample_lights,
                                         direct_lighting,
                                         lambda s: (jnp.array([0.0, 0.0, 0.0]), s["key"]),
                                         operand=state)
        new_L = new_L + added_direct

        # BSDF sampling.
        key_bsdf1, key_bsdf2, key_new = random.split(state["key"], 3)
        u_val = random.uniform(key_bsdf1)
        u2 = jnp.array([random.uniform(key_bsdf2), random.uniform(key_bsdf2)])
        bs = bsdf.sample_f(wo, u_val, u2)
        state = jax.lax.cond(jnp.logical_or(is_black(bs.f), bs.pdf == 0),
                             lambda s: {**s, "done": True},
                             lambda s: s,
                             operand=state)
        new_beta = state["beta"] * bs.f * jnp.abs(jnp.dot(bs.wi, isect.normal)) / bs.pdf
        # Update throughput.
        state = {**state, "beta": new_beta}
        # Spawn new ray.
        new_ray = spawn_ray(isect.intersected_point, isect.normal, bs.wi)
        state = {**state, "ray": new_ray, "L": new_L, "key": key_new}
        return state

    final_state = jax.lax.while_loop(cond_fn, body_fn, state)
    return final_state["L"], final_state["key"]
