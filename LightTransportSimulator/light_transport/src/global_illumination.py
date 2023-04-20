import math
import logging
import numba
import numpy as np

from .brdf import *
from .bvh import traverse_bvh
from .constants import inv_pi, EPSILON, MatType, ONES
from .control_variates import calculate_dlogpdu, estimate_alpha
from .light_samples import cast_one_shadow_ray
from .rays import Ray
from .utils import uniform_hemisphere_sampling, hit_object, nearest_intersected_object, \
    cosine_weighted_hemisphere_sampling
from .vectors import normalize


@numba.njit
def trace_path(scene, spheres, triangles, bvh, ray, bounce):
    throughput = ONES
    light = ZEROS
    specular_bounce = False

    while True:
        if bounce>=scene.max_depth:
            break
        # intersect ray with scene
        isect = hit_object(spheres, triangles, bvh, ray)

        nearest_triangle = isect.nearest_triangle
        nearest_sphere = isect.nearest_sphere

        # terminate path if no intersection is found
        if nearest_triangle is None and nearest_sphere is None:
            # no object was hit
            break

        if nearest_triangle is None:
            nearest_object_material = nearest_sphere.material
        else:
            nearest_object_material = nearest_triangle.material

        min_distance = isect.min_distance
        intersection = isect.intersected_point
        surface_normal = isect.normal

        # add emitted light at intersection
        light = light + (nearest_object_material.emission * throughput)

        # update throughput
        throughput = throughput * nearest_object_material.color.diffuse

        # print('P: ', intersection, ' C: ',  nearest_object.material.color.diffuse, ' L: ', light, ' F: ', throughput)

        # Russian roulette for variance reduction
        if bounce> 4:
            r_r = np.amax(nearest_object_material.color.diffuse)
            if np.random.random() >= r_r:
                break
            throughput = throughput/r_r

        bounce += 1 # increment the bounce

        if nearest_object_material.type==MatType.DIFFUSE.value:
            # diffuse surface
            # print('surface normal: ', surface_normal)
            nl = surface_normal
            # flip normal if facing opposite direction
            if np.dot(surface_normal, ray.direction) > 0:
                nl = -surface_normal
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_diffuse(nearest_object_material, nl, ray)
            intersection = intersection+EPSILON*new_ray_direction
            ray = Ray(intersection, new_ray_direction, 1e-4)
            # Next Event Estimation - Direct Light
            direct_light = cast_one_shadow_ray(scene, spheres, triangles, bvh, nearest_object_material, intersection, nl)
            # print(direct_light)
            light = light+direct_light
            continue
        elif nearest_object_material.type==MatType.MIRROR.value:
            # perfect mirror reflection
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_mirror(nearest_object_material, surface_normal, ray)
            intersection = intersection+EPSILON*new_ray_direction
            ray = Ray(intersection, new_ray_direction, 1e-4)
            continue
        elif nearest_object_material.type==MatType.SPECULAR.value:
            # specular reflection (only dielectric materials)
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_specular(nearest_object_material, surface_normal, ray)
            throughput *= pdf_fwd # update throughput
            if intr_type==Medium.REFRACTION.value:
                intersection = intersection+(-EPSILON)*new_ray_direction
            else:
                intersection = intersection+EPSILON*new_ray_direction
            ray = Ray(intersection, new_ray_direction, 1e-4)
            continue
        else:
            # error in material metadata
            break

    return light



@numba.njit(parallel=True)
def render_scene(scene, spheres, triangles, bvh):
    np.random.seed(79402371)

    eye = scene.camera.position
    look_at = scene.camera.look_at

    fov =  scene.camera.fov # field of view

    cam_x = np.array([scene.width * fov / scene.height, 0.0, 0.0], dtype=np.float64)
    cam_y = normalize(np.cross(cam_x, look_at)) * fov

    # print(eye, look_at, fov, cam_x, cam_y)

    h = scene.height
    w = scene.width
    spp = scene.number_of_samples

    for y in numba.prange(h):
        print(100.0 * y / (h - 1))
        for x in numba.prange(w):
            # for each pixel
            for sy in range(2):
                i = (h - 1 - y) * w + x
                for sx in range(2):
                    color = ZEROS
                    for s in numba.prange(spp):
                        # two random vars
                        u1 = 2.0 * np.random.random()
                        u2 = 2.0 * np.random.random()

                        # ray differentials for anti-aliasing
                        dx = np.sqrt(u1) - 1.0 if u1 < 1 else 1.0 - np.sqrt(2.0 - u1)
                        dy = np.sqrt(u2) - 1.0 if u2 < 1 else 1.0 - np.sqrt(2.0 - u2)

                        cam_direction = cam_x * (((sx + 0.5 + dx) / 2.0 + x) / w - 0.5) + \
                                        cam_y * (((sy + 0.5 + dy) / 2.0 + y) / h - 0.5) + look_at

                        cam_origin = eye + cam_direction * 130
                        cam_direction = normalize(cam_direction)

                        # print('cam: ', cam_origin, cam_direction)

                        cam_ray = Ray(cam_origin, cam_direction, 1e-4)

                        color = color + trace_path(scene, spheres, triangles, bvh, cam_ray, 0)

                    color = color/spp
                    scene.image[y, x, :] = scene.image[y, x, :] + 0.25 * np.clip(color, 0, 1)

    return scene.image

