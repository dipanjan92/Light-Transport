import math
import logging
import numba
import numpy as np

from .brdf import *
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
        # terminate path if max depth is reached
        if bounce>=scene.max_depth:
            break
        # intersect ray with scene
        isect = hit_object(spheres, triangles, bvh, ray)

        # terminate path if no intersection is found
        if isect.min_distance is None:
            break

        min_distance = isect.min_distance
        nearest_object_material = isect.material
        intersection = isect.intersected_point
        surface_normal = isect.normal

        # add emitted light at intersection
        # if bounce==0 or specular_bounce:
        light = light + (nearest_object_material.emission * throughput)

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
            # Next Event Estimation - Direct Light
            direct_light = throughput * cast_one_shadow_ray(scene, spheres, triangles, bvh, nearest_object_material, intersection, surface_normal)
            light = light+direct_light

            # Indirect Light
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_diffuse(nearest_object_material, surface_normal, ray)

            throughput = throughput * (nearest_object_material.color.diffuse * brdf * np.abs(np.dot(new_ray_direction, surface_normal)) / pdf_fwd)

            # intersection = intersection+EPSILON*new_ray_direction
            ray = Ray(intersection, new_ray_direction, EPSILON)
            specular_bounce = False
            continue

        elif nearest_object_material.type==MatType.MIRROR.value:
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_mirror(nearest_object_material, surface_normal, ray)
            # intersection = intersection+EPSILON*new_ray_direction
            ray = Ray(intersection, new_ray_direction, EPSILON)
            specular_bounce = True
            continue

        elif nearest_object_material.type==MatType.SPECULAR.value:
            # specular reflection (only dielectric materials)
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_specular(nearest_object_material, surface_normal, ray)

            if pdf_fwd!=0:
                throughput = throughput * brdf / pdf_fwd # update throughput

            # if intr_type==Medium.REFRACTION.value:
            #     intersection = intersection+(-EPSILON)*new_ray_direction
            # else:
            #     intersection = intersection+EPSILON*new_ray_direction
            ray = Ray(intersection, new_ray_direction, EPSILON)
            specular_bounce = False
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

                        cam_ray = Ray(cam_origin, cam_direction, 0)

                        color = color + trace_path(scene, spheres, triangles, bvh, cam_ray, 0)

                    color = color/spp
                    scene.image[y, x, :] = scene.image[y, x, :] + 0.25 * np.clip(color, 0, 1)

    return scene.image

