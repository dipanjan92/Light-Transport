import math
import logging
import numba
import numpy as np

from .brdf import *
from .constants import inv_pi, EPSILON, MatType, ONES
from .control_variates import calculate_dlogpdu, estimate_alpha
from .light_samples import cast_one_shadow_ray, sample_light, sample_light_direction
from .rays import Ray
from .utils import uniform_hemisphere_sampling, hit_object, nearest_intersected_object, \
    cosine_weighted_hemisphere_sampling
from .vectors import normalize


@numba.njit
def trace_light_path(scene, spheres, triangles, bvh, light_ray, bounce):
    light_throughput = ONES

    # intersect light ray with scene
    light_isect = hit_object(spheres, triangles, bvh, light_ray)

    light_nearest_triangle = light_isect.nearest_triangle
    light_nearest_sphere = light_isect.nearest_sphere

    if light_nearest_triangle is None and light_nearest_sphere is None:
        return None, None, None

    if light_nearest_triangle is None:
        light_nearest_object_material = light_nearest_sphere.material
    else:
        light_nearest_object_material = light_nearest_triangle.material

    light_intersection = light_isect.intersected_point
    light_surface_normal = light_isect.normal

    return light_intersection, light_surface_normal, light_nearest_object_material



@numba.njit
def is_close(a, b, rtol=1e-5, atol=1e-8):
    return np.abs(a - b) <= (atol + rtol * np.abs(b))


@numba.njit
def connect_paths(scene, spheres, triangles, bvh, intersection, surface_normal, object_material, light_intersection, light_surface_normal, light_object_material, camera_ray, light_ray, camera_throughput, light_throughput):
    connection = light_intersection - intersection

    # shadow ray
    shadow_ray = Ray(intersection, normalize(connection), EPSILON)
    shadow_intersection = hit_object(spheres, triangles, bvh, shadow_ray)

    if is_close(shadow_intersection.min_distance, np.linalg.norm(connection)): # if intersections match
        geometric_term = np.abs(np.dot(shadow_ray.direction, light_surface_normal)) * np.abs(np.dot(-shadow_ray.direction, surface_normal)) / np.dot(connection, connection)

        camera_brdf = None
        if object_material.type == MatType.DIFFUSE.value:
            camera_brdf = object_material.color.diffuse * inv_pi

        connection_weight_camera = np.dot(-camera_ray.direction, surface_normal)
        connection_weight_light = np.dot(light_ray.direction, surface_normal)

        camera_contribution = camera_throughput * camera_brdf * connection_weight_camera
        light_contribution = light_throughput * light_object_material.emission * connection_weight_light

        multiple_importance_sampling_weight = (np.dot(camera_contribution, camera_contribution) / (np.dot(camera_contribution, camera_contribution) + np.dot(light_contribution, light_contribution)))

        return camera_contribution * geometric_term * multiple_importance_sampling_weight
    else:
        return np.zeros_like(camera_throughput)


@numba.njit
def trace_path(scene, spheres, triangles, bvh, ray, bounce):
    throughput = ONES
    light = ZEROS
    specular_bounce = False

    while True:
        if bounce >= scene.max_depth:
            break

        isect = hit_object(spheres, triangles, bvh, ray)

        nearest_triangle = isect.nearest_triangle
        nearest_sphere = isect.nearest_sphere

        if nearest_triangle is None and nearest_sphere is None:
            break

        if nearest_triangle is None:
            nearest_object_material = nearest_sphere.material
        else:
            nearest_object_material = nearest_triangle.material

        min_distance = isect.min_distance
        intersection = isect.intersected_point
        surface_normal = isect.normal

        light = light + (nearest_object_material.emission * throughput)

        if bounce > 4:
            r_r = np.amax(nearest_object_material.color.diffuse)
            if np.random.random() >= r_r:
                break
            throughput = throughput/r_r

        bounce += 1

        light_ray_direction, light_ray_origin, pdf_light = sample_light_direction(scene.lights[0].source)
        light_ray = Ray(light_ray_origin, light_ray_direction, EPSILON)
        light_intersection, light_surface_normal, light_object_material = trace_light_path(scene, spheres, triangles, bvh, light_ray, bounce)

        if light_intersection is not None and light_surface_normal is not None and light_object_material is not None:
            path_contribution = connect_paths(scene, spheres, triangles, bvh,intersection, surface_normal, nearest_object_material, light_intersection, light_surface_normal, light_object_material, ray, light_ray, throughput, ONES)

            if np.any(path_contribution):
                light = light + path_contribution

        if nearest_object_material.type == MatType.DIFFUSE.value:
            direct_light = throughput * cast_one_shadow_ray(scene, spheres, triangles, bvh, nearest_object_material, intersection, surface_normal)
            light = light + direct_light

            new_ray_direction, pdf_fwd, brdf, intr_type = sample_diffuse(nearest_object_material, surface_normal, ray)

            throughput = throughput * (nearest_object_material.color.diffuse * brdf * np.abs(np.dot(new_ray_direction, surface_normal)) / pdf_fwd)

            ray = Ray(intersection, new_ray_direction, EPSILON)
            specular_bounce = False
            continue

        elif nearest_object_material.type == MatType.MIRROR.value:
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_mirror(nearest_object_material, surface_normal, ray)

            ray = Ray(intersection, new_ray_direction, EPSILON)
            specular_bounce = True
            continue

        elif nearest_object_material.type == MatType.SPECULAR.value:
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_specular(nearest_object_material, surface_normal, ray)

            if pdf_fwd != 0:
                throughput = throughput * brdf / pdf_fwd

            ray = Ray(intersection, new_ray_direction, EPSILON)
            specular_bounce = False
            continue

        else:
            break

    return light


@numba.njit(parallel=True)
def render_scene(scene, spheres, triangles, bvh):
    np.random.seed(79402371)

    eye = scene.camera.position
    look_at = scene.camera.look_at
    fov = scene.camera.fov
    cam_x = np.array([scene.width * fov / scene.height, 0.0, 0.0], dtype=np.float64)
    cam_y = normalize(np.cross(cam_x, look_at)) * fov

    h = scene.height
    w = scene.width
    spp = scene.number_of_samples

    for y in numba.prange(h):
        print(100.0 * y / (h - 1))
        for x in numba.prange(w):
            for sy in range(2):
                i = (h - 1 - y) * w + x
                for sx in range(2):
                    color = ZEROS
                    for s in numba.prange(spp):
                        u1 = 2.0 * np.random.random()
                        u2 = 2.0 * np.random.random()
                        dx = np.sqrt(u1) - 1.0 if u1 < 1 else 1.0 - np.sqrt(2.0 - u1)
                        dy = np.sqrt(u2) - 1.0 if u2 < 1 else 1.0 - np.sqrt(2.0 - u2)

                        cam_direction = cam_x * (((sx + 0.5 + dx) / 2.0 + x) / w - 0.5) + \
                                        cam_y * (((sy + 0.5 + dy) / 2.0 + y) / h - 0.5) + look_at
                        cam_origin = eye + cam_direction * 130
                        cam_direction = normalize(cam_direction)

                        cam_ray = Ray(cam_origin, cam_direction, 0)
                        color = color + trace_path(scene, spheres, triangles, bvh, cam_ray, 0)

                    color = color / spp
                    scene.image[y, x, :] = scene.image[y, x, :] + 0.25 * np.clip(color, 0, 1)

    return scene.image
