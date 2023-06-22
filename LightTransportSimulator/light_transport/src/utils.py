import math
from typing import Union, Tuple

import numpy as np
import numba
from numba.misc.special import literal_unroll

from .bvh_new import intersect_bvh
from .constants import inv_2_pi, pi_over_4, pi_over_2, inv_pi
from .intersects import sphere_intersect, triangle_intersect, plane_intersect, __triangle_intersect, pc_triangle_intersect
from .primitives import Triangle, Sphere, Plane, ShapeOptions, Primitive, Intersection
from .vectors import normalize


@numba.njit
def nearest_intersected_object(objects, ray_origin, ray_direction, t0=0.0, t1=np.inf):
    """
    returns the nearest object that a ray intersects, if it exists
    :param objects: list of all objects
    :param ray_origin: origin of the ray (eg. camera)
    :param ray_end: pixel on the object
    :return: nearest object and the minimum distance to that object
    """
    # distances = []
    #
    # for obj in objects:
    #     if obj.type == ShapeOptions.SPHERE.value:
    #         dist = sphere_intersect(ray_origin, ray_direction, obj)
    #     else:
    #         dist = triangle_intersect(ray_origin, ray_direction, obj)
    #
    #     distances.append(dist)

        # distances.append(triangle_intersect(ray_origin, ray_direction, obj))

    distances = [triangle_intersect(ray_origin, ray_direction, obj) for obj in objects]

    # print(distances)

    nearest_object = None
    min_distance = np.inf #t1

    for index, distance in enumerate(distances):
        if distance is not None and distance < min_distance:
            if t0<distance<(t1-0.00001):
            # if not objects[index].is_light:
                min_distance = distance
                nearest_object = objects[index]

    return nearest_object, min_distance


@numba.njit
def intersect_spheres(ray, primitives):
    isec = None
    for i in range(len(primitives)): #numba.literal_unroll(range(len(primitives)))
        if primitives[i].intersect(ray):
            isec = primitives[i]
    return isec


@numba.njit
def hit_object(spheres, triangles, bvh, ray):

    nearest_triangle = None
    nearest_sphere = None

    if len(spheres)>0:
        nearest_sphere = intersect_spheres(ray, spheres)
        material = nearest_sphere.material
        if nearest_sphere is None:
            # no object was hit
            min_distance = None
            intersected_point = None
            normal = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        else:
            min_distance = ray.tmax
            intersected_point = ray.origin + min_distance * ray.direction
            normal = nearest_sphere.get_normal(intersected_point)

    if triangles is not None and len(triangles)>0:
        # nearest_triangle = intersect_bvh(ray, triangles, bvh)
        nearest_triangle = intersect_spheres(ray, triangles)

        if nearest_triangle is not None:
            nearest_sphere = None
            material = nearest_triangle.material
            min_distance = ray.tmax
            intersected_point = ray.origin + min_distance * ray.direction
            normal = nearest_triangle.get_normal(intersected_point)

    isect = Intersection(material, min_distance, intersected_point, normal)

    return isect


@numba.njit
def create_orthonormal_system(normal):
    if abs(normal[0]) > abs(normal[1]):
        v2 = np.array([-normal[2], 0.0, normal[0]], dtype=np.float64) / np.sqrt(np.array([normal[0] * normal[0] + normal[2] * normal[2]], dtype=np.float64))
    else:
        v2 = np.array([0.0, normal[2], -normal[1]], dtype=np.float64) / np.sqrt(np.array([normal[1] * normal[1] + normal[2] * normal[2]], dtype=np.float64))

    v3 = np.cross(normal, v2)

    return v2, v3


@numba.njit
def uniform_hemisphere_sampling(surface_normal):

    # w = surface_normal if np.dot(surface_normal, incoming_direction) < 0 else -surface_normal
    # u = normalize(np.cross(np.array([0.0, 1.0, 0.0], np.float64) if np.fabs(surface_normal[0]) > 0.1 else np.array([1.0, 0.0, 0.0], np.float64), w))
    # v = np.cross(surface_normal, u)

    rand = np.array([np.random.random(), np.random.random()], dtype=np.float64)
    z = rand[0]
    r = np.sqrt(np.max(0, 1 - z * z))
    phi = 2 * np.pi * rand[1]
    outgoing_direction = np.array([r * np.cos(phi), r * np.sin(phi), z], dtype=np.float64)
    # outgoing_direction = normalize(outgoing_direction[0] * u + outgoing_direction[1] * v + outgoing_direction[2] * w)

    pdf = inv_2_pi

    return outgoing_direction, pdf


@numba.njit
def concentric_sample_disk(u):
    u_offset = 2.0*u-np.array([1,1], dtype=np.float64)
    if u_offset[0] == 0 and u_offset[1] == 0:
        return np.array([0,0], dtype=np.float64)
    if abs(u_offset[0]) > abs(u_offset[1]):
        r = u_offset[0]
        theta = pi_over_4 * (u_offset[1] / u_offset[0])
    else:
        r = u_offset[1]
        theta = pi_over_2 - pi_over_4 * (u_offset[0] / u_offset[1])

    _point = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)

    return r * _point


@numba.njit
def sample_cosine_hemisphere(u):
    d = concentric_sample_disk(u)
    z = np.sqrt(max(0, 1 - d[0]**2 - d[1]**2))
    return np.array([d[0], d[1], z], dtype=np.float64)


@numba.njit
def get_cosine_hemisphere_pdf(cos_theta):
    return cos_theta*inv_pi


@numba.njit
def cosine_weighted_hemisphere_sampling(surface_normal, incoming_direction):

    if incoming_direction is not None:
        w = surface_normal if np.dot(surface_normal, incoming_direction) < 0 else -surface_normal
    else:
        w = surface_normal

    u = normalize(np.cross(np.array([0.0, 1.0, 0.0], np.float64) if np.fabs(surface_normal[0]) > 0.1 else np.array([1.0, 0.0, 0.0], np.float64), w))
    v = np.cross(surface_normal, u)

    rand = np.array([np.random.random(), np.random.random()], dtype=np.float64)
    outgoing_direction = sample_cosine_hemisphere(rand)
    outgoing_direction = normalize(outgoing_direction[0] * u + outgoing_direction[1] * v + outgoing_direction[2] * w)

    pdf = get_cosine_hemisphere_pdf(np.abs(outgoing_direction[2]))

    return outgoing_direction, pdf


@numba.njit
def _cosine_weighted_hemisphere_sampling(normal_at_intersection, incoming_direction, rand):
    incoming_direction = -incoming_direction
    # random uniform samples
    r1 = rand[0]
    r2 = rand[1]

    phi = 2*np.pi*r2
    theta = np.arccos(np.sqrt(r1))
    cos_theta = np.cos(theta)

    outgoing_direction = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), cos_theta], dtype=np.float64)

    v2, v3 = create_orthonormal_system(normal_at_intersection)

    # #TODO: Check if reversing z required
    if incoming_direction[2] < 0:
        outgoing_direction[2] *= -1

    # pdf = np.dot(global_ray_dir, normal_at_intersection)*inv_pi
    if incoming_direction[2] * outgoing_direction[2] > 0:
        pdf = np.abs(cos_theta)*inv_pi
    else:
        pdf = 0
    # pdf = np.abs(cos_theta)*inv_pi

    outgoing_direction = np.array([outgoing_direction[0] * v2[0] + outgoing_direction[1] * v3[0] + outgoing_direction[2] * normal_at_intersection[0],
                                   outgoing_direction[0] * v2[1] + outgoing_direction[1] * v3[1] + outgoing_direction[2] * normal_at_intersection[1],
                                   outgoing_direction[0] * v2[2] + outgoing_direction[1] * v3[2] + outgoing_direction[2] * normal_at_intersection[2]
                                   ], dtype=np.float64)

    return outgoing_direction, pdf
