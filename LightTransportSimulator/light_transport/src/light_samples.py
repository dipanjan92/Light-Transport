import math

import numba
import numpy as np

from LightTransportSimulator.light_transport.src.bvh_new import intersect_bvh
from LightTransportSimulator.light_transport.src.constants import inv_pi, EPSILON, Medium, ZEROS
from LightTransportSimulator.light_transport.src.rays import Ray
from LightTransportSimulator.light_transport.src.scene import Light
from LightTransportSimulator.light_transport.src.utils import nearest_intersected_object, uniform_hemisphere_sampling, \
    cosine_weighted_hemisphere_sampling, create_orthonormal_system, sample_cosine_hemisphere, get_cosine_hemisphere_pdf, \
    hit_object
from LightTransportSimulator.light_transport.src.vectors import normalize
from LightTransportSimulator.light_transport.src.vertex import Vertex, create_light_vertex


def generate_area_light_samples(tri_1, tri_2, source_mat, number_of_samples, total_area):
    light_sources = numba.typed.List()

    light_samples = number_of_samples
    a = np.random.uniform(0,1,light_samples)
    b = np.random.uniform(1,0,light_samples)

    for x in range(light_samples):
        tp1 = tri_1.vertex_1 * (1-math.sqrt(a[x])) + tri_1.vertex_2 * (math.sqrt(a[x])*(1-b[x])) + tri_1.vertex_3 * (b[x]*math.sqrt(a[x]))
        l1 = Light(source=tp1, material=source_mat, normal=tri_1.normal, total_area=total_area)
        light_sources.append(l1)
        tp2 = tri_2.vertex_1 * (1-math.sqrt(a[x])) + tri_2.vertex_2 * (math.sqrt(a[x])*(1-b[x])) + tri_2.vertex_3 * (b[x]*math.sqrt(a[x]))
        l2 = Light(source=tp2, material=source_mat, normal=tri_2.normal, total_area=total_area)
        light_sources.append(l2)

    return light_sources




@numba.njit
def cast_one_shadow_ray(scene, spheres, triangles, bvh, intersected_object_material, intersection_point, intersection_normal):
    light_contrib = np.zeros((3), dtype=np.float64)
    random_light_index = np.random.choice(len(scene.lights), 1)[0]
    light = scene.lights[random_light_index]

    shadow_ray_direction = normalize(light.source - intersection_point)
    shadow_ray_magnitude = np.linalg.norm(light.source - intersection_point)
    shadow_ray = Ray(intersection_point, shadow_ray_direction, EPSILON)

    isect = hit_object(spheres, triangles, bvh, shadow_ray)
    min_distance = isect.min_distance

    if min_distance is None:
        return light_contrib # black background- unlikely

    # visible = min_distance >= shadow_ray_magnitude-EPSILON
    visible = isect.material.emission>0
    if visible:
        brdf = (light.material.emission * light.material.color.diffuse) * (intersected_object_material.color.diffuse * inv_pi)
        cos_theta = np.dot(intersection_normal, shadow_ray_direction)
        cos_phi = np.dot(light.normal, -shadow_ray_direction)
        geometry_term = np.abs(cos_theta * cos_phi)/(shadow_ray_magnitude * shadow_ray_magnitude)
        light_contrib += brdf * geometry_term * light.total_area
        # print('from light: ', light.material.color.diffuse, intersected_object_material.color.diffuse, light.material.emission)

    return light_contrib



@numba.njit
def sample_light(scene):
    random_light_index = np.random.choice(len(scene.lights), 1)[0]
    light = scene.lights[random_light_index]

    # generate a random ray and compute its pdf
    direction, origin, pdf_dir = sample_light_direction(light.source)

    light_ray = Ray(origin, direction, 0)

    light_pdf = 1 # 1/no_of_lights
    pdf_pos = 1/light.total_area

    # create light vertex
    light_vertex = create_light_vertex(light, light_ray, pdf_pos*light_pdf)

    if np.dot(light.normal, light_ray.direction)>0:
        light_vertex.throughput = light.material.emission*light.material.color.diffuse
    else:
        light_vertex.throughput = ZEROS

    throughput = (light_vertex.throughput*np.abs(np.dot(light_vertex.isec.normal, light_ray.direction)))/(light_pdf*light_vertex.pdf_pos*light_vertex.pdf_dir)

    light_vertex.throughput = throughput
    light_vertex.pdf_pos = pdf_pos
    light_vertex.pdf_dir = pdf_dir

    return light_ray, light_vertex, throughput


@numba.njit
def cast_all_shadow_rays(scene, bvh, intersected_object, intersection_point, intersection_normal):
    light_contrib = np.zeros((3), dtype=np.float64)
    for light in scene.lights:
        shadow_ray_direction = normalize(light.source - intersection_point)
        shadow_ray_magnitude = np.linalg.norm(light.source - intersection_point)
        shadow_ray = Ray(intersection_point, shadow_ray_direction)

        _objects = traverse_bvh(bvh, shadow_ray)
        _, min_distance = nearest_intersected_object(_objects, intersection_point, shadow_ray_direction, t1=shadow_ray_magnitude)

        if min_distance is None:
            break

        visible = min_distance > shadow_ray_magnitude
        if visible:
            brdf = (light.material.emission * light.material.color.diffuse) * (intersected_object.material.color.diffuse * inv_pi)
            cos_theta = np.dot(intersection_normal, shadow_ray_direction)
            cos_phi = np.dot(light.normal, -shadow_ray_direction)
            geometry_term = np.abs(cos_theta * cos_phi)/(shadow_ray_magnitude * shadow_ray_magnitude)
            light_contrib += brdf * geometry_term * light.total_area

    light_contrib = light_contrib/len(scene.lights)

    return light_contrib


@numba.njit
def sample_point_on_light(light_source):
    # Assume the disk light source is represented by a sphere object
    # with only the disk part visible as the light source within the Cornell Box.

    # Sample a random point on the disk surface
    u, v = np.random.uniform(size=2)
    theta = 2 * np.pi * u
    radius = light_source.radius
    position = light_source.source + radius * np.array([np.cos(theta), np.sin(theta), 0.0])

    # Calculate the surface normal of the disk (aligned with the z-axis in this case)
    normal = light_source.normal #np.array([0.0, 0.0, 1.0])

    # Calculate the probability density function (PDF) for sampling the point on the disk
    pdf = 1.0 / light_source.total_area

    return position, normal, pdf


@numba.njit
def sample_light_direction(light_point):
    # Assume the disk light source emits light uniformly in all directions.

    # Sample a random direction on the unit hemisphere
    u, v = np.random.random(size=2)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)

    # Calculate the direction vector
    direction = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])

    # Calculate the probability density function (PDF) for sampling the direction
    pdf_dir = 1.0 / (2 * np.pi)

    return direction, light_point, pdf_dir




