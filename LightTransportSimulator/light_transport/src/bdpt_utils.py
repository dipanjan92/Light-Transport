import math

import numba
import numpy as np

from LightTransportSimulator.light_transport.src.bvh_new import intersect_bvh
from LightTransportSimulator.light_transport.src.constants import Medium, inv_pi, ZEROS
from LightTransportSimulator.light_transport.src.rays import Ray
from LightTransportSimulator.light_transport.src.utils import get_cosine_hemisphere_pdf, \
    cosine_weighted_hemisphere_sampling, hit_object
from LightTransportSimulator.light_transport.src.vectors import normalize
from LightTransportSimulator.light_transport.src.vertex import Vertex, create_light_vertex, create_camera_vertex


@numba.njit
def sample_to_add_light(scene, spheres, triangles, bvh, vertices):
    random_light_index = np.random.choice(len(scene.lights), 1)[0]
    sampled_light = scene.lights[random_light_index]

    curr_v = vertices[-1]
    # check if the light is visible from the current vertex
    new_path_direction = normalize(sampled_light.source - curr_v.isec.intersected_point)
    new_path = Ray(curr_v.isec.intersected_point, new_path_direction)
    new_path_magnitude = np.linalg.norm(sampled_light.source - curr_v.isec.intersected_point)

    next_v = create_light_vertex(sampled_light, new_path_direction, new_path_magnitude, 0, 0)

    pdf = get_light_pdf(curr_v, next_v)
    light_choice_pdf = 1

    if pdf>0:
        # set other attributes
        next_v.pdf_dir = pdf
        next_v.color = sampled_light.material.color.diffuse
        # emission from the light source
        next_v.throughput = (sampled_light.material.emission/(pdf*light_choice_pdf))
        next_v.pdf_pos = 1/sampled_light.total_area
        next_v.pdf_fwd = light_choice_pdf*next_v.pdf_pos



        isect = hit_object(spheres, triangles, bvh, new_path)

        if isect.min_distance is None or isect.min_distance >= new_path_magnitude:
            # light is visible from current path
            next_v.throughput *=  next_v.throughput * curr_v.throughput # add the current path throughput

        if is_on_surface(curr_v):
            next_v.throughput *= np.abs(np.dot(new_path_direction, curr_v.isec.normal))

    # append the light vertex to the current path
    vertices.append(next_v)

    return vertices