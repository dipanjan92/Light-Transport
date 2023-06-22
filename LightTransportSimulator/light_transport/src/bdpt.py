import math

import numba
import numpy as np
import logging

from LightTransportSimulator.light_transport.src.brdf import get_reflected_direction, sample_diffuse, sample_mirror, \
    sample_specular, bxdf, get_bsdf_pdf
from LightTransportSimulator.light_transport.src.bvh_new import intersect_bvh
from LightTransportSimulator.light_transport.src.constants import EPSILON, inv_pi, Medium, ZEROS, TransportMode, ONES, \
    MatType
from LightTransportSimulator.light_transport.src.control_variates import calculate_dlogpdu
from LightTransportSimulator.light_transport.src.light_samples import sample_light, cast_one_shadow_ray
from LightTransportSimulator.light_transport.src.rays import Ray
from LightTransportSimulator.light_transport.src.utils import hit_object, cosine_weighted_hemisphere_sampling, \
    get_cosine_hemisphere_pdf
from LightTransportSimulator.light_transport.src.vectors import normalize, find_length, length_squared
from LightTransportSimulator.light_transport.src.vertex import Vertex, create_camera_vertex, create_surface_vertex, \
    create_light_vertex, get_vertex_color, get_vertex_emission, get_transmittance


@numba.njit
def random_walk(scene, spheres, triangles, bvh, ray, vertices, throughput, pdf, max_depth, mode):
    if max_depth==0:
        return vertices

    bounce = 0

    pdf_fwd = pdf
    pdf_rev = 0

    light = ZEROS
    specular_bounce = False

    while True:
        if np.array_equal(throughput, ZEROS):
            break

        # intersect ray with scene
        isect = hit_object(spheres, triangles, bvh, ray)

        # terminate path if no intersection is found
        if isect.min_distance is None or isect.material.type==MatType.NONE.value:
            # no object was hit
            if mode==TransportMode.RADIANCE.value:
                random_light_index = np.random.choice(len(scene.lights), 1)[0]
                l_source = scene.lights[random_light_index]
                light_vertex = create_light_vertex(l_source, ray, pdf_fwd)
                light_vertex.throughput = throughput
                vertices.append(light_vertex)
                bounce+= 1
            break

        # create new vertex with intersection info
        # if isect.material.emission>0:
        #     throughput = throughput*isect.material.emission

        current_vertex = create_surface_vertex(isect, ray, throughput, pdf_fwd, vertices[bounce-1])

        bounce += 1 # increment the bounce

        # terminate path if max depth is reached
        if bounce>=max_depth:
            break

        min_distance = isect.min_distance
        intersection = isect.intersected_point
        surface_normal = isect.normal
        nearest_object_material = isect.material

        # add emitted light at intersection
        # if bounce==0 or specular_bounce:
        # light = light + (nearest_object_material.emission * throughput)

        # print('P: ', intersection, ' C: ',  nearest_object.material.color.diffuse, ' L: ', light, ' F: ', throughput)

        # Russian roulette for variance reduction
        # if bounce> 4:
        #     r_r = np.amax(nearest_object_material.color.diffuse)
        #     if np.random.random() >= r_r:
        #         break
        #     throughput = throughput/r_r

        if nearest_object_material.type==MatType.DIFFUSE.value:
            # diffuse surface

            # Indirect Light
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_diffuse(nearest_object_material, surface_normal, ray)

            if pdf_fwd==0:
                break

            throughput = throughput * (brdf * np.abs(np.dot(new_ray_direction, surface_normal)) / pdf_fwd) * isect.material.color.diffuse

            ray = Ray(intersection, new_ray_direction, EPSILON)
            specular_bounce = False

        elif nearest_object_material.type==MatType.MIRROR.value:
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_mirror(nearest_object_material, surface_normal, ray)

            ray = Ray(intersection, new_ray_direction, EPSILON)
            specular_bounce = True

        elif nearest_object_material.type==MatType.SPECULAR.value:
            # specular reflection (only dielectric materials)
            new_ray_direction, pdf_fwd, brdf, intr_type = sample_specular(nearest_object_material, surface_normal, ray)

            if pdf_fwd!=0:
                throughput = throughput * brdf / pdf_fwd # update throughput

            ray = Ray(intersection, new_ray_direction, EPSILON)
            specular_bounce = False

        else:
            # error in material metadata or light
            # pdf_rev = pdf_fwd = 0
            # vertices[bounce-1].pdf_rev = current_vertex.convert_density(pdf_rev, vertices[bounce-1])
            # vertices.append(current_vertex)
            break

        if specular_bounce:
            current_vertex.is_delta = True
            pdf_rev = pdf_fwd = 0
        else:
            pdf_rev = get_bsdf_pdf(new_ray_direction, ray.direction) # n.b. reverse order

        # Compute reverse area density at preceding vertex
        vertices[bounce-1].pdf_rev = current_vertex.convert_density(pdf_rev, vertices[bounce-1])

        # add current vertex to the list of vertices
        vertices.append(current_vertex)

    return vertices


@numba.njit
def generate_light_subpaths(scene, bvh, spheres, triangles, max_depth):
    # print("light subpaths...")
    light_vertices = numba.typed.List() # will contain the vertices on the path starting from light

    if max_depth==0:
        return light_vertices

    # sample initial light ray
    random_light_index = np.random.choice(len(scene.lights), 1)[0]
    light = scene.lights[random_light_index]
    Le, light_ray, pdf_pos, pdf_dir = light.sample_source()

    if pdf_pos==0  or pdf_dir==0 or np.array_equal(Le, ZEROS):
        return light_vertices

    light_pdf = 1 #only source

    light_vx = create_light_vertex(light, light_ray, pdf_pos*light_pdf)
    light_vx.throughput = Le * np.abs(np.dot(light.normal, light_ray.direction))/ (light_pdf*pdf_pos*pdf_dir) * ONES

    # add the first vertex: light source
    light_vertices.append(light_vx)

    # start random walk
    light_vertices = random_walk(scene,
                                 spheres,
                                 triangles,
                                 bvh,
                                 light_ray,
                                 light_vertices,
                                 light_vx.throughput,
                                 light_vx.pdf_dir,
                                 max_depth-1,
                                 TransportMode.IMPORTANCE.value
                                 )

    return light_vertices


@numba.njit
def generate_camera_subpaths(scene, bvh, spheres, triangles, ray, max_depth):
    camera_vertices = numba.typed.List() # will contain the vertices on the path starting from camera

    if max_depth==0:
        return camera_vertices

    # print("camera subpaths...")

    pdf_pos, pdf_dir = scene.camera.get_pdf(ray)

    throughput = ONES # 1 for simple camera model, otherwise a floating-point value that affects how much
    # the radiance arriving at the film plane along the generated ray will contribute to the final image.

    # camera is starting vertex for backward-path-tracing
    cam_vertex = create_camera_vertex(scene.camera, ray, throughput, pdf_pos, pdf_dir)

    camera_vertices.append(cam_vertex)

    camera_vertices = random_walk(scene,
                                  spheres,
                                  triangles,
                                  bvh,
                                  ray,
                                  camera_vertices,
                                  throughput,
                                  cam_vertex.pdf_dir,
                                  max_depth-1,
                                  TransportMode.RADIANCE.value
                                  )

    return camera_vertices


@numba.njit
def get_mis_weight(scene, light_vertices, camera_vertices, sampled, s, t):
    if s+t == 2:
        return 1

    sum_ri = 0

    re_map = lambda f: f if f != 0 else 1 # to avoid divide by 0

    # Temporarily update vertex properties for current strategy
    # Look up connection vertices and their predecessors

    qs = light_vertices[s - 1] if s > 0 else None
    pt = camera_vertices[t - 1] if t > 0 else None
    qs_minus = light_vertices[s - 2] if s > 1 else None
    pt_minus = camera_vertices[t - 2] if t > 1 else None

    # Update sampled vertex for s=1 or t=1 strategy
    if s == 1:
        qs = sampled
    elif t == 1:
        pt = sampled

    if pt is not None:
        pt.is_delta = False
    if qs is not None:
        qs.is_delta = False

    # Update reverse density of vertex pt_{t-1}
    if pt is not None:
        if s>0:
            pt.pdf_rev = qs.get_pdf(scene, qs_minus, pt)
        else:
            pt.pdf_rev = pt.get_light_origin_pdf(scene, pt_minus)

    # Update reverse density of vertex pt_{t-2}
    if pt_minus is not None:
        if s>0:
            pt_minus.pdf_rev = pt.get_pdf(scene, qs, pt_minus)
        else:
            pt_minus.pdf_rev = pt.get_light_pdf(scene, pt_minus)

    # Update reverse density of vertices qs_{s-1} and qs_{s-2}
    if qs is not None:
        qs.pdf_rev = pt.get_pdf(scene, pt_minus, qs)

    if qs_minus is not None:
        qs_minus.pdf_rev = qs.get_pdf(scene, pt, qs_minus)

    # Consider hypothetical connection strategies along the camera subpath
    ri = 1
    for i in range(t - 1, 0, -1):
        ri *= re_map(camera_vertices[i].pdf_rev) / re_map(camera_vertices[i].pdf_fwd)
        if not camera_vertices[i].is_delta:
            sum_ri += ri

    # Consider hypothetical connection strategies along the light subpath
    ri = 1
    for i in range(s - 1, -1, -1):
        ri *= re_map(light_vertices[i].pdf_rev) / re_map(light_vertices[i].pdf_fwd)
        if not light_vertices[i].is_delta:
            sum_ri += ri

    weight = 1/(1+sum_ri)

    # print('MIS_weight= ', weight)

    return weight


@numba.njit
def G(vertex_0, vertex_1, spheres, triangles, bvh):
    d = vertex_0.isec.intersected_point-vertex_1.isec.intersected_point
    g = 1/length_squared(d)
    d = d*np.sqrt(g)

    if vertex_0.is_on_surface():
        g *= np.abs(np.dot(vertex_0.isec.normal, d))
    if vertex_1.is_on_surface():
        g *= np.abs(np.dot(vertex_1.isec.normal, d))

    tr = 1.0 #get_transmittance(vertex_0.isec, vertex_1.isec, spheres, triangles, bvh)

    return g*tr



@numba.njit
def connect_paths(scene, spheres, triangles, bvh, camera_vertices, light_vertices, s, t):
    light = ZEROS
    sampled = None

    # check for invalid connections
    if t > 1 and s != 0 and camera_vertices[t - 1].type == Medium.LIGHT.value:
        return light


    if s==0:
        # camera subpath is the entire path
        pt = camera_vertices[t-1]
        # print("should found light: s:-", s, ' t:-', t)
        if pt.type == Medium.LIGHT.value:
            # print("did found light: s:-", s, ' t:-', t)
            light = pt.get_light_contribution(scene, camera_vertices[t-2]) * pt.throughput


    elif t==1:
        # connect camera to a light subpath
        qs = light_vertices[s-1]

        if qs.is_connectible():
            # connect camera to the light subpath
            Wi, new_path, pdf, visible = scene.camera.sample_incident(qs.isec, spheres, triangles, bvh)

            if pdf>0 and Wi!=0:
                pdf_pos = 1
                throughput = (Wi/pdf)*ONES
                sampled = create_camera_vertex(scene.camera, new_path, throughput, pdf_pos, pdf)
                light = qs.throughput * qs.f(sampled) * sampled.throughput
                if qs.is_on_surface():
                    light = light * np.abs(np.dot(new_path.direction, qs.isec.normal))


    elif s==1:
        # connect the camera subpath to a light source
        pt = camera_vertices[t-1]

        if pt.is_connectible():
            # sample a point on the light
            random_light_index = np.random.choice(len(scene.lights), 1)[0]
            sampled_light = scene.lights[random_light_index]

            Li, new_path, pdf, visible = sampled_light.sample_incident(pt.isec, spheres, triangles, bvh)

            total_lights = 1
            light_choice_pdf = 1/total_lights

            if pdf>0 and not np.array_equal(Li, ZEROS):
                sampled = create_light_vertex(sampled_light, new_path, 0)
                sampled.throughput = Li/(pdf*light_choice_pdf) * pt.throughput
                sampled.pdf_pos, sampled.pdf_dir = sampled_light.get_pdf(new_path, sampled.isec.normal)
                sampled.pdf_fwd = sampled.get_light_origin_pdf(scene, pt)

                light = pt.f(sampled) * sampled.throughput

                if pt.is_on_surface():
                    light = light * np.abs(np.dot(new_path.direction, pt.isec.normal))


    else:
        # follow rest of the strategies
        qs = light_vertices[s-1]
        pt = camera_vertices[t-1]

        if qs.is_connectible() and pt.is_connectible():

            light = qs.throughput * qs.f(pt) * pt.f(qs) * pt.throughput
            if not np.array_equal(light, ZEROS):
                light *= G(qs, pt, spheres, triangles, bvh)


    # compute MIS-weights for the above connection strategies
    if np.array_equal(light, ZEROS):
        mis_weight = 0.0
    else:
        mis_weight = get_mis_weight(scene, light_vertices, camera_vertices, sampled, s, t)

    # print('mis_weight: ', mis_weight, ', light: ', light)

    light = light * mis_weight

    return light


@numba.njit(parallel=True)
def render_scene(scene, spheres, triangles, bvh):
    np.random.seed(79402371)

    eye = scene.camera.position
    look_at = scene.camera.look_at

    fov =  scene.camera.fov # field of view

    cam_x = np.array([scene.width * fov / scene.height, 0.0, 0.0], dtype=np.float64)
    cam_y = normalize(np.cross(cam_x, look_at)) * fov

    if scene.camera.normal is None:
        scene.camera.normal = normalize(np.cross(cam_x, cam_y))

    if scene.camera.screen_area is None:
        scene.camera.screen_area = scene.width * scene.height

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
                    splat = ZEROS
                    for s_i in numba.prange(spp):
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


                        camera_vertices = generate_camera_subpaths(scene, bvh, spheres, triangles, cam_ray, scene.max_depth+2)
                        light_vertices = generate_light_subpaths(scene, bvh, spheres, triangles, scene.max_depth+1)

                        camera_n = len(camera_vertices)
                        light_n = len(light_vertices)

                        # print("n_camera: ", camera_n, " n_light: ", light_n)

                        for t in range(1, camera_n+1):
                            for s in range(light_n+1):
                                depth = t+s-2
                                if (s == 1 and t == 1) or depth<0 or depth > scene.max_depth:
                                    continue

                                Lp = connect_paths(scene, spheres, triangles, bvh, camera_vertices, light_vertices, s, t)

                                if t!=1:
                                    color = color + Lp
                                else:
                                    splat = Lp




                        # color = color + trace_path(scene, spheres, triangles, bvh, cam_ray, 0)

                    color = color/spp + splat
                    scene.image[y, x, :] = scene.image[y, x, :] + 0.25 * np.clip(color, 0, 1)

    return scene.image

