import numpy as np
import numba

from .brdf import sample_diffuse, sample_mirror, sample_specular, get_bsdf_pdf, oren_nayar_f, schlick_reflectance
from .primitives import Intersection
from .rays import Ray
from .utils import hit_object
from .vectors import normalize, length_squared
from .constants import Medium, ONES, ZEROS, MatType, EPSILON


@numba.experimental.jitclass([
    ('type', numba.intp),
    ('isec', Intersection.class_type.instance_type),
    ('ray', Ray.class_type.instance_type),
    ('throughput', numba.float64[:]),
    ('brdf', numba.float64),
    ('pdf_fwd', numba.float64),
    ('pdf_rev', numba.float64),
    ('pdf_pos', numba.float64),
    ('pdf_dir', numba.float64),
    ('is_delta', numba.boolean)
])
class Vertex:
    def __init__(self, isec, ray):
        self.type = Medium.NONE.value
        self.isec = isec
        self.ray = ray
        self.throughput = ONES
        self.brdf = 0.0
        self.pdf_fwd = 0.0
        self.pdf_rev = 0.0
        self.pdf_pos = 0.0
        self.pdf_dir = 0.0
        self.is_delta = False

    def get_light_contribution(self, scene, next_v):

        w = next_v.isec.intersected_point - self.isec.intersected_point

        if length_squared(w) == 0:
            return 0

        w = normalize(w)

        random_light_index = np.random.choice(len(scene.lights), 1)[0]
        light = scene.lights[random_light_index]

        if np.dot(light.normal, w)>0:
            Le = light.material.emission
        else:
            print("Le...0")
            Le = 0

        return Le

    def is_connectible(self):
        if self.type==Medium.LIGHT.value or self.type==Medium.CAMERA.value:
            # Assuming all lights are area lights, hence no delta light
            return True
        elif self.type==Medium.SURFACE.value and not self.is_delta and self.isec.material.type!=MatType.MIRROR.value:
            return True
        else:
            return False

    def is_on_surface(self):
        return self.type==Medium.SURFACE.value

    def f(self, next_v):
        wi = normalize(next_v.isec.intersected_point - self.isec.intersected_point)
        if self.type==Medium.SURFACE.value:
            if self.isec.material.type==MatType.DIFFUSE.value:
                brdf = oren_nayar_f(self.ray.direction, wi)
            elif self.isec.material.type==MatType.MIRROR.value:
                if np.dot(wi, self.isec.normal) == np.dot(self.ray.direction, self.isec.normal):
                    # perfect mirror reflection
                    return 1
                else:
                    return 0
            elif self.isec.material.type==MatType.SPECULAR.value:
                reflect = np.dot(self.ray.direction, self.isec.normal) * np.dot(wi, self.isec.normal) > 0

                out_to_in = np.dot(self.isec.normal, self.ray.direction) < 0
                nl = self.isec.normal if out_to_in else -self.isec.normal
                cos_theta = np.dot(self.ray.direction, nl)

                n_out = 1
                n_in = self.isec.material.ior

                eta = n_out / n_in if out_to_in else n_in / n_out
                cos2_phi = 1.0 - eta * eta * (1.0 - cos_theta * cos_theta)
                if cos2_phi < 0:
                    #TIR
                    return 1

                c = 1.0 - (-cos_theta if out_to_in else np.dot(wi, self.isec.normal))

                Re = schlick_reflectance(n_out, n_in, c)

                if reflect:
                    return Re
                else:
                    return 1-Re
            else:
                brdf = 0
            return brdf
        else:
            return 0 # TODO: check for lights

    def convert_density(self, pdf, next_v):
        w = next_v.isec.intersected_point - self.isec.intersected_point

        if length_squared(w) == 0:
            return 0

        inv_dist_sqr = 1/length_squared(w)

        if next_v.is_on_surface():
            pdf *= np.abs(np.dot(next_v.isec.normal, w*np.sqrt(inv_dist_sqr)))

        return pdf*inv_dist_sqr

    def get_pdf(self, scene, pre_v, next_v):
        if self.type == Medium.LIGHT.value:
            return self.get_light_pdf(scene, next_v)

        # Compute directions to preceding and next vertex
        wn = next_v.isec.intersected_point-self.isec.intersected_point
        if length_squared(wn)==0:
            return 0
        wn = normalize(wn)

        if pre_v is not None and pre_v.type!=Medium.NONE.value:
            wp = pre_v.isec.intersected_point-self.isec.intersected_point
            if length_squared(wp)==0:
                return 0
            wp = normalize(wp)

        # Compute directional density depending on the vertex type

        if self.type==Medium.CAMERA.value:
            new_ray = Ray(self.isec.intersected_point, wn, EPSILON)
            pdf_pos, pdf = scene.camera.get_pdf(new_ray)
        elif self.type==Medium.SURFACE.value:
            pdf = get_bsdf_pdf(wp, wn)
            pdf = self.convert_density(pdf, next_v)
        else:
            pdf = 0

        return pdf

    def get_light_pdf(self, scene, vx):
        w = vx.isec.intersected_point - self.isec.intersected_point
        inv_dist_sqr = 1/length_squared(w)
        w = w*np.sqrt(inv_dist_sqr)
        random_light_index = np.random.choice(len(scene.lights), 1)[0]
        light = scene.lights[random_light_index]
        new_ray = Ray(self.isec.intersected_point, w, EPSILON)
        pdf_pos, pdf_dir = light.get_pdf(new_ray, vx.isec.normal)
        pdf = pdf_dir*inv_dist_sqr
        return pdf

    def get_light_origin_pdf(self, scene, vx):
        w = vx.isec.intersected_point - self.isec.intersected_point
        if length_squared(w)==0:
            return 0
        w = normalize(w)
        random_light_index = np.random.choice(len(scene.lights), 1)[0]
        light = scene.lights[random_light_index]
        total_lights = 1
        light_choice_pdf = 1/total_lights
        new_ray = Ray(self.isec.intersected_point, w, EPSILON)
        pdf_pos, pdf_dir = light.get_pdf(new_ray, vx.isec.normal)
        pdf = pdf_pos*light_choice_pdf
        return pdf




@numba.njit
def create_camera_vertex(camera, ray, throughput, pdf_pos, pdf_dir):
    isec = Intersection(None, None, camera.position, camera.normal)
    vx = Vertex(isec, ray)
    vx.type = Medium.CAMERA.value
    vx.throughput = throughput
    vx.pdf_pos = pdf_pos
    vx.pdf_dir = pdf_dir
    return vx


@numba.njit
def create_light_vertex(light, ray, pdf_fwd):
    isec = Intersection(light.material, None, light.source, light.normal)
    light_v = Vertex(isec, ray)
    light_v.type = Medium.LIGHT.value
    light_v.pdf_fwd = pdf_fwd
    return light_v


@numba.njit
def create_surface_vertex(isec, ray, throughput, pdf_fwd, prev_v):
    surface = Vertex(isec, ray)
    surface.type = Medium.SURFACE.value
    surface.throughput = throughput
    surface.pdf_fwd = surface.convert_density(pdf_fwd, prev_v)
    return surface


@numba.njit
def get_vertex_color(vx):
    if vx.type==Medium.SURFACE.value or vx.type==Medium.LIGHT.value:
        return vx.isec.material.color.diffuse
    else:
        return ONES

@numba.njit
def get_vertex_emission(vx):
    if vx.type==Medium.LIGHT.value:
        return vx.isec.material.emission
    else:
        return 1

@numba.njit
def get_transmittance(p0, p1, spheres, triangles, bvh):
    r = spawn_ray(p0, p1)
    tr = 1.0
    # while True:
    #     isec = hit_object(spheres, triangles, bvh, r)
    #     # n.b. no medium interaction
    #     if isec.intersected_point is None:
    #         break
    #     if isec.material is not None and isec.material.type != MatType.NONE.value:
    #         return 0
    #     r = spawn_ray(isec, p1)
    return tr


@numba.njit
def spawn_ray(p0, p1):
    d = normalize(p1.intersected_point - p0.intersected_point)
    r = Ray(p0.intersected_point, d, EPSILON)
    return r

