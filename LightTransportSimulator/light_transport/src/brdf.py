import numba
import numpy as np
from numba import jit

from LightTransportSimulator.light_transport.src.constants import inv_pi, ZEROS, Medium, TransportMode, EPSILON, MatType
from LightTransportSimulator.light_transport.src.utils import cosine_weighted_hemisphere_sampling, \
    get_cosine_hemisphere_pdf
from LightTransportSimulator.light_transport.src.vectors import normalize


@jit(nopython=True)
def get_reflected_direction(vector, axis):
    return normalize(vector - 2 * np.dot(vector, axis) * axis)


@jit(nopython=True)
def get_ambience(amb_obj, amb_light):
    """
    illumination model for ambient BRDF
    :param amb_obj: ambient coefficient of an object
    :param amb_light: ambient intensity of a light source
    :return: illumination factor for ambience
    """
    return amb_obj * amb_light


@jit(nopython=True)
def get_diffuse(diff_obj, diff_light, light_vec, surf_norm, shiny_fact=1):
    """
    illumination model for diffuse BRDF
    :param diff_obj: diffuse coefficient of an abject
    :param diff_light: diffuse coefficient of a light source
    :param light_vec: a vector pointing towards light
    :param surf_norm: surface normal
    :param shiny_fact: shininess factor
    :return: illumination factor for diffuse
    """
    return diff_obj * diff_light * (np.dot(light_vec, surf_norm)**shiny_fact)


@jit(nopython=True)
def get_specular(spec_obj, spec_light, view_dir, surf_norm, shiny_fact):
    """
    illumination model for specular BRDF
    :param spec_obj: specular coefficient of an object
    :param spec_light: spec
    :param view_dir:
    :param surf_norm:
    :param shiny_fact:
    :return: illumination factor for specular
    """
    return spec_obj * spec_light * (np.dot(surf_norm, view_dir) ** (shiny_fact / 4))


@numba.njit
def fresnel_dielectric(cos_theta_i, n1, n2):
    eta = n1 / n2
    sin2_theta_t = eta**2 * (1 - cos_theta_i**2)
    if sin2_theta_t > 1:
        fr = 1
        cos_theta_t = 0
    else:
        cos_theta_t = np.sqrt(1 - sin2_theta_t)
        fr = 0.5 * ((n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t))**2 + \
                           0.5 * ((n1 * cos_theta_t - n2 * cos_theta_i) / (n1 * cos_theta_t + n2 * cos_theta_i))**2

    return fr, sin2_theta_t


@numba.njit
def oren_nayar_f(old_ray, new_ray):
    sigma = np.radians(0.25)
    sigma2 = sigma * sigma
    A = 1 - (sigma2 / (2 * (sigma2 + 0.33)))
    B = 0.45 * sigma2 / (sigma2 + 0.09)

    sin_theta_n = new_ray[1]
    sin_theta_o = old_ray[1]

    # Compute cosine term of Oren-Nayar model
    max_cos = 0
    if sin_theta_n > EPSILON and sin_theta_o > EPSILON:
        sin_phi_n = new_ray[0] / sin_theta_n
        cos_phi_n = new_ray[2] / sin_theta_n
        sin_phi_o = old_ray[0] / sin_theta_o
        cos_phi_o = old_ray[2] / sin_theta_o
        d_cos = cos_phi_n * cos_phi_o + sin_phi_n * sin_phi_o
        max_cos = max(0, d_cos)

    # Compute sine and tangent terms of Oren-Nayar model
    sin_alpha, tan_beta = 0, 0
    if abs(new_ray[2]) > abs(old_ray[2]):
        sin_alpha = sin_theta_o
        tan_beta = sin_theta_n / np.abs(new_ray[2])
    else:
        sin_alpha = sin_theta_n
        tan_beta = sin_theta_o / np.abs(old_ray[2])

    return  inv_pi * (A + B * max_cos * sin_alpha * tan_beta)


@numba.njit
def sample_diffuse(nearest_object_material, surface_normal, ray):
    new_ray_direction, pdf_fwd = cosine_weighted_hemisphere_sampling(surface_normal, ray.direction)
    # new_ray_direction = normalize(new_ray_direction)
    brdf = 1 * oren_nayar_f(ray.direction, new_ray_direction)
    intr_type = MatType.DIFFUSE.value
    return new_ray_direction, pdf_fwd, brdf, intr_type


@numba.njit
def sample_mirror(nearest_object_material, surface_normal, ray):
    new_ray_direction = get_reflected_direction(ray.direction, surface_normal)
    pdf_fwd = 1
    # fr, _ = fresnel_dielectric(np.abs(new_ray_direction[2]), 1, nearest_object_material.ior)
    # brdf =  fr * 1/np.abs(new_ray_direction[2]) # reflectance is 1 for perfect mirrors
    intr_type = MatType.MIRROR.value
    return new_ray_direction, pdf_fwd, 1, intr_type


@numba.njit
def reflectance0(n1, n2):
    sqrt_R0 = np.float64(n1 - n2) / (n1 + n2)
    return sqrt_R0 * sqrt_R0


@numba.njit
def schlick_reflectance(n1, n2, c):
    R0 = reflectance0(n1, n2)
    return R0 + (1.0 - R0) * c * c * c * c * c


@numba.njit
def sample_specular(nearest_object_material, surface_normal, ray):
    d = ray.direction
    n = surface_normal

    n_out = 1
    n_in = nearest_object_material.ior
    n_out, n_in = np.float64(n_out), np.float64(n_in)

    reflected_direction = get_reflected_direction(d, n)

    # switch n if ray inside object
    out_to_in = np.dot(n, d) < 0
    nl = n if out_to_in else -n
    eta = n_out / n_in if out_to_in else n_in / n_out

    cos_theta = np.dot(d, nl)
    cos2_phi = 1.0 - eta * eta * (1.0 - cos_theta * cos_theta)

    # Total Internal Reflection
    if cos2_phi < 0:
        intr_type = MatType.TIR.value
        pdf = 0
        brdf = 1.0
        return reflected_direction, pdf, 1.0, intr_type

    transmitted_direction = normalize(eta * d - nl * (eta * cos_theta + np.sqrt(cos2_phi)))
    c = 1.0 - (-cos_theta if out_to_in else np.dot(transmitted_direction, n))

    Re = schlick_reflectance(n_out, n_in, c)
    p_Re = 0.25 + 0.5 * Re

    if np.random.random() < p_Re:
        # reflection
        intr_type = MatType.REFLECTION.value
        pdf = p_Re
        brdf = Re
        return reflected_direction, pdf, brdf, intr_type
    else:
        # refraction
        intr_type = MatType.REFRACTION.value
        pdf = 1.0 - p_Re
        brdf = 1 - Re
        return transmitted_direction, pdf, brdf, intr_type


@numba.njit
def bxdf(prev_v, next_v):
    # camera or light endpoints should not arrive here
    w = normalize(next_v.isec.intersected_point - prev_v.isec.intersected_point)
    if prev_v.type==Medium.SURFACE.value:
        if prev_v.isec.material.type==MatType.DIFFUSE.value:
            return oren_nayar_f(prev_v.ray.direction, w)
        else:
            # for all other specular events
            return 0
    else:
        return 0


@numba.njit
def get_bsdf_pdf(wp, wn):
    if wp is None:
        return get_cosine_hemisphere_pdf(np.abs(wn[2]))
    elif wp[2]*wn[2]>0:
        return get_cosine_hemisphere_pdf(np.abs(wn[2]))
    else:
        return 0
