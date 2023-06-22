import enum

import numpy as np

from .material import Color, Material

inv_pi = 1/np.pi
inv_2_pi = 0.5*inv_pi
inv_4_pi = 0.25*inv_pi
pi_over_2 = np.pi/2
pi_over_4 = 0.5*pi_over_2
EPSILON = 0.000001

ZEROS = np.zeros((3), dtype=np.float64)
ONES = np.ones((3), dtype=np.float64)

class Medium(enum.Enum):
    NONE = 0
    CAMERA = 1
    LIGHT = 2
    SURFACE = 3


class MatType(enum.Enum):
    DIFFUSE = 1
    MIRROR = 2
    SPECULAR = 3
    REFLECTION = 4
    REFRACTION = 5
    TIR = 6
    NONE = 0


class TransportMode(enum.Enum):
    RADIANCE = 1
    IMPORTANCE = 2
    NONE = 0


# WHITE = Color(ambient=np.array([1, 1, 1], dtype=np.float64),
#               diffuse=np.array([1, 1, 1], dtype=np.float64),
#               specular=np.array([1, 1, 1], dtype=np.float64))
#
# WHITE_2 = Color(ambient=np.array([0, 0, 0], dtype=np.float64),
#               diffuse=np.array([0.55, 0.55, 0.55], dtype=np.float64),
#               specular=np.array([0.7, 0.7, 0.7], dtype=np.float64))
#
# RED = Color(ambient=np.array([0.1, 0, 0], dtype=np.float64),
#             diffuse=np.array([0.7, 0, 0], dtype=np.float64),
#             specular=np.array([1, 1, 1], dtype=np.float64))
#
# LEFT = Color(ambient=np.array([0.1, 0, 0], dtype=np.float64),
#             diffuse=np.array([10, 2, 2], dtype=np.float64),
#             specular=np.array([1, 1, 1], dtype=np.float64))
#
# PURPLE = Color(ambient=np.array([0.1, 0, 0.1], dtype=np.float64),
#             diffuse=np.array([0.7, 0, 0.7], dtype=np.float64),
#             specular=np.array([1, 1, 1], dtype=np.float64))
#
# YELLOW = Color(ambient=np.array([0.05, 0.05, 0.0], dtype=np.float64),
#                diffuse=np.array([0.5, 0.5, 0.4], dtype=np.float64),
#                specular=np.array([0.7, 0.7, 0.04], dtype=np.float64))
#
#
# SILVER = Color(ambient=np.array([0.23125, 0.23125, 0.23125], dtype=np.float64),
#                diffuse=np.array([0.2775, 0.2775, 0.2775], dtype=np.float64),
#                specular=np.array([0.773911, 0.773911, 0.773911], dtype=np.float64))
#
# GREEN = Color(ambient=np.array([0, 0.1, 0], dtype=np.float64),
#               diffuse=np.array([0, 0.6, 0], dtype=np.float64),
#               specular=np.array([1, 1, 1], dtype=np.float64))
#
# RIGHT = Color(ambient=np.array([0, 0.1, 0], dtype=np.float64),
#               diffuse=np.array([2, 10, 2], dtype=np.float64),
#               specular=np.array([1, 1, 1], dtype=np.float64))
#
# GREY = Color(ambient=np.array([0.1, 0.1, 0.1], dtype=np.float64),
#               diffuse=np.array([0.6, 0.6, 0.6], dtype=np.float64),
#               specular=np.array([1, 1, 1], dtype=np.float64))
#
# SURFACE = Color(ambient=np.array([0.1, 0.1, 0.1], dtype=np.float64),
#              diffuse=np.array([6, 6, 6], dtype=np.float64),
#              specular=np.array([1, 1, 1], dtype=np.float64))
#
# TURQUOISE = Color(ambient=np.array([0.1, 0.18725, 0.1745], dtype=np.float64),
#                   diffuse=np.array([0.396, 0.74151, 0.69102], dtype=np.float64),
#                   specular=np.array([0.297254, 0.30829, 0.306678], dtype=np.float64))
# TURQUOISE_MAT = Material(color=TURQUOISE, shininess=0.1, reflectance=2, ior=1.65)
#
# BRONZE = Color(ambient=np.array([0.2125, 0.1275, 0.054], dtype=np.float64),
#                   diffuse=np.array([0.714, 0.4284, 0.18144], dtype=np.float64),
#                   specular=np.array([0.393548, 0.271906, 0.166721], dtype=np.float64))
# BRONZE_MAT = Material(color=PURPLE, shininess=10, reflectance=0.75, ior=1.180, transmittance=0, is_diffuse=True, is_mirror=False)
#
# GLASS = Color(ambient=np.array([0.0, 0.0, 0.0], dtype=np.float64),
#               diffuse=np.array([0.588235, 0.670588, 0.729412], dtype=np.float64),
#               specular=np.array([0.9, 0.9, 0.9], dtype=np.float64))
# GLASS_MAT = Material(color=GLASS, shininess=96, reflectance=0.2, ior=1.5, transmittance=1.0, is_diffuse=False, is_mirror=False)