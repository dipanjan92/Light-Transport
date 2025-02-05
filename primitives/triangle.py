
import numpy as np
import numba
import enum

from utils.vectors import *
from utils.constants import *


@numba.experimental.jitclass([
    ('vertex_1', numba.float64[:]),
    ('vertex_2', numba.float64[:]),
    ('vertex_3', numba.float64[:]),
    ('centroid', numba.float64[:]),
    ('normal', numba.float64[:])
])
class Triangle():
    def __init__(self, vertex_1, vertex_2, vertex_3):
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        self.vertex_3 = vertex_3
        self.centroid = (vertex_1+vertex_2+vertex_3)/3
        self.normal = normalize(np.cross(vertex_2-vertex_1, vertex_3-vertex_1))

    def intersect(self, ray):

        vertex_a = self.vertex_1
        vertex_b = self.vertex_2
        vertex_c = self.vertex_3

        plane_normal = self.normal

        ab = vertex_b - vertex_a
        ac = vertex_c - vertex_a

        # ray_direction = normalize(ray_end - ray_origin)

        ray_dot_plane = np.dot(ray.direction, plane_normal)

        if abs(ray_dot_plane)<=EPSILON:
            return False

        pvec = np.cross(ray.direction, ac)

        det = np.dot(ab, pvec)

        if -EPSILON < det < EPSILON:
            return False

        inv_det = 1.0 / det

        tvec = ray.origin - vertex_a

        u = np.dot(tvec, pvec) * inv_det

        if u < 0 or u > 1:
            return False

        qvec = np.cross(tvec, ab)

        v = np.dot(ray.direction, qvec) * inv_det

        if v < 0 or u+v > 1:
            return False

        t = np.dot(ac, qvec) * inv_det

        if t < ray.tmin or t > ray.tmax:
            return False

        ray.tmax = t

        return True

    def get_area(self):
        return 0.5 * normalize(np.cross(self.vertex_2-self.vertex_1, self.vertex_3-self.vertex_1))

    def get_normal(self, intersection):
        return self.normal