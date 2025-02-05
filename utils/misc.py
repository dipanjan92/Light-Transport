import numba
import numpy as np
import pyvista as pv

from accelerators.bvh import intersect_bvh
from primitives.intersects import Intersection
from primitives.triangle import Triangle


@numba.njit
def intersect_primitives(ray, triangles):
    isec = None
    for i in range(len(triangles)): #numba.literal_unroll(range(len(primitives)))
        if triangles[i].intersect(ray):
            isec = triangles[i]
    return isec


@numba.njit
def hit_object(primitives, bvh, ray):
    nearest_object = intersect_bvh(ray, primitives, bvh)
    # nearest_object = intersect_bvh(ray, primitives, bvh)

    if nearest_object is None:
        # no object was hit
        return None, None, None, None

    min_distance = ray.tmax

    intersected_point = ray.origin + min_distance * ray.direction
    normal = nearest_object.normal

    return nearest_object, min_distance, intersected_point, normal


def get_floor(x_dim, y_dim, z_dim):
    box_triangles = numba.typed.List()

    a = np.array([-x_dim, -y_dim, -z_dim], dtype=np.float64)
    b = np.array([-x_dim, -y_dim, z_dim], dtype=np.float64)
    c = np.array([x_dim, -y_dim, z_dim], dtype=np.float64)
    d = np.array([x_dim, -y_dim, -z_dim], dtype=np.float64)

    rectangle = pv.Rectangle([a, b, c ,d])
    tri_rect = rectangle.triangulate()

    # if sub_divide is not None and sub_divide>0:
    #     tri_rect = tri_rect.subdivide_adaptive(max_n_passes=sub_divide)

    rect_points = np.ascontiguousarray(tri_rect.points)
    rect_faces = tri_rect.faces.reshape((-1,4))[:, 1:4]
    rect_vx = np.ascontiguousarray(rect_points[rect_faces], dtype=np.float64)

    for vx in rect_vx:
        p, q, r = vx[0], vx[1], vx[2]

        triangle = Triangle(vertex_1=np.ascontiguousarray(p, dtype=np.float64),
                            vertex_2=np.ascontiguousarray(q, dtype=np.float64),
                            vertex_3=np.ascontiguousarray(r, dtype=np.float64)
                            )
        box_triangles.append(triangle)

    return box_triangles