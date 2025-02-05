import numpy as np
import numba


@numba.njit
def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


# @numba.njit
# def normalize(vector):
#     norm = np.sqrt(np.sum(vector ** 2))
#     normalized_vector = vector / norm
#     return normalized_vector


@numba.njit
def length_squared(vector):
    return np.dot(vector, vector)


@numba.njit
def find_length(vector):
    return np.linalg.norm(vector)


def unit_vector(vector):
    """
    :param vector: Any vector as numpy array
    :return: unit vector
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    :param v1: first vector
    :param v2: second vector
    :return: angle between the two vectors
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


@numba.njit
def get_direction(p1, p2):
    """
    :param p1: first point
    :param p2: second point
    :return: direction vector
    """
    direction = (p1 - p2) / np.linalg.norm(p1 - p2)
    return direction


def rotate(origin, point, angle):
    """
    :param origin: start point the vector
    :param point: end point to be rotated
    :param angle: positive angle for counterclockwise direction
    :return: rotated vector by the provided angle
    """
    ox, oy = origin
    px, py = point

    qx = oy + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy
