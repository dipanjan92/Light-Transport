import numba
import numpy as np


@numba.experimental.jitclass([
    ('min_point', numba.float64[:]),
    ('max_point', numba.float64[:]),
    ('centroid', numba.float64[:])
])
class AABB():
    def __init__(self, min_point=None, max_point=None):
        self.min_point = np.minimum(min_point, max_point) if min_point is not None else np.array([np.inf, np.inf, np.inf])
        self.max_point = np.maximum(min_point, max_point) if max_point is not None else np.array([-np.inf, -np.inf, -np.inf])
        self.centroid = (self.min_point + self.max_point) / 2


# @numba.experimental.jitclass([
#     ('type', numba.intp),
#     ('min_point', numba.float64[:]),
#     ('max_point', numba.float64[:]),
#     ('centroid', numba.float64[:])
# ])
# class AABB:
#     def __init__(self, min_point, max_point):
#         self.min_point = min_point
#         self.max_point = max_point
#         self.centroid = (min_point+max_point)/2