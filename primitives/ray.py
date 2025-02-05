import numpy as np
import numba
from utils.vectors import normalize


c_1d_vec = numba.types.Array(dtype=numba.float64, ndim=1, layout="C")

@numba.experimental.jitclass([
    ('origin', numba.float64[:]),
    ('direction', numba.float64[:]),
    ('tmin', numba.float64),
    ('tmax', numba.float64)
])
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
        self.tmin = 0.0
        self.tmax = np.inf
