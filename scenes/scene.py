import numpy as np
import numba



@numba.experimental.jitclass([
    ('position', numba.float64[:]),
    ('focal_length', numba.intp)
])
class Camera:
    def __init__(self, position, focal_length):
        self.position = position
        self.focal_length = focal_length


@numba.experimental.jitclass([
    ('look_at', numba.float64[:]),
    ('camera', numba.float64[:]),
    ('width', numba.uintp),
    ('height', numba.uintp),
    ('image', numba.float64[:,:,:])
])
class Scene:
    def __init__(self, look_at, camera, width=400, height=400):
        self.look_at = look_at
        self.camera = camera
        self.width = width
        self.height = height
        self.image = np.zeros((height, width, 3), dtype=np.float64)