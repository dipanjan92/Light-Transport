import numpy as np
import numba

from .constants import EPSILON, ZEROS
from .material import Material
from .primitives import Sphere
from .rays import Ray
from .utils import hit_object, get_cosine_hemisphere_pdf, sample_cosine_hemisphere, cosine_weighted_hemisphere_sampling
from .vectors import normalize, find_length, length_squared


@numba.experimental.jitclass([
    ('source', numba.float64[:]),
    ('radius', numba.float64),
    ('material', Material.class_type.instance_type),
    ('normal', numba.float64[:]),
    ('total_area', numba.float64)
])
class Light:
    def __init__(self, source, radius, material, normal, total_area):
        self.source = source
        self.radius = radius
        self.material = material
        self.normal = normal
        self.total_area = total_area

    def sample_incident(self, ref_isec, spheres, triangles, bvh):
        new_path_direction = normalize(self.source - ref_isec.intersected_point)
        new_path = Ray(ref_isec.intersected_point, new_path_direction, EPSILON)
        new_path_magnitude = find_length(self.source - ref_isec.intersected_point)

        isec = hit_object(spheres, triangles, bvh, new_path)
        visible = isec.min_distance is None or isec.min_distance > new_path_magnitude

        Li = 0
        pdf = 0

        light_pdf = 1/self.total_area

        if light_pdf==0 or length_squared(self.source - ref_isec.intersected_point)==0:
            return Li, new_path, 0, visible

        if np.dot(self.normal, -new_path_direction)>0:
            Li = self.material.emission
        else:
            Li = 0

        return Li, new_path, pdf, visible

    def get_pdf(self, r, isec_n):
        pdf_pos = 1/self.total_area
        pdf_dir = get_cosine_hemisphere_pdf(np.dot(isec_n, r.direction))
        return pdf_pos, pdf_dir

    def sample_source(self):
        pdf_pos = 1/self.total_area
        outgoing_direction, pdf_dir = cosine_weighted_hemisphere_sampling(self.normal, None)

        light_ray = Ray(self.source, outgoing_direction, 0)

        if np.dot(self.normal, outgoing_direction)>0:
            Le = self.material.emission
        else:
            Le = 0

        return Le, light_ray, pdf_pos, pdf_dir



@numba.experimental.jitclass([
    ('position', numba.float64[:]),
    ('look_at', numba.float64[:]),
    ('scene_normal', numba.float64[:]),
    ('fov', numba.float64),
    ('focal_length', numba.optional(numba.intp)),
    ('normal', numba.optional(numba.float64[:])),
    ('screen_area', numba.optional(numba.float64))
])
class Camera:
    def __init__(self, position):
        self.position = position
        self.look_at = normalize(np.array([0, -0.042612, -1], dtype=np.float64))
        self.scene_normal = np.array([0.0, 1.0, 0.0], np.float64)
        self.fov = np.deg2rad(30) #0.5135
        self.focal_length = None
        self.normal = None
        self.screen_area = None

    def get_importance(self, new_path):
        cos_theta = np.dot(new_path.direction, self.normal)
        if cos_theta<=0:
            return 0
        lens_area = 1
        cos2_theta = cos_theta**2
        Wi = 1 / (self.screen_area * 1 * cos2_theta * cos2_theta)
        return Wi

    def sample_incident(self, ref_isec, spheres, triangles, bvh):
        new_path_direction = normalize(self.position - ref_isec.intersected_point)
        new_path = Ray(ref_isec.intersected_point, new_path_direction, EPSILON)
        new_path_magnitude = find_length(self.position - ref_isec.intersected_point)

        isec = hit_object(spheres, triangles, bvh, new_path)
        visible = isec.min_distance is None or isec.min_distance > new_path_magnitude

        Wi =0
        pdf = 0

        if visible:
            pdf = (new_path_magnitude**2) / (np.abs(np.dot(self.normal, new_path_direction)) * 1) # lens_area = 1
            Wi = self.get_importance(new_path)

        return Wi, new_path, pdf, visible

    def get_pdf(self, r):
        cos_theta = np.dot(r.direction, self.normal)
        if cos_theta<=0:
            return 0, 0
        lens_area = 1
        pdf_pos = 1/lens_area
        pdf_dir = 1/(self.screen_area*(cos_theta**3))
        return pdf_pos, pdf_dir


def generate_custom_camera(width, height, fov):
    aspect_ratio = width / height
    near = 0.1
    far = 100.0
    gaze = np.array([0, -0.042612, -1])
    gaze /= np.linalg.norm(gaze)
    right = np.cross(np.array([0, 1, 0]), gaze)
    right /= np.linalg.norm(right)
    up = np.cross(gaze, right)
    up /= np.linalg.norm(up)

    h = 2 * near * np.tan(fov / 2)
    w = h * aspect_ratio

    M = np.zeros((4,4))
    M[0][0] = w / width
    M[1][1] = h / height
    M[2][2] = -(far + near) / (far - near)
    M[2][3] = -(2 * far * near) / (far - near)
    M[3][2] = -1

    R = np.identity(4)
    R[:3, :3] = np.column_stack((right, up, -gaze))

    T = np.identity(4)
    T[:3, 3] = [0, 0, 0]

    camera_matrix = np.dot(np.dot(M, T), R)

    return camera_matrix



@numba.experimental.jitclass([
    ('camera', Camera.class_type.instance_type),
    ('lights', numba.types.ListType(Light.class_type.instance_type)),
    # ('lights', numba.types.ListType(numba.float64[::1])),
    ('width', numba.uintp),
    ('height', numba.uintp),
    ('max_depth', numba.uintp),
    ('aspect_ratio', numba.float64),
    ('left', numba.float64),
    ('top', numba.float64),
    ('right', numba.float64),
    ('bottom', numba.float64),
    ('f_distance', numba.float64),
    ('number_of_samples', numba.uintp),
    ('image', numba.float64[:,:,:]),
    ('rand_0', numba.float64[:,:,:,:]),
    ('rand_1', numba.float64[:,:,:,:])
])
class Scene:
    def __init__(self, camera, lights, width=400, height=400, max_depth=3, f_distance=5, number_of_samples=8):
        self.camera = camera
        self.lights = lights
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.aspect_ratio = width/height
        self.left = -1
        self.top = 1/self.aspect_ratio
        self.right = 1
        self.bottom = -1/self.aspect_ratio
        self.f_distance = f_distance
        self.image = np.zeros((height, width, 3), dtype=np.float64)
        self.number_of_samples = number_of_samples
        self.rand_0 = np.random.rand(height, width, number_of_samples, max_depth)
        self.rand_1 = np.random.rand(height, width, number_of_samples, max_depth)



@numba.experimental.jitclass([
    ('camera', Camera.class_type.instance_type),
    # ('lights', Sphere.class_type.instance_type),
    ('lights', numba.types.ListType(Light.class_type.instance_type)),
    ('width', numba.uintp),
    ('height', numba.uintp),
    ('max_depth', numba.uintp),
    ('aspect_ratio', numba.float64),
    ('left', numba.float64),
    ('top', numba.float64),
    ('right', numba.float64),
    ('bottom', numba.float64),
    ('f_distance', numba.float64),
    ('number_of_samples', numba.uintp),
    ('image', numba.float64[:,:,:]),
    ('rand_0', numba.float64[:,:,:,:]),
    ('rand_1', numba.float64[:,:,:,:]),
    ('t_matrix', numba.float64[:, :])
])
class SphereScene:
    def __init__(self, camera, lights, width=400, height=400, max_depth=3, f_distance=5, number_of_samples=8):
        self.camera = camera
        self.lights = lights
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.aspect_ratio = width/height
        self.left = -1
        self.top = 1/self.aspect_ratio
        self.right = 1
        self.bottom = -1/self.aspect_ratio
        self.f_distance = f_distance
        self.image = np.zeros((height, width, 3), dtype=np.float64)
        self.number_of_samples = number_of_samples
        self.rand_0 = np.random.rand(width, height, number_of_samples, max_depth)
        self.rand_1 = np.random.rand(width, height, number_of_samples, max_depth)
        self.t_matrix = np.identity(4) # required transformations


