# render.py
import jax
import jax.numpy as jnp
from base.camera import PerspectiveCamera
from base.frame import frame_from_z
from accelerators.bvh import intersect_bvh, build_bvh, BVH
import jax.numpy as jnp


def create_default_camera(triangles: dict, width: int, height: int, fov: float) -> PerspectiveCamera:
    """
    Compute the overall bounds of the object from triangle data and create a default
    perspective camera positioned so that the entire object appears on screen.

    Parameters:
      triangles: Dictionary with keys "vertex_1", "vertex_2", "vertex_3", etc.
      width: Image width in pixels.
      height: Image height in pixels.
      fov: Vertical field-of-view in degrees.

    Returns:
      A PerspectiveCamera instance.
    """
    # Compute overall bounds from all vertices.
    v_all = jnp.concatenate([triangles["vertex_1"],
                             triangles["vertex_2"],
                             triangles["vertex_3"]], axis=0)
    obj_min = jnp.min(v_all, axis=0)
    obj_max = jnp.max(v_all, axis=0)
    center = (obj_min + obj_max) / 2.0
    extent = obj_max - obj_min
    radius = 0.5 * jnp.max(extent)

    # Convert fov to radians.
    fov_rad = fov * jnp.pi / 180.0
    # Compute a distance that fits the object in the view. (Add a little extra margin.)
    d = radius / jnp.tan(fov_rad / 2.0) + radius

    # Position the camera along the positive z-axis so that it looks toward the object center.
    position = center + jnp.array([0.0, 0.0, d])
    # The forward direction is from camera position to the object center.
    forward = (center - position) / jnp.linalg.norm(center - position)
    # Use a fixed up vector.
    up = jnp.array([0.0, 1.0, 0.0])
    # Create a frame from the forward direction.
    frame = frame_from_z(forward)

    aspect_ratio = width / height
    # Compute the physical film/sensor size from the fov.
    screen_dy = 2 * jnp.tan(fov_rad / 2.0)
    screen_dx = screen_dy * aspect_ratio
    # Compute differentials in camera space.
    dx_camera = jnp.array([screen_dx / width, 0.0, 0.0])
    dy_camera = jnp.array([0.0, screen_dy / height, 0.0])
    lens_radius = 0.0  # Pinhole camera.
    focal_distance = d

    return PerspectiveCamera(
        width=width,
        height=height,
        position=position,
        frame=frame,
        fov=fov,
        aspect_ratio=aspect_ratio,
        lens_radius=lens_radius,
        focal_distance=focal_distance,
        screen_dx=screen_dx,
        screen_dy=screen_dy,
        dx_camera=dx_camera,
        dy_camera=dy_camera
    )


def render(triangles: dict, bvh: BVH, camera: PerspectiveCamera) -> jnp.ndarray:
    """
    Render an image by shooting rays from the camera into the scene and testing
    intersections against the BVH. If a ray intersects the object, the corresponding
    pixel is painted blue; otherwise, it is black.

    Parameters:
      triangles: Dictionary of triangle arrays.
      bvh: BVH instance for the scene.
      camera: PerspectiveCamera instance.

    Returns:
      An image as a jnp.ndarray of shape (height, width, 3) with values in [0,1].
    """
    height = camera.height
    width = camera.width

    # Define a function that computes the color for one pixel.
    def pixel_color(s, t):
        # Generate a ray from the camera given raster coordinates (s,t) in [0,1].
        ray_origin, ray_dir = camera.generate_ray(s, t)
        # Intersect the ray with the scene BVH.
        isec = intersect_bvh(ray_origin, ray_dir, bvh, triangles, t_max=1e10)
        # If an intersection is found, return blue; otherwise black.
        return jnp.where(isec.intersected == 1,
                         jnp.array([0.0, 0.0, 1.0]),
                         jnp.array([0.0, 0.0, 0.0]))

    # Create arrays of raster coordinates.
    s_coords = (jnp.arange(width) + 0.5) / width  # horizontal coordinate in [0,1]
    t_coords = (jnp.arange(height) + 0.5) / height  # vertical coordinate in [0,1]

    # Vectorize pixel_color over the image.
    pixel_color_vmap = jax.vmap(lambda t: jax.vmap(lambda s: pixel_color(s, t))(s_coords), in_axes=0)
    image = pixel_color_vmap(t_coords)
    # The resulting image has shape (height, width, 3).
    return image

