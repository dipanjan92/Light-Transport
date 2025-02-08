import jax
import jax.numpy as jnp
from dataclasses import dataclass
from math import pi
from typing import Tuple
from base.frame import Frame
from primitives.ray import Ray


def sample_uniform_disk_concentric(u: float, v: float) -> jnp.ndarray:
    """
    Map two uniform random numbers in [0,1] to a point on the unit disk using
    concentric mapping.
    """
    # Map [0,1] -> [-1,1]
    u = 2 * u - 1
    v = 2 * v - 1

    def nonzero_fn():
        abs_u = jnp.abs(u)
        abs_v = jnp.abs(v)

        def branch_u():
            r = u
            theta = (pi / 4) * (v / u)
            return r * jnp.cos(theta), r * jnp.sin(theta)

        def branch_v():
            r = v
            theta = (pi / 2) - (pi / 4) * (u / v)
            return r * jnp.cos(theta), r * jnp.sin(theta)

        rx, ry = jax.lax.cond(abs_u > abs_v, branch_u, branch_v)
        return jnp.array([rx, ry])

    result = jax.lax.cond(jnp.logical_and(u == 0.0, v == 0.0),
                          lambda: jnp.array([0.0, 0.0]),
                          nonzero_fn)
    return result


@dataclass(frozen=True)
class PerspectiveCamera:
    width: int
    height: int
    position: jnp.ndarray  # shape (3,)
    frame: Frame  # our JAX Frame type
    fov: float  # vertical field-of-view in degrees
    aspect_ratio: float
    lens_radius: float  # >0 for depth-of-field, 0 for pinhole
    focal_distance: float
    screen_dx: float  # physical film/sensor width scaling factor
    screen_dy: float  # physical film/sensor height scaling factor
    dx_camera: jnp.ndarray  # differential in camera space (3,)
    dy_camera: jnp.ndarray  # differential in camera space (3,)

    def camera_from_raster(self, p_film: jnp.ndarray) -> jnp.ndarray:
        """
        Convert film (raster) coordinates to camera space.
        p_film is a vector (x,y,0).
        """
        return jnp.array([
            (p_film[0] - 0.5) * self.screen_dx,
            (p_film[1] - 0.5) * self.screen_dy,
            1.0
        ])

    def generate_ray(self, s: float, t: float, lens_u: float = 0.5, lens_v: float = 0.5) -> Tuple[
        jnp.ndarray, jnp.ndarray]:
        """
        Generate a camera ray from raster coordinates (s,t). If lens_radius > 0,
        additional random numbers (lens_u, lens_v) in [0,1] are used for depth of field.
        Returns (world_ray_origin, world_ray_direction).
        """
        # Compute raster position in camera space.
        p_film = jnp.array([s, t, 0.0])
        p_camera = jnp.array([
            (p_film[0] - 0.5) * self.screen_dx,
            (p_film[1] - 0.5) * self.screen_dy,
            1.0
        ])
        # Initial ray (pinhole)
        ray_dir = p_camera / jnp.linalg.norm(p_camera)
        ray_origin = jnp.array([0.0, 0.0, 0.0])

        # Depth of field: perturb the origin if a lens is used.
        if self.lens_radius > 0.0:
            p_lens = self.lens_radius * sample_uniform_disk_concentric(lens_u, lens_v)
            ft = self.focal_distance / ray_dir[2]
            p_focus = ray_origin + ft * ray_dir
            ray_origin = jnp.array([p_lens[0], p_lens[1], 0.0])
            ray_dir = (p_focus - ray_origin) / jnp.linalg.norm(p_focus - ray_origin)

        # Transform ray from camera space to world space.
        world_ray_origin = self.position + self.frame.from_local(ray_origin)
        world_ray_dir = self.frame.from_local(ray_dir)
        world_ray_dir = world_ray_dir / jnp.linalg.norm(world_ray_dir)
        return world_ray_origin, world_ray_dir

    def generate_ray_differential(self, s: float, t: float, lens_u: float = 0.5, lens_v: float = 0.5
                                  ) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Generate a ray along with its differentials.
        Returns a tuple:
          (world_ray_origin, world_ray_dir,
           world_rx_origin, world_rx_dir,
           world_ry_origin, world_ry_dir)
        """
        p_film = jnp.array([s, t, 0.0])
        p_camera = jnp.array([
            (p_film[0] - 0.5) * self.screen_dx,
            (p_film[1] - 0.5) * self.screen_dy,
            1.0
        ])
        ray_dir = p_camera / jnp.linalg.norm(p_camera)
        ray_origin = jnp.array([0.0, 0.0, 0.0])
        rx_direction = (p_camera + self.dx_camera) / jnp.linalg.norm(p_camera + self.dx_camera)
        ry_direction = (p_camera + self.dy_camera) / jnp.linalg.norm(p_camera + self.dy_camera)
        rx_origin = ray_origin
        ry_origin = ray_origin

        if self.lens_radius > 0.0:
            p_lens = self.lens_radius * sample_uniform_disk_concentric(lens_u, lens_v)
            ft = self.focal_distance / ray_dir[2]
            p_focus = ray_origin + ft * ray_dir
            ray_origin = jnp.array([p_lens[0], p_lens[1], 0.0])
            ray_dir = (p_focus - ray_origin) / jnp.linalg.norm(p_focus - ray_origin)

            dx = (p_camera + self.dx_camera) / jnp.linalg.norm(p_camera + self.dx_camera)
            ft_x = self.focal_distance / dx[2]
            p_focus_x = ray_origin + ft_x * dx
            rx_origin = jnp.array([p_lens[0], p_lens[1], 0.0])
            rx_direction = (p_focus_x - rx_origin) / jnp.linalg.norm(p_focus_x - rx_origin)

            dy = (p_camera + self.dy_camera) / jnp.linalg.norm(p_camera + self.dy_camera)
            ft_y = self.focal_distance / dy[2]
            p_focus_y = ray_origin + ft_y * dy
            ry_origin = jnp.array([p_lens[0], p_lens[1], 0.0])
            ry_direction = (p_focus_y - ry_origin) / jnp.linalg.norm(p_focus_y - ry_origin)

        world_ray_origin = self.position + self.frame.from_local(ray_origin)
        world_ray_dir = self.frame.from_local(ray_dir)
        world_ray_dir = world_ray_dir / jnp.linalg.norm(world_ray_dir)
        world_rx_origin = self.position + self.frame.from_local(rx_origin)
        world_rx_dir = self.frame.from_local(rx_direction)
        world_rx_dir = world_rx_dir / jnp.linalg.norm(world_rx_dir)
        world_ry_origin = self.position + self.frame.from_local(ry_origin)
        world_ry_dir = self.frame.from_local(ry_direction)
        world_ry_dir = world_ry_dir / jnp.linalg.norm(world_ry_dir)
        return (world_ray_origin, world_ray_dir,
                world_rx_origin, world_rx_dir,
                world_ry_origin, world_ry_dir)
