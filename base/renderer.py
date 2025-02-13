# renderer.py
import jax
import jax.numpy as jnp
from base.camera import PerspectiveCamera
from base.frame import frame_from_z
from accelerators.bvh import intersect_bvh  # our jitted BVH traversal
from primitives.ray import Ray  # Ensure that Ray is defined as a PyTree
from primitives.intersects import Intersection


def create_default_camera(triangles: dict, width: int, height: int, fov: float) -> PerspectiveCamera:
    # (Same as your existing code)
    v_all = jnp.concatenate([triangles["vertex_1"],
                             triangles["vertex_2"],
                             triangles["vertex_3"]], axis=0)
    obj_min = jnp.min(v_all, axis=0)
    obj_max = jnp.max(v_all, axis=0)
    center = (obj_min + obj_max) / 2.0
    extent = obj_max - obj_min
    radius = 0.5 * jnp.max(extent)
    fov_rad = fov * jnp.pi / 180.0
    d = radius / jnp.tan(fov_rad / 2.0) + radius
    position = center + jnp.array([0.0, 0.0, d])
    forward = (center - position) / jnp.linalg.norm(center - position)
    frame = frame_from_z(forward)
    aspect_ratio = width / height
    screen_dy = 2 * jnp.tan(fov_rad / 2.0)
    screen_dx = screen_dy * aspect_ratio
    dx_camera = jnp.array([screen_dx / width, 0.0, 0.0])
    dy_camera = jnp.array([0.0, screen_dy / height, 0.0])
    lens_radius = 0.0
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


def generate_rays(camera: PerspectiveCamera) -> Ray:
    """
    Generate a batched Ray for every pixel.
    Assumes camera.generate_ray(s, t) returns (origin, direction) for normalized s,t.
    """
    height = camera.height
    width = camera.width
    s = (jnp.arange(width) + 0.5) / width  # (width,)
    t = (jnp.arange(height) + 0.5) / height  # (height,)
    s_grid, t_grid = jnp.meshgrid(s, t)  # (height, width)
    s_flat = s_grid.ravel()  # (N,)
    t_flat = t_grid.ravel()  # (N,)
    v_generate_ray = jax.vmap(lambda s, t: camera.generate_ray(s, t), in_axes=(0, 0))
    origins, directions = v_generate_ray(s_flat, t_flat)
    return Ray(origin=origins, direction=directions)


def render(flattened_bvh: dict, ordered_prims: dict, camera: PerspectiveCamera,
           batch_size: int = 1024) -> jnp.ndarray:
    """
    Render an image by processing rays in parallel batches.
    Uses padding and reshaping so that the BVH traversal (which is already jitted
    and vectorized) is applied over a 2D batch without Python loops.
    Pixels are blue if a ray hit the object, black otherwise.
    """
    # Generate all rays.
    rays = generate_rays(camera)  # rays.origin: (N, 3), rays.direction: (N, 3)
    N = rays.origin.shape[0]

    # Compute number of batches and pad if necessary.
    num_batches = -(-N // batch_size)  # ceil division
    padded_N = num_batches * batch_size
    pad_amount = padded_N - N

    # Pad the rays along axis 0 (we pad with zeros; these extra rays will be ignored later).
    origins_padded = jnp.pad(rays.origin, ((0, pad_amount), (0, 0)))
    directions_padded = jnp.pad(rays.direction, ((0, pad_amount), (0, 0)))

    # Reshape into (num_batches, batch_size, 3)
    origins_batches = origins_padded.reshape((num_batches, batch_size, 3))
    directions_batches = directions_padded.reshape((num_batches, batch_size, 3))

    # Create a batched Ray object.
    batched_rays = Ray(origin=origins_batches, direction=directions_batches)

    # Create a vectorized BVH intersection function that works on a batch of rays.
    # Here we first vmap over the inner axis (rays in a batch) and then over the batch axis.
    intersect_batch = jax.vmap(
        jax.vmap(lambda ray: intersect_bvh(ray, ordered_prims, flattened_bvh, t_max=1e10), in_axes=0), in_axes=0)

    # Compute intersections for all rays.
    intersections_batches = intersect_batch(batched_rays)

    # Now, each field of Intersection is an array of shape (num_batches, batch_size, ...).
    # Remove the padded rays: slice the first N elements along axis 0 after flattening.
    def flatten_and_trim(x):
        # Flatten the first two dimensions and take only the first N entries.
        return x.reshape((-1,) + x.shape[2:])[:N]

    final_intersection = Intersection(
        min_distance=flatten_and_trim(intersections_batches.min_distance),
        intersected_point=flatten_and_trim(intersections_batches.intersected_point),
        normal=flatten_and_trim(intersections_batches.normal),
        shading_normal=flatten_and_trim(intersections_batches.shading_normal),
        dpdu=flatten_and_trim(intersections_batches.dpdu),
        dpdv=flatten_and_trim(intersections_batches.dpdv),
        dndu=flatten_and_trim(intersections_batches.dndu),
        dndv=flatten_and_trim(intersections_batches.dndv),
        nearest_object=flatten_and_trim(intersections_batches.nearest_object),
        intersected=flatten_and_trim(intersections_batches.intersected)
    )

    # Map the "intersected" flag to colors: blue if hit (1), black if miss (0).
    v_color = jax.vmap(lambda flag: jnp.where(flag == 1,
                                              jnp.array([0.0, 0.0, 1.0]),
                                              jnp.array([0.0, 0.0, 0.0])))
    colors = v_color(final_intersection.intersected)  # shape (N, 3)

    # Reshape back to the image shape (height, width, 3).
    image = colors.reshape((camera.height, camera.width, 3))
    return image

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from io import load_obj, create_triangle_arrays  # your I/O functions

    # Load the OBJ file.
    file_path = "path/to/your.obj"
    vertices, faces = load_obj(file_path)
    triangles = create_triangle_arrays(vertices, faces)

    # Build primitives and the BVH.
    from accelerators.bvh import create_primitives, create_bvh_primitives, build_bvh, flatten_bvh, pack_primitives, pack_linear_bvh

    primitives = create_primitives(triangles)
    bvh_primitives = create_bvh_primitives(triangles)
    split_method = 0  # for example, using SAH
    nodes, ordered_prims = build_bvh(primitives, bvh_primitives, 0, len(bvh_primitives), [], split_method)
    packed_prims = pack_primitives(ordered_prims)
    linear_bvh_list = flatten_bvh(nodes, 0)
    linear_bvh = pack_linear_bvh(linear_bvh_list)

    # Create a default camera.
    width = 640
    height = 480
    fov = 45.0
    camera = create_default_camera(triangles, width, height, fov)

    # Render the image using the parallel batched renderer.
    image = render(linear_bvh, packed_prims, camera, batch_size=1024)

    # Display the image.
    plt.imshow(image)
    plt.title("Rendered Image (Parallel Batching)")
    plt.axis("off")
    plt.show()
