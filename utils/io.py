import jax.numpy as jnp


def load_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                # Convert vertex coordinates to floats
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
            elif line.startswith("f "):
                parts = line.strip().split()
                # For each face, use only the first index (assumes triangular faces)
                # and convert from 1-based to 0-based indexing.
                face = [int(idx.split('/')[0]) - 1 for idx in parts[1:4]]
                faces.append(face)
    # Convert lists to JAX arrays
    vertices_arr = jnp.array(vertices)  # shape: (num_vertices, 3)
    faces_arr = jnp.array(faces)          # shape: (num_faces, 3)
    return vertices_arr, faces_arr


def create_triangle_arrays(vertices, faces):
    # Gather vertices for each triangle (faces has shape (N,3))
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]

    centroids = (v1 + v2 + v3) / 3.0
    edge1 = v2 - v1
    edge2 = v3 - v1
    normals = jnp.cross(edge1, edge2)
    normals = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)

    # Return a dictionary of arrays (a common pattern for JAX pipelines)
    triangles = {
        "vertex_1": v1,  # shape: (N, 3)
        "vertex_2": v2,  # shape: (N, 3)
        "vertex_3": v3,  # shape: (N, 3)
        "centroid": centroids,  # shape: (N, 3)
        "normal": normals,  # shape: (N, 3)
        "edge_1": edge1,  # shape: (N, 3)
        "edge_2": edge2,  # shape: (N, 3)
    }
    return triangles

