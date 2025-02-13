# main.py

import jax
import jax.numpy as jnp


# -------------------------------
# IO and Triangle Data Creation
# -------------------------------
# (This is your io.py code.)

def load_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
            elif line.startswith("f "):
                parts = line.strip().split()
                # OBJ face indices are 1-based.
                face = [int(idx.split('/')[0]) - 1 for idx in parts[1:4]]
                faces.append(face)
    vertices_arr = jnp.array(vertices)  # shape (num_vertices, 3)
    faces_arr = jnp.array(faces)  # shape (num_faces, 3)
    return vertices_arr, faces_arr


def create_triangle_arrays(vertices, faces):
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    centroids = (v1 + v2 + v3) / 3.0
    edge1 = v2 - v1
    edge2 = v3 - v1
    normals = jnp.cross(edge1, edge2)
    # Normalize normals (avoid division by zero)
    norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
    normals = jnp.where(norms > 0, normals / norms, normals)
    triangles = {
        "vertex_1": v1,  # (N, 3)
        "vertex_2": v2,  # (N, 3)
        "vertex_3": v3,  # (N, 3)
        "centroid": centroids,  # (N, 3)
        "normal": normals,  # (N, 3)
        "edge_1": edge1,  # (N, 3)
        "edge_2": edge2,  # (N, 3)
    }
    return triangles
