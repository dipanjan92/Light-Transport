# bvh.py
import jax
import jax.numpy as jnp
from jax import tree_util
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Import the AABB functions from your aabb.py.
from primitives.aabb import AABB, union, union_p, aabb_intersect
from primitives.triangle import triangle_intersect
from primitives.intersects import Intersection, set_intersection


# -----------------------------------------------------------------------------
# BVH Node Definition (for the builder)
# -----------------------------------------------------------------------------
@dataclass
class BVHNode:
    """
    A BVH node used during BVH construction. For an internal node, left and right
    are indices (into the final flat node arrays) of its children. For a leaf node,
    tri_offset and tri_count describe which triangles are stored in the leaf.
    """
    aabb_min: jnp.ndarray  # shape (3,)
    aabb_max: jnp.ndarray  # shape (3,)
    left: int  # index of left child (-1 for a leaf)
    right: int  # index of right child (-1 for a leaf)
    tri_offset: int  # if leaf, offset into the flat triangle index array
    tri_count: int  # if leaf, number of triangles in this leaf


# -----------------------------------------------------------------------------
# Custom BVH Dataclass and PyTree Registration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BVH:
    """
    BVH structure stored as flat JAX arrays.
    Fields:
      aabb_mins: (num_nodes, 3) array of node minimum coordinates.
      aabb_maxs: (num_nodes, 3) array of node maximum coordinates.
      lefts: (num_nodes,) array of left child indices (or -1 if leaf).
      rights: (num_nodes,) array of right child indices (or -1 if leaf).
      tri_offsets: (num_nodes,) array with the starting index in the leaf triangle list.
      tri_counts: (num_nodes,) array with the number of triangles in the leaf.
      leaf_tri_indices: (total_leaf_triangles,) array of triangle indices.
      root: scalar int, index of the root node.
    """
    aabb_mins: jnp.ndarray
    aabb_maxs: jnp.ndarray
    lefts: jnp.ndarray
    rights: jnp.ndarray
    tri_offsets: jnp.ndarray
    tri_counts: jnp.ndarray
    leaf_tri_indices: jnp.ndarray
    root: jnp.ndarray


def bvh_flatten(bvh: BVH):
    children = (
        bvh.aabb_mins,
        bvh.aabb_maxs,
        bvh.lefts,
        bvh.rights,
        bvh.tri_offsets,
        bvh.tri_counts,
        bvh.leaf_tri_indices,
        bvh.root,
    )
    aux = None
    return children, aux


def bvh_unflatten(aux, children):
    return BVH(
        aabb_mins=children[0],
        aabb_maxs=children[1],
        lefts=children[2],
        rights=children[3],
        tri_offsets=children[4],
        tri_counts=children[5],
        leaf_tri_indices=children[6],
        root=children[7],
    )


tree_util.register_pytree_node(BVH, bvh_flatten, bvh_unflatten)


# -----------------------------------------------------------------------------
# BVH Builder Function
# -----------------------------------------------------------------------------
def build_bvh(triangles: dict, max_leaf_size: int = 4) -> BVH:
    """
    Build a BVH from triangle data. The triangles dictionary is expected to have
    the following keys (each with shape (N,3)):
       - "vertex_1"
       - "vertex_2"
       - "vertex_3"
       - "centroid"

    This function computes a bounding box for each triangle and then builds a BVH
    by recursively splitting the set of triangles (using a median-split along the
    largest axis of the centroids). The resulting BVH is returned as a BVH instance.

    Parameters:
      triangles: dictionary of triangle arrays.
      max_leaf_size: maximum number of triangles per leaf.

    Returns:
      A BVH instance containing flat arrays:
         - aabb_mins: (num_nodes, 3) array.
         - aabb_maxs: (num_nodes, 3) array.
         - lefts: (num_nodes,) array.
         - rights: (num_nodes,) array.
         - tri_offsets: (num_nodes,) array.
         - tri_counts: (num_nodes,) array.
         - leaf_tri_indices: (total_leaf_triangles,) array.
         - root: scalar int (as a jnp.array).
    """
    # Number of triangles.
    N = triangles["vertex_1"].shape[0]
    # Compute triangle bounding boxes.
    v0 = triangles["vertex_1"]
    v1 = triangles["vertex_2"]
    v2 = triangles["vertex_3"]
    tri_mins = jnp.minimum(jnp.minimum(v0, v1), v2)
    tri_maxs = jnp.maximum(jnp.maximum(v0, v1), v2)
    centroids = triangles["centroid"]

    # For the builder, we convert to NumPy arrays.
    import numpy as np
    tri_mins_np = np.array(tri_mins)
    tri_maxs_np = np.array(tri_maxs)
    centroids_np = np.array(centroids)
    indices = np.arange(N)

    # Lists to collect the built BVH nodes and leaf triangle indices.
    nodes: List[BVHNode] = []
    leaf_tri_indices: List[int] = []

    def recursive_build(tri_indices: np.ndarray) -> int:
        """
        Recursively build the BVH and return the index of the current node in the
        nodes list.
        """
        # Compute the bounding box for the current set of triangles.
        bbox_min = np.min(tri_mins_np[tri_indices], axis=0)
        bbox_max = np.max(tri_maxs_np[tri_indices], axis=0)

        # If we have few triangles, create a leaf node.
        if len(tri_indices) <= max_leaf_size:
            tri_offset = len(leaf_tri_indices)
            tri_count = len(tri_indices)
            leaf_tri_indices.extend(tri_indices.tolist())
            node = BVHNode(
                aabb_min=jnp.array(bbox_min),
                aabb_max=jnp.array(bbox_max),
                left=-1,
                right=-1,
                tri_offset=tri_offset,
                tri_count=tri_count
            )
            nodes.append(node)
            return len(nodes) - 1

        # Otherwise, split the triangle set.
        extent = bbox_max - bbox_min
        axis = np.argmax(extent)
        # Sort the triangle indices according to the centroid coordinate along the chosen axis.
        sorted_order = np.argsort(centroids_np[tri_indices, axis])
        sorted_indices = tri_indices[sorted_order]
        mid = len(sorted_indices) // 2
        left_indices = sorted_indices[:mid]
        right_indices = sorted_indices[mid:]
        # Recursively build children.
        left_child = recursive_build(left_indices)
        right_child = recursive_build(right_indices)
        # The current node’s bounding box is the union of its children.
        left_bbox_min = np.array(nodes[left_child].aabb_min)
        right_bbox_min = np.array(nodes[right_child].aabb_min)
        left_bbox_max = np.array(nodes[left_child].aabb_max)
        right_bbox_max = np.array(nodes[right_child].aabb_max)
        node_bbox_min = np.minimum(left_bbox_min, right_bbox_min)
        node_bbox_max = np.maximum(left_bbox_max, right_bbox_max)
        node = BVHNode(
            aabb_min=jnp.array(node_bbox_min),
            aabb_max=jnp.array(node_bbox_max),
            left=left_child,
            right=right_child,
            tri_offset=-1,
            tri_count=0
        )
        nodes.append(node)
        return len(nodes) - 1

    # Build the BVH tree starting from all triangle indices.
    root_index = recursive_build(indices)

    # Convert the list of leaf triangle indices to a JAX array.
    leaf_tri_indices_arr = jnp.array(leaf_tri_indices, dtype=jnp.int32)

    # Now, flatten the nodes list into arrays.
    num_nodes = len(nodes)
    aabb_mins = jnp.stack([node.aabb_min for node in nodes])  # (num_nodes, 3)
    aabb_maxs = jnp.stack([node.aabb_max for node in nodes])  # (num_nodes, 3)
    lefts = jnp.array([node.left for node in nodes], dtype=jnp.int32)
    rights = jnp.array([node.right for node in nodes], dtype=jnp.int32)
    tri_offsets = jnp.array([node.tri_offset for node in nodes], dtype=jnp.int32)
    tri_counts = jnp.array([node.tri_count for node in nodes], dtype=jnp.int32)

    # Pack the BVH data into our BVH dataclass.
    bvh = BVH(
        aabb_mins=aabb_mins,
        aabb_maxs=aabb_maxs,
        lefts=lefts,
        rights=rights,
        tri_offsets=tri_offsets,
        tri_counts=tri_counts,
        leaf_tri_indices=leaf_tri_indices_arr,
        root=jnp.array(root_index, dtype=jnp.int32)
    )
    return bvh


# -----------------------------------------------------------------------------
# BVH Traversal: intersect_bvh
# -----------------------------------------------------------------------------
def intersect_bvh(ray_origin: jnp.ndarray,
                  ray_direction: jnp.ndarray,
                  bvh: BVH,
                  triangles: dict,
                  t_max: float) -> Intersection:
    """
    Traverse the BVH and intersect the ray with the scene.

    Parameters:
      ray_origin: (3,) array with the ray origin.
      ray_direction: (3,) array with the ray direction.
      bvh: BVH instance.
      triangles: dictionary of triangle arrays (keys: "vertex_1", "vertex_2", "vertex_3").
      t_max: maximum allowed intersection distance.

    Returns:
      An Intersection object corresponding to the nearest hit.
    """
    # Get triangle arrays.
    v0_all = triangles["vertex_1"]
    v1_all = triangles["vertex_2"]
    v2_all = triangles["vertex_3"]

    # Allocate a fixed-size stack (size = number of BVH nodes).
    max_stack = bvh.aabb_mins.shape[0]
    stack = jnp.full((max_stack,), -1, dtype=jnp.int32)
    stack = stack.at[0].set(bvh.root)
    stack_ptr = jnp.array(1, dtype=jnp.int32)
    best_t = t_max
    best_isec = Intersection()  # Default intersection (with min_distance=t_max, etc.)

    # Loop state: (stack, stack_ptr, best_t, best_isec)
    state = (stack, stack_ptr, best_t, best_isec)

    def cond_fun(state):
        _, stack_ptr, _, _ = state
        return stack_ptr > 0

    def body_fun(state):
        stack, stack_ptr, best_t, best_isec = state
        # Pop a node index from the stack.
        node_index = stack[stack_ptr - 1]
        stack_ptr = stack_ptr - 1

        # Reconstruct the node's AABB.
        aabb_min = bvh.aabb_mins[node_index]
        aabb_max = bvh.aabb_maxs[node_index]
        node_aabb = AABB(min_point=aabb_min, max_point=aabb_max, centroid=(aabb_min + aabb_max) * 0.5)
        hit_aabb = aabb_intersect(node_aabb, ray_origin, ray_direction)
        state = (stack, stack_ptr, best_t, best_isec)
        state = jax.lax.cond(hit_aabb,
                             lambda s: process_node(s, node_index),
                             lambda s: s,
                             operand=state)
        return state

    def process_node(state, node_index):
        stack, stack_ptr, best_t, best_isec = state
        left = bvh.lefts[node_index]
        right = bvh.rights[node_index]
        tri_offset = bvh.tri_offsets[node_index]
        tri_count = bvh.tri_counts[node_index]

        def leaf_fn(state):
            # Process leaf: iterate over the triangles in this leaf.
            def leaf_body(i, state_inner):
                stack, stack_ptr, best_t, best_isec = state_inner
                tri_index = bvh.leaf_tri_indices[tri_offset + i]
                v0_tri = v0_all[tri_index]
                v1_tri = v1_all[tri_index]
                v2_tri = v2_all[tri_index]
                hit, t = triangle_intersect(ray_origin, ray_direction, v0_tri, v1_tri, v2_tri, best_t)

                def update_fn(_):
                    new_isec = set_intersection(ray_origin, ray_direction, v0_tri, v1_tri, v2_tri, t)
                    return (stack, stack_ptr, t, new_isec)

                return jax.lax.cond(hit & (t < best_t),
                                    update_fn,
                                    lambda _: (stack, stack_ptr, best_t, best_isec),
                                    operand=None)

            return jax.lax.fori_loop(0, tri_count, leaf_body, state)

        def internal_fn(state):
            # For an internal node, push both children onto the stack.
            stack, stack_ptr, best_t, best_isec = state
            stack = stack.at[stack_ptr].set(left)
            stack = stack.at[stack_ptr + 1].set(right)
            stack_ptr = stack_ptr + 2
            return (stack, stack_ptr, best_t, best_isec)

        is_leaf = jnp.logical_and(left == -1, right == -1)
        state = jax.lax.cond(is_leaf, leaf_fn, internal_fn, operand=state)
        return state

    final_state = jax.lax.while_loop(cond_fun, body_fun, state)
    _, _, best_t, best_isec = final_state
    return best_isec

# -----------------------------------------------------------------------------
# Example Usage:
#
# Assuming you have already loaded triangles via your io.py and built the triangle arrays:
#
#   from io import load_obj, create_triangle_arrays
#   vertices, faces = load_obj("path/to/model.obj")
#   triangles = create_triangle_arrays(vertices, faces)
#
# Then you can build the BVH as follows:
#
#   from bvh import build_bvh, intersect_bvh
#   bvh_instance = build_bvh(triangles, max_leaf_size=4)
#   ray_origin = jnp.array([0.0, 0.0, -5.0])
#   ray_direction = jnp.array([0.0, 0.0, 1.0])
#   isec = intersect_bvh(ray_origin, ray_direction, bvh_instance, triangles, t_max=1e10)
#
# The resulting Intersection object, 'isec', holds the hit information.
# -----------------------------------------------------------------------------
