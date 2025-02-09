# debug_bvh.py
import jax
import jax.numpy as jnp
import numpy as np

# Import your AABB and BVH definitions.
from primitives.aabb import AABB, get_largest_dim, get_surface_area
from accelerators.bvh import BVH, build_bvh


# -----------------------------------------------------------------------------
# Debugging functions for AABB and BVH
# -----------------------------------------------------------------------------
def debug_print_aabb(aabb: AABB):
    """Print details about an AABB."""
    print("  AABB:")
    print("    Min bounds   :", aabb.min_point)
    print("    Max bounds   :", aabb.max_point)
    # Compute the centroid as the average of min and max.
    centroid = (aabb.min_point + aabb.max_point) * 0.5
    print("    Centroid     :", centroid)
    diag = aabb.max_point - aabb.min_point
    print("    Diagonal     :", diag)
    surface_area = 2.0 * (diag[0] * diag[1] + diag[0] * diag[2] + diag[1] * diag[2])
    print("    Surface Area :", surface_area)
    # Using your helper, get the largest dimension (which is often used as the split axis).
    largest_dim = int(get_largest_dim(aabb))
    print("    Largest dim (split axis):", largest_dim)


def debug_print_bvh(bvh: BVH, node_index: int = None, depth: int = 0):
    """
    Recursively print the BVH structure.

    Parameters:
      bvh: A BVH instance (the custom dataclass we built).
      node_index: current node index (start with the BVH root).
      depth: current recursion depth (used for indentation).
    """
    if node_index is None:
        node_index = int(bvh.root)
    indent = "  " * depth

    # Retrieve this node's bounding box.
    aabb_min = bvh.aabb_mins[node_index]
    aabb_max = bvh.aabb_maxs[node_index]
    # Reconstruct an AABB instance for printing.
    node_aabb = AABB(min_point=aabb_min, max_point=aabb_max, centroid=(aabb_min + aabb_max) * 0.5)

    print(indent + f"Node {node_index}:")
    debug_print_aabb(node_aabb)

    left = int(bvh.lefts[node_index])
    right = int(bvh.rights[node_index])
    if left == -1 and right == -1:
        # Leaf node.
        tri_offset = int(bvh.tri_offsets[node_index])
        tri_count = int(bvh.tri_counts[node_index])
        print(indent + f"  Leaf node: tri_offset = {tri_offset}, tri_count = {tri_count}")
    else:
        # Internal node.
        print(indent + f"  Internal node: left = {left}, right = {right}")
        debug_print_bvh(bvh, node_index=left, depth=depth + 1)
        debug_print_bvh(bvh, node_index=right, depth=depth + 1)



