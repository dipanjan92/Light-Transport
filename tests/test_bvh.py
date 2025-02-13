# bvh_inspect.py

from typing import List
from accelerators.bvh import BVHNode, LinearBVHNode
from primitives.aabb import AABB


def print_bvh_tree(nodes: List[BVHNode], node_index: int, indent: int = 0):
    """
    Recursively print the details of a BVH tree.

    Args:
        nodes: List of BVHNode objects representing the BVH tree.
        node_index: Index of the current node to print.
        indent: Current indentation level (for recursive printing).
    """
    if node_index == -1 or node_index >= len(nodes):
        return

    node = nodes[node_index]
    prefix = "  " * indent
    print(f"{prefix}Node[{node_index}]:")
    print(f"{prefix}  Bounds:")
    print(f"{prefix}    min: {node.bounds.min_point}")
    print(f"{prefix}    max: {node.bounds.max_point}")
    print(f"{prefix}    centroid: {node.bounds.centroid}")

    if node.n_primitives > 0:
        # Leaf node
        print(f"{prefix}  Leaf node:")
        print(f"{prefix}    first_prim_offset: {node.first_prim_offset}")
        print(f"{prefix}    n_primitives: {node.n_primitives}")
    else:
        # Interior node
        print(f"{prefix}  Interior node:")
        print(f"{prefix}    split_axis: {node.split_axis}")
        print(f"{prefix}    child_0: {node.child_0}")
        print(f"{prefix}    child_1: {node.child_1}")
        # Recursively print children
        if node.child_0 != -1:
            print(f"{prefix}    Child 0:")
            print_bvh_tree(nodes, node.child_0, indent + 2)
        if node.child_1 != -1:
            print(f"{prefix}    Child 1:")
            print_bvh_tree(nodes, node.child_1, indent + 2)


def print_linear_bvh(linear_bvh: List[LinearBVHNode]):
    """
    Print details of the flattened (linear) BVH.

    Args:
        linear_bvh: List of LinearBVHNode objects representing the flattened BVH.
    """
    print("Flattened BVH:")
    for i, node in enumerate(linear_bvh):
        print(f"Node[{i}]:")
        print(f"  Bounds:")
        print(f"    min: {node.bounds.min_point}")
        print(f"    max: {node.bounds.max_point}")
        print(f"    centroid: {node.bounds.centroid}")
        if node.n_primitives > 0:
            print(f"  Leaf node:")
            print(f"    primitives_offset: {node.primitives_offset}")
            print(f"    n_primitives: {node.n_primitives}")
        else:
            print(f"  Interior node:")
            print(f"    axis: {node.axis}")
            print(f"    second_child_offset: {node.second_child_offset}")
        print("-" * 40)


