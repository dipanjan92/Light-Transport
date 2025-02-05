def test_linear_bvh(linear_bvh):
    for i in range(len(linear_bvh)):
        node = linear_bvh[i]
        # If leaf node, no further checks
        if node.n_primitives > 0:
            if node.primitives_offset is None:
                print("Leaf node's primitives_offset is None at index:", i)
                return False
            continue

        # Check if we're not exceeding list boundary
        if i + 1 >= len(linear_bvh):
            continue

        # The first child should be the very next node
        first_child = linear_bvh[i + 1]
        if first_child is None:
            print("First child node doesn't exist in Linear BVH at index:", i)
            return False

        # If node.second_child_offset isn't defined, no further checks
        if node.second_child_offset is None:
            continue

        # Check second child index isn't out of range
        if node.second_child_offset >= len(linear_bvh):
            print("Second child offset is out of range of Linear BVH. Node index:", i,
                  "Second child offset:", node.second_child_offset)
            return False

        # The second child should be at the specified offset
        second_child = linear_bvh[node.second_child_offset]

        if second_child is None:
            print("Second child node doesn't exist in Linear BVH at second_child_offset:", node.second_child_offset)
            return False

        # Check if axis is set (not None) for non-leaf nodes
        if node.axis is None:
            print("Node's axis is None at index:", i)
            return False

    # If we have checked all nodes without returning False, then the BVH is correct
    return True


def visualize_bvh_tree(linear_bvh):
    def print_node(node, level):
        indent = "    " * level
        print(f"{indent}Node: primitives_offset={node.primitives_offset}, Primitives={node.n_primitives}, Axis={node.axis}")

    def visualize_recursive(node_index, level):
        if node_index >= len(linear_bvh):
            return
        node = linear_bvh[node_index]
        print_node(node, level)

        if node.n_primitives == 0:
            visualize_recursive(node_index + 1, level + 1)
            visualize_recursive(node.second_child_offset, level + 1)

    print("BVH Tree Visualization:")
    visualize_recursive(0, 0)
