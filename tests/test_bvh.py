from accelerators.bvh import BVHNode


def check_bound_intersects(this, other):
    # Checks overlap in x, y and z directions. If overlap is there in all three directions, the boxes intersect
    if this.min_point[0] <= other.max_point[0] and this.max_point[0] >= other.min_point[0] and \
            this.min_point[1] <= other.max_point[1] and this.max_point[1] >= other.min_point[1] and \
            this.min_point[2] <= other.max_point[2] and this.max_point[2] >= other.min_point[2]:
        return True
    return False


def check_bvh(node: BVHNode, primitives, depth=0, validate_bounds=False):
    assert node is not None, "Node in the BVH tree is null"
    if node.n_primitives > 0:  # leaf node
        assert node.n_primitives <= len(primitives), "Node has more primitives than it should"
        assert node.first_prim_offset < len(primitives), "Node first primitive offset is out of range"
        if validate_bounds:
            for i in range(node.first_prim_offset, node.first_prim_offset + node.n_primitives):
                assert node.bounds.intersects(primitives[i].bounds), "Primitive not within node bounds"
    else:  # interior node
        assert node.child_0 is not None, "Child 0 of the node is null"
        assert node.child_1 is not None, "Child 1 of the node is null"
        assert node.split_axis is not None, "Split axis of the node is None"
        check_bvh(node.child_0, primitives, depth + 1, validate_bounds)
        check_bvh(node.child_1, primitives, depth + 1, validate_bounds)
        if validate_bounds:
            assert check_bound_intersects(node, node.child_0.bounds), "Child 0 not within node bounds"
            assert check_bound_intersects(node, node.child_1.bounds), "Child 1 not within node bounds"


def test_bvh(root, primitives, validate_bounds=False):
    # Test the BVH tree
    try:
        check_bvh(root, primitives, validate_bounds=False)
        print("BVH tree is correctly constructed.")
    except AssertionError as e:
        print("BVH tree is not correctly constructed:", e)
