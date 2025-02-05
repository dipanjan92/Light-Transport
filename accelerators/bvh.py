import random

import numba
import numpy as np

from utils.constants import EPSILON
from primitives.intersects import aabb_intersect, triangle_intersect, intersect_bounds
from primitives.triangle import Triangle
from primitives.aabb import AABB
from utils.stdlib import nth_element, partition, mid_point_partition


@numba.experimental.jitclass([
    ('prim', Triangle.class_type.instance_type),
    ('prim_num', numba.intp),
    ('bounds', AABB.class_type.instance_type)
])
class BoundedBox:
    def __init__(self, prim, n):
        self.prim = prim
        self.prim_num = n
        self.bounds = get_bounds(prim)


node_type = numba.deferred_type()


@numba.experimental.jitclass([
    ('bounds', numba.optional(AABB.class_type.instance_type)),
    ('child_0', numba.optional(node_type)),
    ('child_1', numba.optional(node_type)),
    ('split_axis', numba.optional(numba.intp)),
    ('first_prim_offset', numba.optional(numba.intp)),
    ('n_primitives', numba.optional(numba.intp))
])
class BVHNode:
    def __init__(self):
        self.bounds = None
        self.child_0 = None
        self.child_1 = None
        self.split_axis = 0
        self.first_prim_offset = 0
        self.n_primitives = 0

    def init_leaf(self, first, n, box):
        self.first_prim_offset = first
        self.n_primitives = n
        self.bounds = box

    def init_interior(self, axis, c0, c1):
        self.child_0 = c0
        self.child_1 = c1
        self.bounds = enclose_volumes(c0.bounds, c1.bounds)
        self.split_axis = axis
        self.n_primitives = 0


node_type.define(BVHNode.class_type.instance_type)


@numba.experimental.jitclass([
    ('bounds', numba.optional(AABB.class_type.instance_type)),
    ('primitives_offset', numba.optional(numba.intp)),
    ('second_child_offset', numba.optional(numba.intp)),
    ('n_primitives', numba.optional(numba.intp)),
    ('axis', numba.optional(numba.intp))
])
class LinearBVHNode:
    def __init__(self):
        self.bounds = None
        self.primitives_offset = 0
        self.second_child_offset = 0
        self.n_primitives = 0
        self.axis = 0
        # self.pad = [0]


@numba.experimental.jitclass([
    ('count', numba.intp),
    ('bounds', numba.optional(AABB.class_type.instance_type))
])
class BucketInfo:
    def __init__(self):
        self.count = 0
        self.bounds = None


@numba.njit
def get_bounds(prim):
    # min_p = np.minimum.reduce([prim.vertex_1, prim.vertex_2, prim.vertex_3])
    # max_p = np.maximum.reduce([prim.vertex_1, prim.vertex_2, prim.vertex_3])
    min_p = prim.vertex_1
    for vertex in [prim.vertex_2, prim.vertex_3]:
        min_p = np.minimum(min_p, vertex)

    max_p = prim.vertex_1
    for vertex in [prim.vertex_2, prim.vertex_3]:
        max_p = np.maximum(max_p, vertex)

    bounded_box = AABB(min_p, max_p)

    return bounded_box


@numba.njit
def enclose_volumes(box_1, box_2):
    if box_1 is not None:
        if box_2 is None:
            bounded_box = box_1
        else:
            min_p = np.minimum(box_1.min_point, box_2.min_point)
            max_p = np.maximum(box_1.max_point, box_2.max_point)
            bounded_box = AABB(min_p, max_p)
    else:
        bounded_box = box_2
    return bounded_box


@numba.njit
def enclose_centroids(box, cent):
    if box is not None:
        min_p = np.minimum(box.min_point, cent)
        max_p = np.maximum(box.max_point, cent)
        bounded_box = AABB(min_p, max_p)
    else:
        bounded_box = AABB(cent, cent)

    return bounded_box


@numba.njit
def compute_bounding_box(node_list):
    if len(node_list) == 0:
        # Handle the case of an empty list
        return None

    # Initialize the min and max points to the first AABB's values
    min_point = node_list[0].bounds.min_point
    max_point = node_list[0].bounds.max_point

    # Iterate through the list to find the overall min and max points
    for node in node_list[1:]:
        min_point = np.minimum(min_point, node.bounds.min_point)
        max_point = np.maximum(max_point, node.bounds.max_point)

    # Create a new AABB object for the enclosing bounding box
    enclosing_box = AABB(min_point, max_point)
    return enclosing_box


@numba.njit
def get_largest_dim(box):
    dx = abs(box.max_point[0] - box.min_point[0])
    dy = abs(box.max_point[1] - box.min_point[1])
    dz = abs(box.max_point[2] - box.min_point[2])
    if dx > dy and dx > dz:
        return 0
    elif dy > dz:
        return 1
    else:
        return 2


@numba.njit
def get_surface_area(box):
    diagonal = box.max_point - box.min_point
    surface_area = 2 * (diagonal[0] * diagonal[1] + diagonal[0] * diagonal[2] + diagonal[1] * diagonal[2])
    return surface_area


@numba.njit
def offset_bounds(bounds, point):
    o = point - bounds.min_point
    if bounds.max_point[0] > bounds.min_point[0]:
        o[0] /= bounds.max_point[0] - bounds.min_point[0]

    if bounds.max_point[1] > bounds.min_point[1]:
        o[1] /= bounds.max_point[1] - bounds.min_point[1]

    if bounds.max_point[2] > bounds.min_point[2]:
        o[2] /= bounds.max_point[2] - bounds.min_point[2]

    return o


@numba.njit
def partition_pred(x, n_buckets, centroid_bounds, dim, min_cost_split_bucket):
    b = n_buckets * offset_bounds(centroid_bounds, x.bounds.centroid)[dim]

    if b == n_buckets:
        b = n_buckets - 1

    assert b >= 0, "b is less than 0"
    assert b < n_buckets, "b is not less than n_buckets"

    return b <= min_cost_split_bucket


@numba.experimental.jitclass([
    ('n_buckets', numba.int32),
    ('centroid_bounds', AABB.class_type.instance_type),
    ('dim', numba.int32),
    ('min_cost_split_bucket', numba.int32)
])
class PredicateWrapper:
    def __init__(self, n_buckets, centroid_bounds, dim, min_cost_split_bucket):
        self.n_buckets = n_buckets
        self.centroid_bounds = centroid_bounds
        self.dim = dim
        self.min_cost_split_bucket = min_cost_split_bucket

    def partition_pred(self, x):
        b = self.n_buckets * offset_bounds(self.centroid_bounds, x.bounds.centroid)[self.dim]
        if b == self.n_buckets:
            b = self.n_buckets - 1
        assert b >= 0, "b is less than 0"
        assert b < self.n_buckets, "b is not less than n_buckets"
        return b <= self.min_cost_split_bucket


@numba.experimental.jitclass([('dim', numba.int32)])
class MidPointWrapper:
    def __init__(self, dim):
        self.dim = dim

    def mid_point_pred(self, x, pmid):
        return x.bounds.centroid[self.dim] < pmid


@numba.experimental.jitclass([('dim', numba.int32)])
class DimWrapper:
    def __init__(self, dim):
        self.dim = dim

    def get_centroid_dim(self, x):
        return x.bounds.centroid[self.dim]


@numba.njit
def nth_element_with_dim(arr, n, dim, first, last):
    dim_wrapper = DimWrapper(dim)
    return nth_element(arr, n, first=first, last=last, key=dim_wrapper.get_centroid_dim)




@numba.njit
def build_bvh(primitives, bounded_boxes, start, end, ordered_prims, total_nodes, split_method=0):
    # split_method = 2  # 0: surface area heuristics, 1: middle point, 2: equal parts

    n_boxes = len(bounded_boxes)
    max_prims_in_node = int(0.1 * n_boxes)
    # max_prims_in_node = max_prims_in_node if max_prims_in_node < 4 else 4
    node = BVHNode()
    total_nodes[0] += 1
    bounds = None
    for i in range(start, end):
        bounds = enclose_volumes(bounds, bounded_boxes[i].bounds)

    print(bounds.min_point, bounds.max_point)

    if start == end:
        return node

    n_primitives = end - start
    if n_primitives == 1:
        # create left bvh node
        first_prim_offset = len(ordered_prims)
        for i in range(start, end):
            prim_num = bounded_boxes[i].prim_num
            ordered_prims.append(primitives[prim_num])
        node.init_leaf(first_prim_offset, n_primitives, bounds)
        return node
    else:
        centroid_bounds = None
        for i in range(start, end):
            centroid_bounds = enclose_centroids(centroid_bounds, bounded_boxes[i].bounds.centroid)
        dim = get_largest_dim(centroid_bounds)

        # Partition primitives into two sets and build children
        mid = (start + end) // 2
        if centroid_bounds.max_point[dim] == centroid_bounds.min_point[dim]:
            # Create leaf BVH node
            first_prim_offset = len(ordered_prims)
            for i in range(start, end):
                prim_num = bounded_boxes[i].prim_num
                ordered_prims.append(primitives[prim_num])

            node.init_leaf(first_prim_offset, n_primitives, bounds)
            return node

        else:
            if split_method == 0:
                # Partition primitives based on Surface Area Heuristic
                if n_primitives <= 2:
                    # Partition primitives into equally sized subsets
                    mid = (start + end) // 2
                    # nth_element(bounded_boxes, mid, first=start, last=end,
                    #             key=lambda x: x.bounds.centroid[dim])

                    nth_element_with_dim(bounded_boxes, mid, dim, first=start, last=end)

                else:
                    n_buckets = 12
                    buckets = [BucketInfo() for _ in range(n_buckets)]
                    # Initialize BucketInfo for SAH partition buckets
                    for i in range(start, end):
                        b = n_buckets * offset_bounds(centroid_bounds, bounded_boxes[i].bounds.centroid)[dim]
                        b = int(b)
                        if b == n_buckets:
                            b = n_buckets - 1
                        buckets[b].count += 1
                        buckets[b].bounds = enclose_volumes(buckets[b].bounds, bounded_boxes[i].bounds)

                    # compute cost for splitting each bucket
                    costs = []
                    for i in range(n_buckets - 1):
                        b0 = b1 = None
                        count_0 = 0
                        count_1 = 0
                        for j in range(i + 1):
                            b0 = enclose_volumes(b0, buckets[j].bounds)
                            count_0 += buckets[j].count
                        for j in range(i + 1, n_buckets):
                            b1 = enclose_volumes(b1, buckets[j].bounds)
                            count_1 += buckets[j].count

                        _cost = .125 * (
                                count_0 * get_surface_area(b0) + count_1 * get_surface_area(b1)) / get_surface_area(
                            bounds)
                        costs.append(_cost)

                    # find bucket to split at which minimizes SAH metric
                    min_cost = costs[0]
                    min_cost_split_bucket = 0
                    for i in range(1, n_buckets - 1):
                        if costs[i] < min_cost:
                            min_cost = costs[i]
                            min_cost_split_bucket = i

                    # Either create leaf or split primitives at selected SAH bucket
                    leaf_cost = n_primitives
                    if n_primitives > max_prims_in_node or min_cost < leaf_cost:
                        pred_wrapper = PredicateWrapper(n_buckets, centroid_bounds, dim, min_cost_split_bucket)
                        pmid = partition(bounded_boxes[start:end], pred_wrapper.partition_pred)
                        # pmid = partition(bounded_boxes[start:end],
                        #                  lambda x: partition_pred(x, n_buckets, centroid_bounds, dim,
                        #                                           min_cost_split_bucket))

                        mid = pmid + start
                    else:
                        # Create leaf BVH Node
                        first_prim_offset = len(ordered_prims)
                        for i in range(start, end):
                            prim_num = bounded_boxes[i].prim_num
                            ordered_prims.append(primitives[prim_num])
                        node.init_leaf(first_prim_offset, n_primitives, bounds)
                        return node

            elif split_method == 1:
                # partition primitives through node's midpoint
                pmid = (centroid_bounds.min_point[dim] + centroid_bounds.max_point[dim]) / 2
                midpoint_wrapper = MidPointWrapper(dim)
                mid_ptr = mid_point_partition(bounded_boxes[start:end], midpoint_wrapper, pmid)
                # mid_ptr = partition(bounded_boxes[start:end],
                #                     lambda x: x.bounds.centroid[dim] < pmid)

                mid = mid_ptr + start

                if mid == start or mid == end:
                    # fallback to next split_method
                    mid = (start + end) // 2
                    # nth_element(bounded_boxes, mid, first=start, last=end,
                    #             key=lambda x: x.bounds.centroid[dim])
                    nth_element_with_dim(bounded_boxes, mid, dim, first=start, last=end)
            else:
                # partition primitives into equally-sized subsets
                mid = (start + end) // 2
                # nth_element(bounded_boxes, mid, first=start, last=end,
                #             key=lambda x: x.bounds.centroid[dim])
                nth_element_with_dim(bounded_boxes, mid, dim, first=start, last=end)

        # print(start, mid, end)

        child_0 = build_bvh(primitives, bounded_boxes, start, mid,
                            ordered_prims, total_nodes, split_method)

        child_1 = build_bvh(primitives, bounded_boxes, mid, end,
                            ordered_prims, total_nodes, split_method)

        node.init_interior(dim, child_0, child_1)

    return node


def flatten_bvh(node_list, node, offset):
    linear_node = node_list[offset[0]]
    linear_node.bounds = node.bounds
    _offset = offset[0]
    offset[0] += 1

    if node.n_primitives > 0:
        assert node.child_0 is None and node.child_1 is None, "Both children None"
        assert node.n_primitives < 65536, "n_primitives LT 65536"
        linear_node.primitives_offset = node.first_prim_offset
        linear_node.n_primitives = node.n_primitives
    else:
        # Create interior flattened BVH node
        linear_node.axis = node.split_axis
        linear_node.n_primitives = 0
        flatten_bvh(node_list, node.child_0, offset)
        linear_node.second_child_offset = flatten_bvh(node_list, node.child_1, offset)

    return _offset


@numba.njit
def intersect_bvh(ray, primitives, linear_bvh):
    triangle = None

    inv_dir = 1 / ray.direction

    dir_is_neg = [inv_dir[0] < 0, inv_dir[1] < 0, inv_dir[2] < 0]
    to_visit_offset = 0
    current_node_index = 0
    nodes_to_visit = []

    while True:
        node = linear_bvh[int(current_node_index)]
        # print(f"Checking node: {current_node_index}, bounds: {node.bounds.min_point} to {node.bounds.max_point}")
        # print(f"ray direction: {ray.direction}")
        # print(f"inv_dir: {inv_dir[2]}")
        if intersect_bounds(node.bounds, ray, inv_dir):
            if node.n_primitives > 0:
                for i in range(node.n_primitives):
                    if primitives[node.primitives_offset + i].intersect(ray):
                        triangle = primitives[node.primitives_offset + i]
                if to_visit_offset == 0:
                    break

                to_visit_offset -= 1
                current_node_index = nodes_to_visit.pop()

            else:
                if dir_is_neg[node.axis]:
                    nodes_to_visit.append(current_node_index + 1)
                    to_visit_offset += 1
                    current_node_index = node.second_child_offset
                else:
                    nodes_to_visit.append(node.second_child_offset)
                    to_visit_offset += 1
                    current_node_index += 1
        else:
            if to_visit_offset == 0:
                break

            to_visit_offset -= 1
            current_node_index = nodes_to_visit.pop()

    return triangle

# @numba.njit
# def _intersect_bvh(ray, primitives, linear_bvh):
#     triangle = None
#
#     inv_dir = 1 / ray.direction
#
#     dir_is_neg = [inv_dir[0] < 0, inv_dir[1] < 0, inv_dir[2] < 0]
#     to_visit_offset = 0
#     current_node_index = 0
#     nodes_to_visit = np.empty(64) #[None] * 64
#
#     while True:
#         node = linear_bvh[int(current_node_index)]
#         if intersect_bounds(node.bounds, ray, inv_dir):
#             if node.n_primitives > 0:
#                 for i in range(node.n_primitives):
#                     if primitives[node.primitives_offset + i].intersect(ray):
#                         triangle = primitives[node.primitives_offset + i]
#                 if to_visit_offset == 0:
#
#                     break
#
#                 to_visit_offset -= 1
#                 current_node_index = nodes_to_visit[to_visit_offset]
#
#             else:
#                 if dir_is_neg[node.axis]:
#                     nodes_to_visit[to_visit_offset] = current_node_index + 1
#                     to_visit_offset += 1
#                     current_node_index = node.second_child_offset
#                 else:
#                     nodes_to_visit[to_visit_offset] = node.second_child_offset
#                     to_visit_offset += 1
#                     current_node_index += 1
#         else:
#             if to_visit_offset == 0:
#                 break
#
#             to_visit_offset -= 1
#             current_node_index = nodes_to_visit[to_visit_offset]
#
#     return triangle

