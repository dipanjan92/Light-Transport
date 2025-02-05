import struct

import numba
import numpy as np

from accelerators.bvh import BVHNode, BucketInfo, enclose_volumes, get_largest_dim, enclose_centroids, get_surface_area, \
    offset_bounds, partition_pred, compute_bounding_box
from primitives.aabb import AABB
from utils.stdlib import _partition, find_interval


#@numba.experimental.jitclass([
#     ('prim_ix', numba.intp),
#     ('morton_code', numba.int32)
# ])
class MortonPrimitive():
    def __init__(self):
        self.prim_ix = 0
        self.morton_code = 0


#@numba.experimental.jitclass([
#     ('start_ix', numba.intp),
#     ('n_primitives', numba.intp),
#     ('build_nodes', numba.types.ListType(BVHNode.class_type.instance_type))
# ])
class LBVHTreelet():
    def __init__(self, start_ix, n_primitives, build_nodes):
        self.start_ix = start_ix
        self.n_primitives = n_primitives
        self.build_nodes = build_nodes


#@numba.experimental.jitclass([
    ('n_buckets', numba.int32),
#     ('centroid_bounds', AABB.class_type.instance_type),
#     ('dim', numba.int32),
#     ('min_cost_split_bucket', numba.int32)
# ])
class PartitionWrapper:
    def __init__(self, n_buckets, centroid_bounds, dim, min_cost_split_bucket):
        self.n_buckets = n_buckets
        self.centroid_bounds = centroid_bounds
        self.dim = dim
        self.min_cost_split_bucket = min_cost_split_bucket

    def partition_pred(self, x):
        centroid = (x.bounds.min_point[self.dim] + x.bounds.max_point[self.dim]) * 0.5
        b = (centroid - self.centroid_bounds.min_point[self.dim]) / (
                self.centroid_bounds.max_point[self.dim] - self.centroid_bounds.min_point[self.dim])
        b = int(self.n_buckets * b)
        if b == self.n_buckets:
            b = self.n_buckets - 1
        assert b >= 0, "b is less than 0"
        assert b < self.n_buckets, "b is not less than n_buckets"
        return b <= self.min_cost_split_bucket


#@numba.experimental.jitclass([
#     ('morton_prims', numba.types.ListType(MortonPrimitive.class_type.instance_type)),
#     ('mask', numba.int32)
# ])
class IntervalWrapper:
    def __init__(self, morton_prims, mask):
        self.morton_prims = morton_prims
        self.mask = mask

    def interval_pred(self, i):
        return self.morton_prims[0].morton_code & self.mask == self.morton_prims[i].morton_code & self.mask


# def left_shift_3(x):
#     assert x <= (1 << 10), "x is bigger than 2^10"
#     if x == (1 << 10):
#         x -= 1
#
#     x = (x | (x << 16)) & 0x30000ff
#     # x = ---- --98 ---- ---- ---- ---- 7654 3210
#     x = (x | (x << 8)) & 0x300f00f
#     # x = ---- --98 ---- ---- 7654 ---- ---- 3210
#     x = (x | (x << 4)) & 0x30c30c3
#     # x = ---- --98 ---- 76-- --54 ---- 32-- --10
#     x = (x | (x << 2)) & 0x9249249
#     # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
#
#     return x

#@numba.njit
def left_shift_3(x):
    x = int(x)

    assert x <= (1 << 10), "x is bigger than 2^10"
    if x == (1 << 10):
        x = x - 1

    x = (x | (x << 16)) & 0b00000011000000000000000011111111
    # x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x | (x << 8)) & 0b00000011000000001111000000001111
    # x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x | (x << 4)) & 0b00000011000011000011000011000011
    # x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x | (x << 2)) & 0b00001001001001001001001001001001
    # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0

    return x


#@numba.njit
def encode_morton_3(v):
    assert v[0] >= 0, "x must be non-negative"
    assert v[1] >= 0, "y must be non-negative"
    assert v[2] >= 0, "z must be non-negative"
    return (left_shift_3(v[2]) << 2) | (left_shift_3(v[1]) << 1) | left_shift_3(v[0])


#@numba.njit
def radix_sort(v):
    temp_vector = [MortonPrimitive() for _ in range(len(v))]
    bits_per_pass = 6
    n_bits = 30
    assert (n_bits % bits_per_pass) == 0, "Radix sort bitsPerPass must evenly divide nBits"
    n_passes = int(n_bits / bits_per_pass)

    for _pass in range(n_passes):
        # Perform one pass of radix sort, sorting _bitsPerPass_ bits
        low_bit = _pass * bits_per_pass

        # Set in and out vector pointers for radix sort pass
        in_v = [i for i in temp_vector] if _pass & 1 else [i for i in v]
        out_v = [i for i in v] if _pass & 1 else [i for i in temp_vector]

        # Count number of zero bits in array for current radix sort bit
        n_buckets = 1 << bits_per_pass
        bucket_count = [0 for _ in range(n_buckets)]
        bit_mask = (1 << bits_per_pass) - 1
        for mp in in_v:
            bucket = (mp.morton_code >> low_bit) & bit_mask
            assert 0 <= bucket < n_buckets
            bucket_count[bucket] += 1

        # Compute starting index in output array for each bucket
        out_ix = [0 for _ in range(n_buckets)]
        for i in range(1, n_buckets):
            out_ix[i] = out_ix[i - 1] + bucket_count[i - 1]

        # Store sorted values in output array
        for mp in in_v:
            bucket = (mp.morton_code >> low_bit) & bit_mask
            out_v[out_ix[bucket]] = mp
            out_ix[bucket] += 1

    # Copy final result from _tempVector_, if needed
    if n_passes & 1:
        v, temp_vector = temp_vector, v


#@numba.njit
def build_upper_sah(treelet_roots, start, end, total_nodes):
    assert start < end, "start should be less than end"
    n_nodes = end - start
    if n_nodes == 1:
        return treelet_roots[start]
    total_nodes[0] += 1
    node = BVHNode()  # Assuming default constructor

    # Compute bounds of all nodes under this HLBVH node
    bounds = None
    for i in range(start, end):
        bounds = enclose_volumes(bounds, treelet_roots[i].bounds)

    # Compute bound of HLBVH node centroids, choose split dimension _dim_
    centroid_bounds = None
    for i in range(start, end):
        centroid = (treelet_roots[i].bounds.min_point + treelet_roots[i].bounds.max_point) * 0.5
        centroid_bounds = enclose_centroids(centroid_bounds, centroid)
    dim = get_largest_dim(centroid_bounds)

    # assert centroid_bounds.min_point != centroid_bounds.max_point, "Error!!"

    n_buckets = 12
    buckets = [BucketInfo() for _ in range(n_buckets)]
    for i in range(start, end):
        centroid = (treelet_roots[i].bounds.min_point[dim] + treelet_roots[i].bounds.max_point[dim]) * 0.5
        b = int(n_buckets * ((centroid - centroid_bounds.min_point[dim]) / (
                centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim])))
        if b == n_buckets:
            b = n_buckets - 1
        assert 0 <= b < n_buckets
        buckets[b].count += 1
        buckets[b].bounds = enclose_volumes(buckets[b].bounds, treelet_roots[i].bounds)

    costs = []
    for i in range(n_buckets - 1):
        b0, b1 = None, None
        count0, count1 = 0, 0
        for j in range(i + 1):
            b0 = enclose_volumes(b0, buckets[j].bounds)
            count0 += buckets[j].count
        for j in range(i + 1, n_buckets):
            b1 = enclose_volumes(b1, buckets[j].bounds)
            count1 += buckets[j].count
        _cost = .125 + (count0 * get_surface_area(b0) + count1 * get_surface_area(b1)) / get_surface_area(bounds)
        costs.append(_cost)

    # find bucket to split at which minimizes SAH metric
    min_cost = costs[0]
    min_cost_split_bucket = 0
    for i in range(1, n_buckets - 1):
        if costs[i] < min_cost:
            min_cost = costs[i]
            min_cost_split_bucket = i

    # Split nodes and create interior HLBVH SAH node
    pred_wrapper = PartitionWrapper(n_buckets, centroid_bounds, dim, min_cost_split_bucket)
    mid = _partition(treelet_roots[start:end], pred_wrapper.partition_pred)
    mid += start

    assert mid > start, "Error: mid is not greater than start"
    assert mid < end, "Error: mid is not less than end"

    node.init_interior(
        dim,
        build_upper_sah(treelet_roots, start, mid, total_nodes),
        build_upper_sah(treelet_roots, mid, end, total_nodes))

    return node


#@numba.njit
def emit_lbvh(build_nodes, primitives, bounded_boxes, morton_prims, n_primitives, total_nodes,
              ordered_prims, ordered_prims_offset, bit_index):
    assert n_primitives > 0

    n_boxes = len(bounded_boxes)
    max_prims_in_node = int(0.1 * n_boxes)
    # max_prims_in_node = max_prims_in_node if max_prims_in_node < 4 else 4

    if bit_index == -1 or n_primitives < max_prims_in_node:
        # Create and return leaf node of LBVH treelet
        total_nodes[0] += 1
        node = build_nodes.pop(0)
        bounds = None
        first_prim_offset = ordered_prims_offset[0]
        ordered_prims_offset[0] += n_primitives
        for i in range(n_primitives):
            primitive_index = morton_prims[i].prim_ix
            ordered_prims[first_prim_offset + i] = primitives[primitive_index]
            bounds = enclose_volumes(bounds, bounded_boxes[primitive_index].bounds)
        node.init_leaf(first_prim_offset, n_primitives, bounds)
        return node
    else:
        mask = 1 << bit_index
        # Advance to next subtree level if there's no LBVH split for this bit
        if (morton_prims[0].morton_code & mask) == (morton_prims[n_primitives - 1].morton_code & mask):
            return emit_lbvh(build_nodes, primitives, bounded_boxes, morton_prims, n_primitives,
                             total_nodes, ordered_prims, ordered_prims_offset, bit_index - 1)

        # Find LBVH split point for this dimension
        # search_start = 0
        # search_end = n_primitives - 1
        # while search_start + 1 != search_end:
        #     assert search_start != search_end
        #     mid = (search_start + search_end) // 2
        #     if (morton_prims[search_start].morton_code & mask) == (morton_prims[mid].morton_code & mask):
        #         search_start = mid
        #     else:
        #         assert morton_prims[mid].morton_code & mask == morton_prims[search_end].morton_code & mask
        #         search_end = mid
        # split_offset = search_end

        interval_wrapper = IntervalWrapper(morton_prims, mask)
        split_offset = find_interval(n_primitives, interval_wrapper.interval_pred)

        # split_offset = find_interval(n_primitives, lambda index: (morton_prims[0].morton_code & mask) == (
        #             morton_prims[index].morton_code & mask))

        split_offset += 1

        assert split_offset <= n_primitives - 1
        assert (morton_prims[split_offset - 1].morton_code & mask) != (morton_prims[split_offset].morton_code & mask)

        # Create and return interior LBVH node
        total_nodes[0] += 1
        node = build_nodes.pop(0)
        lbvh = [
            emit_lbvh(build_nodes, primitives, bounded_boxes, morton_prims[:split_offset], split_offset,
                      total_nodes, ordered_prims, ordered_prims_offset, bit_index - 1),
            emit_lbvh(build_nodes, primitives, bounded_boxes, morton_prims[split_offset:],
                      n_primitives - split_offset, total_nodes, ordered_prims, ordered_prims_offset, bit_index - 1)
        ]
        axis = bit_index % 3
        node.init_interior(axis, lbvh[0], lbvh[1])
        return node


#@numba.njit
def build_hlbvh(primitives, bounded_boxes, ordered_prims, total_nodes):
    bounds = None
    for pi in bounded_boxes:
        bounds = enclose_centroids(bounds, pi.bounds.centroid)

    morton_prims = [] #numba.typed.List() # [MortonPrimitive() for _ in range(len(bounded_boxes))]
    for _ in range(len(bounded_boxes)):
        morton_prims.append(MortonPrimitive())
    morton_bits = 10
    morton_scale = 1 << morton_bits
    for i in range(len(bounded_boxes)):
        morton_prims[i].prim_ix = bounded_boxes[i].prim_num
        centroid_offset = offset_bounds(bounds, bounded_boxes[i].bounds.centroid)  # check if vector or scalar

        scaled_offset = centroid_offset * morton_scale

        morton_prims[i].morton_code = encode_morton_3(scaled_offset)

    radix_sort(morton_prims)

    treelets_to_build = []

    start = 0
    end = 1
    mask = 0b00111111111111000000000000000000

    while end <= len(morton_prims):
        if end == len(morton_prims) or (morton_prims[start].morton_code & mask != morton_prims[end].morton_code & mask):
            n_primitives = end - start
            max_bvh_nodes = int(2 * n_primitives - 1)
            # nodes = [BVHNode() for _ in range(max_bvh_nodes)]
            nodes = [] #numba.typed.List()
            for _ in range(max_bvh_nodes):
                nodes.append(BVHNode())
            treelets_to_build.append(LBVHTreelet(start, n_primitives, nodes))
            start = end
        end += 1

    atomic_total = 0
    ordered_prims_offset = [0]

    finished_treelets = []

    nodes_array = np.zeros(len(treelets_to_build), dtype=np.int32)

    # print(nodes_array, len(treelets_to_build), ordered_prims_offset)

    for i in range(len(treelets_to_build)):
        nodes_created = [0]
        first_bit_index = 29 - 12
        tr = treelets_to_build[i]

        # tr.build_nodes = emit_lbvh(tr.build_nodes, primitives, bounded_boxes,
        #                            morton_prims[tr.start_ix:], tr.n_primitives,
        #                            nodes_created, ordered_prims, ordered_prims_offset, first_bit_index)

        tr_build_nodes = emit_lbvh(tr.build_nodes, primitives, bounded_boxes, morton_prims[tr.start_ix:],
                                   tr.n_primitives,
                                   nodes_created, ordered_prims, ordered_prims_offset, first_bit_index)

        finished_treelets.append(tr_build_nodes)

        # atomic_total += nodes_created[0]
        nodes_array[i] = nodes_created[0]

        # print('nodes_created: ', nodes_created, ' ordered_prims_offset: ', ordered_prims_offset, ' n_primitives: ', tr.n_primitives)

    # total_nodes[0] = atomic_total
    total_nodes[0] = np.sum(nodes_array)

    # finished_treelets = [treelet.build_nodes for treelet in treelets_to_build]

    return build_upper_sah(finished_treelets, 0, len(finished_treelets),
                           total_nodes)  # primitives, bounded_boxes, start, end, ordered_prims, total_nodes