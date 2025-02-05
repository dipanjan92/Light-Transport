import numba
import numpy as np

from accelerators.bvh import BVHNode, enclose_volumes, enclose_centroids, get_largest_dim, \
    MidPointWrapper, PredicateWrapper, get_surface_area, offset_bounds, BucketInfo
from numba import int32, float32, types
from numba.typed import List

from utils.stdlib import mid_point_partition, partition


@numba.njit
def build_bvh(primitives, bounded_boxes, start, end, ordered_prims, total_nodes, split_method=0):
    n_boxes = len(bounded_boxes)
    max_prims_in_node = max(4, int(0.1 * n_boxes))

    stack = [(start, end, -1, False)]
    nodes = []

    while stack:
        start, end, parent_idx, is_second_child = stack.pop()

        # print(start, end, parent_idx, is_second_child)

        node = BVHNode()
        current_node_idx = len(nodes)
        total_nodes[0] += 1

        if parent_idx != -1:
            parent = nodes[parent_idx]
            if is_second_child:
                parent.child_1 = node
            else:
                parent.child_0 = node

        bounds = None
        for i in range(start, end):
            bounds = enclose_volumes(bounds, bounded_boxes[i].bounds)

        # print(bounds.centroid)

        n_primitives = end - start

        if n_primitives == 1:
            first_prim_offset = len(ordered_prims)
            for i in range(start, end):
                prim_num = bounded_boxes[i].prim_num
                ordered_prims.append(primitives[prim_num])
            node.init_leaf(first_prim_offset, n_primitives, bounds)
        else:
            centroid_bounds = None
            for i in range(start, end):
                centroid_bounds = enclose_centroids(centroid_bounds, bounded_boxes[i].bounds.centroid)

            dim = get_largest_dim(centroid_bounds)

            mid = (start + end) // 2

            print(centroid_bounds.centroid, dim, mid)

            if np.all(centroid_bounds.max_point == centroid_bounds.min_point):
                first_prim_offset = len(ordered_prims)
                for i in range(start, end):
                    prim_num = bounded_boxes[i].prim_num
                    ordered_prims.append(primitives[prim_num])
                node.init_leaf(first_prim_offset, n_primitives, bounds)

            else:
                if split_method == 0:
                    # Partition primitives based on Surface Area Heuristic
                    if n_primitives <= 2:
                        # Partition primitives into equally sized subsets
                        mid = (start + end) // 2
                        # nth_element(bounded_boxes, mid, first=start, last=end,
                        #             key=lambda x: x.bounds.centroid[dim])

                        nth_element_with_dim(bounded_boxes, mid, dim, start, end)

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
                            nodes.append(node)
                            continue
                elif split_method == 1:
                    # Middle split
                    pmid = (centroid_bounds.min_point[dim] + centroid_bounds.max_point[dim]) / 2
                    midpoint_wrapper = MidPointWrapper(dim)
                    mid_ptr = mid_point_partition(bounded_boxes[start:end], midpoint_wrapper, pmid)
                    mid = mid_ptr + start
                    if mid == start or mid == end:
                        mid = (start + end) // 2
                        nth_element_with_dim(bounded_boxes, mid, dim, start, end)
                else:
                    # Equal parts
                    mid = (start + end) // 2
                    # nth_element_with_dim(bounded_boxes, mid, dim, first=start, last=end)
                    nth_element_with_dim(bounded_boxes, mid, dim, start, end)

                node.split_axis = dim
                node.n_primitives = 0
                node.bounds = bounds
                stack.append((mid, end, current_node_idx, True))
                stack.append((start, mid, current_node_idx, False))

        nodes.append(node)

    return nodes[0]  # Return the root node


@numba.njit
def flatten_bvh(node_list, root):
    stack = [(root, -1, False)]  # (node, parent_idx, is_second_child)
    offset = 0

    while stack:
        node, parent_idx, is_second_child = stack.pop()

        if node is None:
            continue

        current_idx = offset
        linear_node = node_list[current_idx]
        linear_node.bounds = node.bounds
        offset += 1

        if parent_idx != -1:
            parent_node = node_list[parent_idx]
            if is_second_child:
                parent_node.second_child_offset = current_idx

        if node.n_primitives > 0:
            assert node.child_0 is None and node.child_1 is None, "Both children None"
            assert node.n_primitives < 65536, "n_primitives LT 65536"
            linear_node.primitives_offset = node.first_prim_offset
            linear_node.n_primitives = node.n_primitives
        else:
            linear_node.axis = node.split_axis
            linear_node.n_primitives = 0
            linear_node.primitives_offset = 0

            # Push children onto stack in reverse order (so left child is processed first)
            if node.child_1 is not None:
                stack.append((node.child_1, current_idx, True))
            if node.child_0 is not None:
                stack.append((node.child_0, current_idx, False))

    return offset  # Total number of nodes



@numba.njit
def nth_element_with_dim(bounded_boxes, start, end, mid, dim):
    left = start
    right = end - 1

    while left < right:
        pivot_index = (left + right) // 2
        pivot_value = bounded_boxes[pivot_index].bounds.centroid[dim]

        # Swap pivot to the end
        bounded_boxes[pivot_index], bounded_boxes[right] = bounded_boxes[right], bounded_boxes[pivot_index]

        store_index = left
        for i in range(left, right):
            if bounded_boxes[i].bounds.centroid[dim] < pivot_value:
                bounded_boxes[i], bounded_boxes[store_index] = bounded_boxes[store_index], bounded_boxes[i]
                store_index += 1

        # Move pivot to its final place
        bounded_boxes[store_index], bounded_boxes[right] = bounded_boxes[right], bounded_boxes[store_index]

        if store_index == mid:
            break
        elif store_index < mid:
            left = store_index + 1
        else:
            right = store_index - 1