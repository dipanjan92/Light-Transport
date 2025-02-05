'''
Standard Library Functions
'''
import numba


# def partition(arr, pred):
#     # Use two pointers to create partitions
#     i = 0  # The 'true' partition
#     j = len(arr) - 1  # The 'false' partition
#
#     # Start partitioning
#     while True:
#         # Advance the 'true' pointer
#         while (i < j) and pred(arr[i]):
#             i += 1
#         # Advance the 'false' pointer
#         while (i < j) and not pred(arr[j]):
#             j -= 1
#
#         # If pointers have crossed, partition is complete
#         if i >= j:
#             break
#
#         # Swap elements
#         arr[i], arr[j] = arr[j], arr[i]
#
#     # Return the partition point
#     return i


@numba.njit
def partition(arr, pred, first=0, last=None):
    # if last is None, consider the whole array
    if last is None:
        last = len(arr)

    # i: start of the false partition
    i = first
    for j in range(first, last):
        if pred(arr[j]):
            # if the predicate function returns true,
            # swap the current element with the first element of the false partition
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    # i is now pointing to the first false element
    return i


def _partition(arr, pred, first=0, last=None):
    # if last is None, consider the whole array
    if last is None:
        last = len(arr)

    # i: start of the false partition
    i = first
    for j in range(first, last):
        if pred(arr[j]):
            # if the predicate function returns true,
            # swap the current element with the first element of the false partition
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    # i is now pointing to the first false element
    return i


@numba.njit
def mid_point_partition(arr, midpoint_partition, pmid):
    left_ptr, right_ptr = 0, len(arr) - 1
    while True:
        while left_ptr <= right_ptr and midpoint_partition.mid_point_pred(arr[left_ptr], pmid):
            left_ptr += 1
        while right_ptr >= 0 and not midpoint_partition.mid_point_pred(arr[right_ptr], pmid):
            right_ptr -= 1
        if left_ptr <= right_ptr:
            arr[left_ptr], arr[right_ptr] = arr[right_ptr], arr[left_ptr]
        else:
            return left_ptr


@numba.njit
def nth_element(arr, n, first=0, last=None, key=lambda x: x):
    if last is None:
        last = len(arr) - 1
    if first < last:
        pi = partition(arr, key, first, last)
        if pi > n:
            nth_element(arr, n, first, pi - 1, key)
        elif pi < n:
            nth_element(arr, n, pi + 1, last, key)
        else:
            return arr[n]


@numba.njit
def find_interval(sz, pred):
    size = sz - 2
    first = 1

    while size > 0:
        # Evaluate predicate at midpoint and update first and size
        half = size >> 1
        middle = first + half
        pred_result = pred(middle)

        if pred_result:
            first = middle + 1
        else:
            size = half

    return max(min(first - 1, sz - 2), 0)
