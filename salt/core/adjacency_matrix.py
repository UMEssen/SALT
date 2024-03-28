from functools import reduce
from operator import ior
from typing import Any, Tuple

import numpy as np


def reachability_from_adjacency_matrix(adjacency_matrix: np.ndarray) -> np.ndarray:
    result = np.eye(adjacency_matrix.shape[0], dtype=adjacency_matrix.dtype)
    power = adjacency_matrix
    while power.any():
        result += power
        power = np.matmul(power, adjacency_matrix)
    return result


def sibling_from_adjacency_matrix(adjacency_matrix: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.matmul(adjacency_matrix.T, adjacency_matrix)
    return result


def bitpack_tree(adjacency_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    reachability_matrix = reachability_from_adjacency_matrix(adjacency_matrix)

    # Get node indices within each softmax group
    idx_in_groups = np.cumsum(adjacency_matrix, axis=1) * adjacency_matrix

    # Compute required bits
    max_idx_in_groups = idx_in_groups.max(axis=1)
    required_bits_per_group = np.zeros_like(max_idx_in_groups, dtype=np.uint64)
    required_bits_per_group[max_idx_in_groups > 0] = np.floor(
        np.log2(max_idx_in_groups[max_idx_in_groups > 0]) + 1
    ).astype(np.uint64)
    required_bits = required_bits_per_group.cumsum()

    # Check which data type to choose based on required number of bits
    num_required_bits = int(required_bits[-1])
    print("Required number of bits:", num_required_bits)
    # if num_required_bits > 64:
    return _bitpack_tree_largeint(
        adjacency_matrix=adjacency_matrix,
        reachability_matrix=reachability_matrix,
        num_required_bits=num_required_bits,
        required_bits=required_bits,
        required_bits_per_group=required_bits_per_group,
        idx_in_groups=idx_in_groups,
    )
    # else:
    #     return _bitpack_tree_primitive(
    #         adjacency_matrix=adjacency_matrix,
    #         reachability_matrix=reachability_matrix,
    #         num_required_bits=num_required_bits,
    #         required_bits=required_bits,
    #         required_bits_per_group=required_bits_per_group,
    #         idx_in_groups=idx_in_groups,
    #     )


def _bitpack_tree_primitive(
    adjacency_matrix: np.ndarray,
    reachability_matrix: np.ndarray,
    num_required_bits: int,
    required_bits: np.ndarray,
    required_bits_per_group: np.ndarray,
    idx_in_groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_required_bits <= 8:
        dtype = np.int8
    elif num_required_bits <= 16:
        dtype = np.int16  # type: ignore
    elif num_required_bits <= 32:
        dtype = np.int32  # type: ignore
    else:
        dtype = np.int64  # type: ignore

    print(f"Choosing {dtype.__name__} as data type")  # type: ignore

    # Remove bits from first group
    bit_shifts = np.pad(required_bits, (1, 0))[:-1]

    # Shift group internal indices by the amount of already allocated bits
    shifted_idx_in_groups = np.left_shift(
        idx_in_groups.astype(dtype), bit_shifts[:, np.newaxis].astype(dtype)
    )

    # Matrix multiply with reachability matrix to get all required partial bitmasks
    tmp = np.matmul(shifted_idx_in_groups, reachability_matrix)

    # Combine all partial bitmasks along the path via binary or operation
    bitmask_classes = np.bitwise_or.reduce(tmp, axis=0)

    # Build bitmask for masking relevant bits of sibling and parent nodes
    foo = required_bits_per_group[:, np.newaxis] * adjacency_matrix
    shifted_mask_in_groups = np.left_shift(
        (np.power(2, foo) - 1).astype(dtype), bit_shifts[:, np.newaxis].astype(dtype)
    )
    tmp = np.matmul(shifted_mask_in_groups, reachability_matrix)
    bitmask_groups = np.bitwise_or.reduce(tmp, axis=0)

    return bitmask_classes, bitmask_groups


def _bitpack_tree_largeint(
    adjacency_matrix: np.ndarray,
    reachability_matrix: np.ndarray,
    num_required_bits: int,
    required_bits: np.ndarray,
    required_bits_per_group: np.ndarray,
    idx_in_groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    def _matmul(a: Any, b: Any) -> Any:
        zip_b = list(zip(*b))
        return [
            [
                sum(ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b))
                for col_b in zip_b
            ]
            for row_a in a
        ]

    # Remove bits from first group
    bit_shifts = np.pad(required_bits, (1, 0))[:-1]

    # Shift group internal indices by the amount of already allocated bits
    num_bytes = (num_required_bits + 7) // 8

    shifted_idx_in_groups = [
        [int(xx) << int(s) for xx in x] for x, s in zip(idx_in_groups, bit_shifts)
    ]

    # Matrix multiply with reachability matrix to get all required partial bitmasks
    tmp = _matmul(
        shifted_idx_in_groups,
        [[int(xx) for xx in x] for x in reachability_matrix.tolist()],
    )

    # Combine all partial bitmasks along the path via binary or operation
    bitmask_classes = [reduce(ior, [x[i] for x in tmp]) for i in range(len(tmp))]
    bitmask_classes = np.asarray(
        [list(x.to_bytes(num_bytes, "little")[::-1]) for x in bitmask_classes],
        dtype=np.uint8,
    )

    # Build bitmask for masking relevant bits of sibling and parent nodes
    foo = required_bits_per_group[:, np.newaxis] * adjacency_matrix
    shifted_mask_in_groups = [
        [(2 ** int(ff) - 1) << int(s) for ff in f] for s, f in zip(bit_shifts, foo)
    ]

    tmp = _matmul(
        shifted_mask_in_groups,
        [[int(xx) for xx in x] for x in reachability_matrix.tolist()],
    )
    bitmask_groups = [
        reduce(ior, [tmp[i][j] for i in range(len(tmp))]) for j in range(len(tmp))
    ]
    bitmask_groups = np.asarray(
        [list(x.to_bytes(num_bytes, "little")[::-1]) for x in bitmask_groups],
        dtype=np.uint8,
    )

    return bitmask_classes, bitmask_groups
