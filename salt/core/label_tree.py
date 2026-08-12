from typing import List, Optional, Tuple

import numpy as np

from salt.core import adjacency_matrix

"""
TODO
- Implement tree-like data structure for tree traversal
- Add tree optimizations (pruning, fusing, ...)
"""


class LabelTree:
    def __init__(self) -> None:
        self._adjacency_matrix = np.zeros((1, 1), dtype=np.uint8)
        self._vocabulary: List[Tuple[str, ...]] = [("<<ROOT>>",)]

    def add(self, *labels: str) -> None:
        parent: Tuple[str, ...] = ("<<ROOT>>",)
        for i in range(len(labels)):
            subpath = labels[: i + 1]
            if subpath not in self._vocabulary:
                self._vocabulary.append(subpath)
                self._adjacency_matrix = np.pad(
                    self._adjacency_matrix, [[0, 1], [0, 1]]
                )
            self._adjacency_matrix[
                self._vocabulary.index(parent), self._vocabulary.index(subpath)
            ] = 1
            parent = subpath

    def create_label_lut(
        self, labels: List[Optional[Tuple[str, ...]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        lut = np.zeros(256, dtype=np.uint8)
        lut[255] = 255
        mask = np.zeros(len(self._vocabulary) - 1, dtype=np.uint8)
        for source_idx, label in enumerate(labels):
            if not label:
                target_idx = 0
            else:
                target_idx = self._vocabulary.index(label) - 1
                mask[target_idx] = 1

            lut[source_idx] = target_idx

        # Check if subtrees are fully annotated and enable their parent label
        unique_sibling_groups = np.unique(
            adjacency_matrix.sibling_from_adjacency_matrix(self._adjacency_matrix)[
                1:, 1:
            ],
            axis=0,
        )
        changed = True
        while changed:
            changed = False
            for sibling_group in unique_sibling_groups:
                if np.equal(np.logical_and(sibling_group, mask), sibling_group).all():
                    cls_idx = np.where(sibling_group)[0][0]
                    parent_idx = np.where(self._adjacency_matrix[1:, 1:][:, cls_idx])[0]
                    if mask[parent_idx] == 0:
                        mask[parent_idx] = 1
                        changed = True

        return lut, mask

    def optimize(self) -> None:
        sort_indices = [0] + sorted(
            range(1, len(self._vocabulary)), key=lambda i: self._vocabulary[i]
        )
        self._vocabulary = [self._vocabulary[i] for i in sort_indices]
        self._adjacency_matrix = self._adjacency_matrix[sort_indices, :]
        self._adjacency_matrix = self._adjacency_matrix[:, sort_indices]

    @property
    def adjacency_matrix(self) -> np.ndarray:
        # Has to be symmetric matrix
        assert self._adjacency_matrix.ndim == 2
        assert self._adjacency_matrix.shape[0] == self._adjacency_matrix.shape[1]
        # Only binary edges are supported
        assert np.isin(self._adjacency_matrix, (0, 1)).all()
        # A directed tree with N nodes must have N - 1 edges
        assert self._adjacency_matrix.sum() == self._adjacency_matrix.shape[0] - 1
        # Ensure nodes have only one parent
        parent_count = self._adjacency_matrix.sum(axis=0)[1:]
        assert (parent_count == 1).all(), [
            self.labels[i] for i, c in enumerate(parent_count) if c != 1
        ]
        # Ensure nodes don't have a single children
        child_count = self._adjacency_matrix.sum(axis=1)[1:]
        assert (child_count != 1).all(), [
            self.labels[i] for i, c in enumerate(child_count) if c == 1
        ]

        return self._adjacency_matrix.copy()

    @property
    def leaf_indices(self) -> List[int]:
        row_sums = self._adjacency_matrix.sum(axis=1)
        return list(np.where(row_sums == 0)[0])

    @property
    def leaf_names(self) -> List[Tuple[str, ...]]:
        return [self._vocabulary[i] for i in self.leaf_indices]

    @property
    def labels(self) -> List[Tuple[str, ...]]:
        return list(self._vocabulary[1:])

    @property
    def num_classes(self) -> int:
        return len(self._vocabulary) - 1
