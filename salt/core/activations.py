from typing import Optional

import numpy as np
import torch
from torch import nn

from salt.core.adjacency_matrix import (
    reachability_from_adjacency_matrix,
    sibling_from_adjacency_matrix,
)


class TreeSoftmax(nn.Module):
    def __init__(self, adjacency_matrix: np.ndarray, dim: int = 1) -> None:
        # Has to be symmetric matrix
        assert adjacency_matrix.ndim == 2
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
        # Only binary edges are supported
        assert np.isin(adjacency_matrix, (0, 1)).all()
        # A directed tree with N nodes must have N - 1 edges
        assert adjacency_matrix.sum() == adjacency_matrix.shape[0] - 1
        # Ensure nodes have only one parent
        assert (adjacency_matrix.sum(axis=0)[1:] == 1).all()
        # TODO Requires a check if nodes exist with only a single child node

        super().__init__()
        self.adjacency_matrix = adjacency_matrix
        self.reachability_matrix = reachability_from_adjacency_matrix(adjacency_matrix)
        self.sibling_matrix = sibling_from_adjacency_matrix(adjacency_matrix)
        self.axis = dim

    def forward(
        self, input: torch.Tensor, dim: Optional[int] = None, propagate: bool = True
    ) -> torch.Tensor:
        # Compute sibling normalized probabilities (numerical stable)
        result = [None] * (self.adjacency_matrix.shape[0] - 1)
        # result = torch.empty_like(input)
        for row_idx in range(self.adjacency_matrix.shape[0]):
            sibling_indices = (
                np.where(self.adjacency_matrix[row_idx, :])[0] - 1
            )  # Important: Root node is missing in probability vector
            if len(sibling_indices) > 0:
                g = torch.index_select(
                    input,
                    dim or self.axis,
                    torch.as_tensor(sibling_indices, device=input.device),
                )
                p_max, _ = g.max(self.axis, keepdim=True)
                g_exp = torch.exp(g - p_max)
                p_sum = g_exp.sum(self.axis, keepdim=True)
                p = g_exp / p_sum
                for tensor_idx, sibling_idx in enumerate(sibling_indices):
                    # result[:, sibling_idx : sibling_idx + 1] = torch.index_select(
                    result[sibling_idx] = torch.index_select(
                        input=p,
                        dim=dim or self.axis,
                        index=torch.as_tensor([tensor_idx], device=p.device),
                    )

        # Propagate probabilites
        if propagate:
            nodes_to_process: list[int] = list(np.where(self.adjacency_matrix[0, :])[0])
            while len(nodes_to_process) > 0:
                child_nodes: list[int] = []
                for parent_idx in nodes_to_process:
                    child_indices = np.where(self.adjacency_matrix[parent_idx, :])[0]
                    child_nodes.extend(child_indices)
                    for child_idx in child_indices:
                        result[child_idx - 1] *= result[parent_idx - 1]
                        # result[:, child_idx - 1 : child_idx] *= result[
                        #     :, parent_idx - 1 : parent_idx
                        # ]
                nodes_to_process = child_nodes
        return torch.cat(
            [
                torch.cat(result[i : i + 32], dim=dim or self.axis)
                for i in range(0, len(result), 32)
            ],
            dim=dim or self.axis,
        )
        # return result
