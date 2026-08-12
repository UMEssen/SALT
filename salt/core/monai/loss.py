# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Union

import numpy as np
import torch
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss

from salt.core.adjacency_matrix import bitpack_tree


class HierarchyAwareDiceLoss(_Loss):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        self.adjacency_matrix = adjacency_matrix
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

        bitmask_classes, bitmask_groups = bitpack_tree(adjacency_matrix)
        self.bitmask_classes = torch.as_tensor(bitmask_classes)
        self.bitmask_groups = torch.as_tensor(bitmask_groups)

    def _bitwise_mask_equal(
        self, x: torch.Tensor, bm_class: torch.Tensor, bm_group: torch.Tensor
    ) -> torch.Tensor:
        comp = torch.eq(torch.bitwise_and(x, bm_group), bm_class)
        if comp.ndim == 1:
            return comp
        assert comp.ndim == 2
        return torch.all(comp, axis=1)

    def _argmax_leafs(self, yhat: torch.Tensor) -> torch.Tensor:
        leaf_nodes = np.where(self.adjacency_matrix[1:, 1:].sum(axis=1) == 0)[0]
        indices = np.arange(self.adjacency_matrix.shape[0] - 1, dtype=np.int32)
        indices = indices[leaf_nodes]
        y_pred_leafs = yhat[:, leaf_nodes]
        y_pred_leaf_idx = torch.argmax(y_pred_leafs, axis=1)
        return torch.tensor(indices).to(yhat.device)[y_pred_leaf_idx]

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, class_mask: torch.Tensor
    ) -> torch.Tensor:
        if target.shape != input.shape:
            raise AssertionError(
                f"ground truth has different shape ({target.shape}) from input ({input.shape})"
            )

        bc = self.bitmask_classes.to(input.device)
        bg = self.bitmask_groups.to(input.device)
        class_indices = torch.nonzero(class_mask)[:, 0]

        # Compute leaf predictions and convert to bitmask
        pred = self._argmax_leafs(input.detach())
        yhat_bm = torch.index_select(bc, 0, pred)

        # Compute masks based on adjacency matrix
        intersections: List[torch.Tensor] = []
        input_sums: List[torch.Tensor] = []
        target_sums: List[torch.Tensor] = []
        computed_masks: Dict[int, torch.Tensor] = {}
        class_indices_np = class_indices.cpu().numpy().ravel()
        for class_idx in class_indices_np:
            parent_idx: int = self.adjacency_matrix[:, class_idx + 1].argmax()  # type: ignore

            if parent_idx == 0:
                sub_input = input[:, class_idx]
                sub_target = target[:, class_idx]
            else:
                # computed_masks are cached in case other labels have the same parent
                if parent_idx not in computed_masks:
                    if parent_idx - 1 in class_indices_np:
                        # this dataset should give full information about the parent
                        # (all children are in its label set)
                        computed_masks[parent_idx] = target[:, parent_idx - 1].bool()
                    else:
                        computed_masks[parent_idx] = self._bitwise_mask_equal(
                            yhat_bm, bc[parent_idx], bg[parent_idx]
                        )
                sub_input = input[computed_masks[parent_idx], class_idx]
                sub_target = target[computed_masks[parent_idx], class_idx]

            if self.squared_pred:
                sub_target = torch.pow(sub_target, 2)
                sub_input = torch.pow(sub_input, 2)

            intersections.append(torch.sum(sub_target * sub_input))
            input_sums.append(torch.sum(sub_input))
            target_sums.append(torch.sum(sub_target))

        # Compute dice components
        intersection = torch.stack(intersections)
        ground_o = torch.stack(target_sums)
        pred_o = torch.stack(input_sums)

        # mask = torch.empty((input.shape[0], len(class_indices)), dtype=input.dtype, device=input.device)
        # computed_masks: Dict[int, torch.Tensor] = {}
        # for idx, class_idx in enumerate(class_indices.cpu().numpy().ravel()):
        #     parent_idx = self.adjacency_matrix[:, class_idx + 1].argmax()
        #     if parent_idx == 0:
        #         mask[:, idx] = 1
        #     else:
        #         if parent_idx not in computed_masks:
        #             if parent_idx - 1 in class_indices.cpu().numpy().ravel():
        #                 computed_masks[parent_idx] = target[:, parent_idx - 1].bool()
        #             else:
        #                 computed_masks[parent_idx] = self._bitwise_mask_equal(yhat_bm, bc[parent_idx], bg[parent_idx])
        #         mask[:, idx] = computed_masks[parent_idx]

        # input = input[:, class_indices] * mask
        # target = target[:, class_indices] * mask

        # intersection = torch.sum(target * input)
        # ground_o = torch.sum(target)
        # pred_o = torch.sum(input)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        # Compute final loss
        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (
            denominator + self.smooth_dr
        )

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
            )

        return f
