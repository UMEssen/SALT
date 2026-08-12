from typing import Dict, Hashable, Mapping

import torch
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms import MapTransform, Transform


class PruneToLeafNodes(Transform):
    def __init__(self, adjacency_matrix: NdarrayOrTensor) -> None:
        super().__init__()
        child_count = torch.as_tensor(adjacency_matrix[1:, 1:]).sum(1)
        self.leaf_indices = torch.nonzero(child_count == 0, as_tuple=True)[0]

    def __call__(self, data: NdarrayOrTensor) -> NdarrayOrTensor:
        tmp = torch.index_select(data, 0, self.leaf_indices.to(data.device))
        return tmp


class PruneToLeafNodesd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        adjacency_matrix: NdarrayOrTensor,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.converter = PruneToLeafNodes(adjacency_matrix)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def extend_binary_mask(mask):
    # Find the indices of True values in the mask
    true_indices = torch.argwhere(mask)
    print(true_indices)
    print(torch.min(true_indices, axis=0))
    exit()
    # Calculate the bounding box of the True values
    minx, miny, minz = torch.min(true_indices, axis=0)
    maxx, maxy, maxz = torch.max(true_indices, axis=0)

    # Calculate the amount to extend in each direction
    extendx = int(0.2 * (maxx - minx + 1))
    extendy = int(0.2 * (maxy - miny + 1))
    extendz = int(0.2 * (maxz - minz + 1))

    # Create a new mask with the extended region
    extended_mask = torch.zeros_like(mask)
    extended_mask[
        max(0, minx - extendx) : min(mask.shape[0] - 1, maxx + extendx) + 1,
        max(0, miny - extendy) : min(mask.shape[1] - 1, maxy + extendy) + 1,
        max(0, minz - extendz) : min(mask.shape[2] - 1, maxz + extendz) + 1,
    ] = True

    return extended_mask
