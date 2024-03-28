import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    KeepLargestConnectedComponentd,
    LoadImaged,
    MapLabelValued,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandRotated,
    RandSpatialCropd,
    RandZoomd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)

from salt.data.discovery import DatasetInfo

logger = logging.getLogger(__name__)


@dataclass(init=False)
class IntensityProperties:
    min: float
    max: float
    median: float
    mean: float
    std: float
    robust_lower_bound: float
    robust_upper_bound: float


class AddConstantsd(MapTransform):
    def __init__(self, keys: List[str], constants: List[Any]) -> None:
        super().__init__(keys, True)
        assert len(keys) == len(constants)
        self.constants = constants

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in zip(self.keys, self.constants):
            data[key] = value
        return data


def get_train_transforms(
    info: DatasetInfo,
    roi_size: Sequence[int],
    spacing: Sequence[float],
    intensity_properties: Optional[IntensityProperties] = None,
) -> Compose:
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LPI"),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=["bilinear", "nearest"],
            align_corners=[False, None],
        ),
        MapLabelValued(
            keys=["label"],
            orig_labels=range(256),
            target_labels=info.label_lut.tolist(),
            dtype=np.uint8,
        ),
        SpatialPadd(keys=["image"], spatial_size=roi_size, constant_values=-1024),
        SpatialPadd(keys=["label"], spatial_size=roi_size, constant_values=0),
        ScaleIntensityRanged(  # TODO: Probably should be an if/else with the other transforms
            keys=["image"],
            a_min=-1024,
            a_max=1024,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=[-1, -1, roi_size[-1]],
            random_size=False,
        ),
        RandZoomd(
            keys=["image", "label"],
            prob=1.0,
            min_zoom=(0.8, 0.8, 1.0),
            max_zoom=(1.2, 1.2, 1.0),
            mode=["trilinear", "nearest"],
            align_corners=[False, None],
        ),
        RandRotated(
            keys=["image", "label"],
            range_z=np.deg2rad(20),
            prob=1.0,
            mode=["bilinear", "nearest"],
            padding_mode="zeros",
            dtype=None,
        ),
        EnsureTyped(
            keys=["label"], dtype=np.uint8
        ),  # TODO Write issue about RandRotate output dtype
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=roi_size,
            random_size=False,
        ),
    ]
    if intensity_properties is not None:
        transforms += [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=1.0,
                b_min=-1024,
                b_max=3071,
                clip=True,
            ),
            NormalizeIntensityd(
                keys=["image"],
                subtrahend=intensity_properties.mean,
                divisor=intensity_properties.std,
            ),
        ]
    return Compose(
        transforms
        + [
            AddConstantsd(keys=["mask"], constants=[info.class_mask]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_validation_transforms(
    spacing: Sequence[float],
    info: DatasetInfo = None,
    intensity_properties: Optional[IntensityProperties] = None,
) -> Compose:
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LPI"),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=["bilinear", "nearest"],
            align_corners=[False, None],
        ),
    ]
    if info is not None:
        transforms += [
            MapLabelValued(
                keys=["label"],
                orig_labels=range(256),
                target_labels=info.label_lut.tolist(),
                dtype=np.uint8,
            ),
            AddConstantsd(keys=["mask"], constants=[info.class_mask]),
        ]
    if intensity_properties is not None:
        transforms += [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024,
                a_max=3071,
                b_min=-1024,
                b_max=3071,
                clip=True,
            ),
            NormalizeIntensityd(
                keys=["image"],
                subtrahend=intensity_properties.mean,
                divisor=intensity_properties.std,
            ),
        ]
    else:
        transforms += [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024,
                a_max=1024,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    return Compose(
        transforms
        + [
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_postprocess_transforms(output_dir: Optional[Path] = None):
    transforms = [
        KeepLargestConnectedComponentd(
            keys=["pred"],
            independent=False,
            is_onehot=False,
        ),
    ]
    if output_dir is not None:
        transforms += [
            SaveImaged(
                keys=["pred"],
                meta_keys="image_meta_dict",
                output_dir=output_dir,
                output_dtype=np.uint8,
                output_postfix="pred",
            ),
        ]
    return Compose(transforms)
