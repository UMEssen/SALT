import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import skimage
from nibabel.spatialimages import SpatialImage


@dataclass
class Label:
    original: Optional[str] = None
    tree: Optional[Tuple[str, ...]] = None


FLIP_XY = np.diag((-1, -1, 1))


def _nib_to_sitk(image: SpatialImage) -> sitk.Image:
    # adapted from https://github.com/fepegar/torchio/commit/6080a6fd2793244e5552414c4a0de6cb328c75be
    data = np.asanyarray(image.dataobj)
    affine = image.affine

    origin = np.dot(FLIP_XY, affine[:3, 3]).astype(np.float64)
    RZS = affine[:3, :3]
    spacing = np.sqrt(np.sum(RZS * RZS, axis=0))
    R = RZS / spacing
    direction = np.dot(FLIP_XY, R).flatten()
    image = sitk.GetImageFromArray(data.transpose())
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    return image


def _sitk_to_nib(image: sitk.Image) -> SpatialImage:
    # adapted from https://github.com/fepegar/torchio/commit/6080a6fd2793244e5552414c4a0de6cb328c75be
    data = sitk.GetArrayFromImage(image).transpose()
    spacing = np.array(image.GetSpacing())
    R = np.array(image.GetDirection()).reshape(3, 3)
    R = np.dot(FLIP_XY, R)
    RZS = R * spacing
    translation = np.dot(FLIP_XY, image.GetOrigin())
    affine = np.eye(4)
    affine[:3, :3] = RZS
    affine[:3, 3] = translation
    return SpatialImage(data, affine)


def enforce_orientation(
    image: sitk.Image, axcodes: Tuple[str, ...] = ("L", "P", "I")
) -> sitk.Image:
    nib_image = _sitk_to_nib(image)
    input_axcodes = nib.aff2axcodes(nib_image.affine)
    input_ornt = nib.orientations.axcodes2ornt(input_axcodes)
    expected_ornt = nib.orientations.axcodes2ornt(axcodes)
    transform = nib.orientations.ornt_transform(input_ornt, expected_ornt)
    return _nib_to_sitk(nib_image.as_reoriented(transform))


def resample_to_thickness(
    image: sitk.Image, seg: sitk.Image, thickness: float = 5
) -> Tuple[sitk.Image, sitk.Image]:
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    output_spacing = (input_spacing[0], input_spacing[1], thickness)
    output_direction = image.GetDirection()
    output_origin = image.GetOrigin()
    output_size = [
        int(np.round(input_size[i] * input_spacing[i] / output_spacing[i]))
        for i in range(3)
    ]
    image = sitk.Resample(
        image,
        output_size,
        sitk.Transform(),
        sitk.sitkLinear,
        output_origin,
        output_spacing,
        output_direction,
    )
    seg = sitk.Resample(
        seg,
        output_size,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        output_origin,
        output_spacing,
        output_direction,
    )
    return image, seg


def plot_volume(image: sitk.Image, seg: sitk.Image) -> plt.Figure:
    image_data = sitk.GetArrayViewFromImage(image)
    seg_data = sitk.GetArrayViewFromImage(seg)
    sx, sy, sz = image.GetSpacing()
    cz = image_data.shape[0] // 2
    cy = image_data.shape[1] // 2
    cx = image_data.shape[2] // 2
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(
        _create_overlay_image(image_data[cz], seg_data[cz]),
        aspect=sy / sx,
        interpolation="nearest",
    )
    axs[1].imshow(
        _create_overlay_image(image_data[:, cy], seg_data[:, cy]),
        aspect=sz / sx,
        interpolation="nearest",
    )
    axs[2].imshow(
        _create_overlay_image(image_data[:, :, cx], seg_data[:, :, cx]),
        aspect=sz / sy,
        interpolation="nearest",
    )
    fig.tight_layout()
    return fig


def _create_overlay_image(
    image_data: np.ndarray,
    seg_data: np.ndarray,
    center: int = 50,
    width: int = 400,
    alpha: float = 0.25,
) -> np.ndarray:
    image_data = (image_data - (center - width / 2)) / width
    image_data = np.clip(image_data, 0.0, 1.0)

    result: np.ndarray = np.where(
        seg_data[..., np.newaxis] == 0,
        image_data[..., np.newaxis],
        (1 - alpha) * image_data[..., np.newaxis] + alpha * VOC_PALETTE[seg_data],
    )
    result = (result * 255).astype(np.uint8)
    return result


def _make_palette() -> np.ndarray:
    """
    From: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/data/pascal/convert_sbdd.py

    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """
    palette = np.zeros((256, 3), dtype=np.uint8)
    for k in range(256):
        label = k
        i = 0
        while label:
            palette[k, 0] |= ((label >> 0) & 1) << (7 - i)
            palette[k, 1] |= ((label >> 1) & 1) << (7 - i)
            palette[k, 2] |= ((label >> 2) & 1) << (7 - i)
            label >>= 3
            i += 1
    return palette.astype(np.float32) / 255


VOC_PALETTE = _make_palette()


def write_label_files(output_dir: pathlib.Path, labels: List[Label]) -> None:
    with (output_dir / "labels.txt").open("w") as ofile:
        ofile.write("\n".join([x.original or "-" for x in labels]))

    with (output_dir / "tree-labels.txt").open("w") as ofile:
        ofile.write("\n".join([",".join(x.tree or "-") for x in labels]))


def split_long_organs(
    seg_ts: sitk.Image,
    seg_br: sitk.Image,
    label_names: List[str],
):
    if seg_ts.GetSize() != seg_br.GetSize():
        seg_br = sitk.Resample(
            seg_br, seg_ts, sitk.Transform(), sitk.sitkNearestNeighbor, 0
        )

    # print(label_names)
    seg_ts_data = sitk.GetArrayFromImage(seg_ts)
    seg_br_data = sitk.GetArrayViewFromImage(seg_br)

    abdominal_cavity_mask = np.equal(seg_br_data, 3)
    pericardium_mask = np.equal(seg_br_data, 7)
    # mediastinum_mask = np.equal(seg_br_data, 9)

    if "aorta" in label_names:
        # Split aorta into abdominal and thoracic part
        aorta_idx = label_names.index("aorta")
        aorta_mask = np.equal(seg_ts_data, aorta_idx)
        aorta_abdominalis_mask = np.logical_and(aorta_mask, abdominal_cavity_mask)
        aorta_thoracica_pass_pericardium = np.logical_and(aorta_mask, pericardium_mask)
        seg_ts_data[aorta_abdominalis_mask] = len(label_names)
        seg_ts_data[aorta_thoracica_pass_pericardium] = len(label_names) + 1

    if "inferior_vena_cava" in label_names:
        # Split inferior vena cava into abdominal and thoracic part
        ivc_idx = label_names.index("inferior_vena_cava")
        ivc_mask = np.equal(seg_ts_data, ivc_idx)
        ivc_abdominalis_mask = np.logical_and(ivc_mask, abdominal_cavity_mask)
        seg_ts_data[ivc_abdominalis_mask] = len(label_names) + 2

    # Filter iliac artery/vena to be only present in abdominal cavity
    for label_name in {"iliac_artery", "iliac_vena"}:
        if label_name not in label_names:
            continue
        for side in {"left", "right"}:
            label_idx = label_names.index(f"{label_name}_{side}")
            seg_ts_data[
                np.logical_and(np.equal(seg_ts_data, label_idx), ~abdominal_cavity_mask)
            ] = 0

    if "pulmonary_artery" in label_names:
        # Split pulmunary artery into mediastinum and pericardium part
        pulmonary_artery_idx = label_names.index("pulmonary_artery")
        pulmonary_artery_mask = np.equal(seg_ts_data, pulmonary_artery_idx)
        pulmonary_artery_pass_pericardium_mask = np.logical_and(
            pulmonary_artery_mask, pericardium_mask
        )
        seg_ts_data[pulmonary_artery_pass_pericardium_mask] = len(label_names) + 3

    # Save updates segmentation
    new_seg = sitk.GetImageFromArray(seg_ts_data)
    new_seg.CopyInformation(seg_ts)
    return new_seg


def keep_largest_areas(segmentation: np.ndarray, label_names: List[str]) -> None:
    for label_idx in range(1, len(label_names)):
        mask = segmentation == label_idx
        props = skimage.measure.regionprops(skimage.measure.label(mask))
        if len(props) > 1:
            props = sorted(props, key=lambda x: x.area, reverse=True)
            print(label_names[label_idx], [x.area for x in props])
