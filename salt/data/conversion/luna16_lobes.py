import argparse
import multiprocessing as mp
import pathlib
from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import skimage.measure
from tqdm import tqdm

from salt.data.conversion.utils import (
    Label,
    enforce_orientation,
    plot_volume,
    write_label_files,
)

LABELS = [
    Label(),
    Label("trachea", None),
    Label(
        "upper_right",
        ("body", "thoracic_cavity", "lungs", "lung_right", "lung_upper_lobe_right"),
    ),
    Label(
        "middle_right",
        ("body", "thoracic_cavity", "lungs", "lung_right", "lung_middle_lobe_right"),
    ),
    Label(
        "lower_right",
        ("body", "thoracic_cavity", "lungs", "lung_right", "lung_lower_lobe_right"),
    ),
    Label(
        "upper_left",
        ("body", "thoracic_cavity", "lungs", "lung_left", "lung_upper_lobe_left"),
    ),
    Label(
        "lower_left",
        ("body", "thoracic_cavity", "lungs", "lung_left", "lung_lower_lobe_left"),
    ),
]


def handle_example(
    job_args: Tuple[int, pathlib.Path],
    luna16_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    image_id, seg_path = job_args

    # Derive case ID and image path
    case_id = seg_path.name.split("_")[0]
    image_paths = list(luna16_dir.rglob(f"subset*/{case_id}.mhd"))
    assert len(image_paths) == 1, len(image_paths)
    image_path = image_paths[0]

    # Load data (segmentation has more than 8 Bits)
    image = sitk.ReadImage(str(image_path), sitk.sitkInt16)
    seg = sitk.ReadImage(str(seg_path), sitk.sitkUInt16)

    # Normalize orientation
    image = enforce_orientation(image)
    seg = enforce_orientation(seg)

    # Segmentations contain overlapping bit masks and should be converted.
    # 512=trachea; 4/5/6 right lung upper to lower; 7/8 left lung upper to lower
    seg_data = sitk.GetArrayFromImage(seg)
    new_seg_data = np.zeros_like(seg_data, dtype=np.uint8)
    new_seg_data[_get_largest_region_mask(seg_data == 512)] = 1
    new_seg_data[seg_data == 4] = 2
    new_seg_data[seg_data == 5] = 3
    new_seg_data[seg_data == 6] = 4
    new_seg_data[seg_data == 7] = 5
    new_seg_data[seg_data == 8] = 6
    new_seg = sitk.GetImageFromArray(new_seg_data)
    new_seg.CopyInformation(seg)

    # Resample to uniform slice thickness of 5mm
    # image, new_seg = resample_to_thickness(image, new_seg, 5.0)

    # Save to output directory
    split_name = "test"  # "train" if image_id <= 40 else "test"
    output_image_path = output_dir / split_name / "images" / f"{case_id}.nii.gz"
    sitk.WriteImage(image, str(output_image_path))
    output_seg_path = output_dir / split_name / "labels" / f"{case_id}.nii.gz"
    sitk.WriteImage(new_seg, str(output_seg_path))
    plot_volume(image, new_seg).savefig(
        output_dir / split_name / "previews" / f"{case_id}.png"
    )
    plt.close()


def _get_largest_region_mask(mask: np.ndarray) -> np.ndarray:
    label = skimage.measure.label(mask)
    props = skimage.measure.regionprops(label)
    props = sorted(props, key=lambda x: int(x.area), reverse=True)
    result: np.ndarray = np.equal(label, props[0].label)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--luna16-dir", required=True, type=pathlib.Path)
    parser.add_argument("--annotation-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    # Create output directory structure
    # for split_name in {"train", "test"}:
    split_name = "test"
    (args.output_dir / split_name / "images").mkdir(exist_ok=True, parents=True)
    (args.output_dir / split_name / "labels").mkdir(exist_ok=True, parents=True)
    (args.output_dir / split_name / "previews").mkdir(exist_ok=True, parents=True)

    # Process all examples
    seg_paths = sorted(list(args.annotation_dir.rglob("*.nrrd")))
    with mp.Pool() as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(
                    handle_example,
                    luna16_dir=args.luna16_dir,
                    output_dir=args.output_dir,
                ),
                enumerate(seg_paths),
            ),
            total=len(seg_paths),
            desc="Processing LUNA16 samples",
        ):
            pass

    # Generate label files
    write_label_files(args.output_dir, LABELS)
