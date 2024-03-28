import argparse
import multiprocessing as mp
import pathlib
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from salt.data.conversion.lits import fix_image as fix_lits_image
from salt.data.conversion.utils import (
    Label,
    enforce_orientation,
    plot_volume,
    write_label_files,
)

LABELS = [
    Label(),
    Label("liver", ("body", "abdominal_cavity", "liver")),
    Label("bladder", ("body", "abdominal_cavity", "urinary_bladder")),
    Label("lungs", ("body", "thoracic_cavity", "lungs")),
    Label("kidneys", ("body", "abdominal_cavity", "kidneys")),
    Label("bones", ("body", "bones")),
    Label("brain", ("body", "brain")),
]


def handle_example(image_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    # Derive image ID and segmentation path
    image_id = int(image_path.with_suffix("").with_suffix("").name.split("-")[1])
    # if image_id <= 20:
    split_name = "test"
    # else:
    #     split_name = "train"
    label_path = image_path.parent / f"labels-{image_id}.nii.gz"

    # Load data
    image = sitk.ReadImage(str(image_path), sitk.sitkInt16)
    original_seg = sitk.ReadImage(str(label_path), sitk.sitkUInt8)

    # CT-ORG uses mostly the same data as LiTS and
    # suffers from the same metadata errors
    image, original_seg = fix_lits_image(image_id, image, original_seg)

    # Normalize orientation
    image = enforce_orientation(image)
    seg = enforce_orientation(original_seg)

    # Resample to uniform slice thickness of 5mm
    # image, seg = resample_to_thickness(image, original_seg)

    # Segmentations contain in both, train and test, flaws on the first or last slice
    # (seed points for morphological segmentation algorithms). Those will be masked
    # with an ignore label if they are still present after resampling.
    # seg = _mask_segmentation(seg, original_seg)

    # Save to output directory
    sitk.WriteImage(
        image, str(output_dir / split_name / "images" / f"{image_id:03d}.nii.gz")
    )
    sitk.WriteImage(
        seg, str(output_dir / split_name / "labels" / f"{image_id:03d}.nii.gz")
    )
    plot_volume(image, seg).savefig(
        output_dir / split_name / "previews" / f"{image_id:03d}.png"
    )
    plt.close()


def _mask_segmentation(seg: sitk.Image, orig_seg: sitk.Image) -> sitk.Image:
    y = sitk.GetArrayFromImage(orig_seg).astype(np.uint8)
    x = sitk.GetArrayFromImage(seg).astype(np.uint8)
    if np.equal(y[-1], x[-1]).all():
        x[-1] = 255
    elif np.equal(y[0], x[0]).all():
        x[0] = 255
    x = sitk.GetImageFromArray(x)
    x.CopyInformation(seg)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    # Create output directory structure
    # for split_name in {"train", "test"}:
    split_name = "test"
    (args.output_dir / split_name / "images").mkdir(exist_ok=True, parents=True)
    (args.output_dir / split_name / "labels").mkdir(exist_ok=True, parents=True)
    (args.output_dir / split_name / "previews").mkdir(exist_ok=True, parents=True)

    # Process all examples
    seg_paths = list(args.dataset_dir.rglob("volume-*.nii.gz"))
    # Only keep the ones between 000 and 020, the others are not good
    seg_paths = [
        p for p in seg_paths if int(p.name.split("-")[1].replace(".nii.gz", "")) < 21
    ]
    assert len(seg_paths) == 21
    # seg_paths = seg_paths[:3]
    with mp.Pool() as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(handle_example, output_dir=args.output_dir), seg_paths
            ),
            total=len(seg_paths),
            desc="Processing CT-ORG samples",
        ):
            pass

    # Generate label files
    write_label_files(args.output_dir, LABELS)
