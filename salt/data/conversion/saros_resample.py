import shutil
from argparse import ArgumentParser, Namespace
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from salt.data.conversion.utils import Label, write_label_files

SPLITS = {"fold-1": "val", "test": "test"}
DEFAULT_SPLIT = "train"

LABELS = [
    Label("background", ("background",)),
    Label("subcutaneous tissue", ("body", "subcutaneous_tissue")),
    Label("muscles", ("body", "muscles")),
    Label("abdominal cavity", ("body", "abdominal_cavity")),
    Label("thoracic cavity", ("body", "thoracic_cavity")),
    Label("bones", ("body", "bones")),
    Label("parotid glands", None),
    # Label("parotid glands", ("body", "parotid_glands")),
    Label(
        "pericardium",
        ("body", "thoracic_cavity", "mediastinum", "pericardium"),
    ),
    Label("breast_implant", None),
    # Label("breast implant", ("body", "breast_implant")),
    Label("mediastinum", ("body", "thoracic_cavity", "mediastinum")),
    Label("brain", ("body", "brain")),
    Label("spinal cord", ("body", "spinal_cord")),
    Label("thyroid glands", None),
    # Label("thyroid glands", ("body", "thyroid_glands")),
    Label("submandibular glands", None),
    # Label("submandibular glands", ("body", "submandibular_glands")),
]


def resample_to_thickness(image: sitk.Image, thickness: float = 5) -> sitk.Image:
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    output_spacing = (input_spacing[0], input_spacing[1], thickness)
    output_direction = image.GetDirection()
    output_origin = image.GetOrigin()
    output_size = [
        int(np.round(input_size[i] * input_spacing[i] / output_spacing[i]))
        for i in range(3)
    ]
    return sitk.Resample(
        image,
        output_size,
        sitk.Transform(),
        sitk.sitkBSpline3,
        output_origin,
        output_spacing,
        output_direction,
    )


def handle_case(
    info, dataset_dir: Path, output_dir: Path, use_original_image: bool
) -> None:
    split, case_id = info
    ds_split = SPLITS.get(split, DEFAULT_SPLIT)

    image_path = output_dir / ds_split / "images" / (case_id + ".nii.gz")
    seg_path = output_dir / ds_split / "labels" / (case_id + ".nii.gz")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists() or not seg_path.exists():
        if use_original_image:
            # In case of using original image, we need to resample the image and the segmentation.
            # Segmentation is sparse, so using nearest neighbor interpolation might drop segmented slices.
            # E.g. 1mm -> 1.5mm might drop slices
            img = sitk.ReadImage(str(dataset_dir / case_id / "image_original.nii.gz"))
            orig_spacing = img.GetSpacing()
            img = resample_to_thickness(img, 1.5)
            img_data = sitk.GetArrayViewFromImage(img)
            seg = sitk.ReadImage(str(dataset_dir / case_id / "body-regions.nii.gz"))
            seg_data = sitk.GetArrayViewFromImage(seg)
            indices = np.where((seg_data != 255).any(axis=(1, 2)))[0]

            print(
                case_id,
                img_data.shape,
                indices,
                orig_spacing,
                img.GetSpacing(),
                seg.GetSpacing(),
            )
            result = np.full(img_data.shape, 255, dtype=np.uint8)
            for index in indices:
                point = seg.TransformIndexToPhysicalPoint((0, 0, int(index)))
                coord = img.TransformPhysicalPointToIndex(point)
                assert coord[-1] < img_data.shape[0]
                result[coord[-1], :, :] = seg_data[index, :, :]
            result_img = sitk.GetImageFromArray(result)
            result_img.CopyInformation(img)
            sitk.WriteImage(img, str(image_path))
            sitk.WriteImage(result_img, str(seg_path))
        else:
            shutil.copy2(dataset_dir / case_id / "image.nii.gz", image_path)
            shutil.copy2(dataset_dir / case_id / "body-regions.nii.gz", seg_path)
    else:
        print(f"{case_id} was already resampled, skipping...")


def main(args: Namespace) -> None:
    info_df = pd.read_csv(args.dataset_dir / "info.csv")

    with Pool(processes=8) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(
                    handle_case,
                    dataset_dir=args.dataset_dir,
                    output_dir=args.output_dir,
                    use_original_image=args.use_original_image,
                ),
                [(row.split, row.id) for row in info_df.itertuples()],
            ),
            total=len(info_df),
        ):
            pass

    write_label_files(args.output_dir, LABELS)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-original-image", default=False, action="store_true")

    args = parser.parse_args()

    main(args)
