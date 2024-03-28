import traceback
from argparse import ArgumentParser, Namespace
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from shutil import copy2
from typing import List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# This is needed because some images cannot be read by SimpleITK
def change_info(path):
    img = nib.load(path)
    qform = img.get_qform()
    img.set_qform(qform)
    sform = img.get_sform()
    img.set_sform(sform)
    new_path = path.parent / path.name.replace(".nii.gz", "_fixed.nii.gz")
    nib.save(img, new_path)
    return new_path


def handle_case(case_info, label_names: List[str], input_dir: Path, output_dir: Path):
    split, image_id = case_info
    new_path_images = output_dir / split / "images"
    new_path_segs = output_dir / split / "labels"

    # This image is broken in V1 https://github.com/wasserth/TotalSegmentator/issues/24
    if image_id == "s0864":
        return
    if (new_path_images / f"{image_id}.nii.gz").exists() and (
        new_path_segs / f"{image_id}.nii.gz"
    ).exists():
        return

    segmentation: Optional[np.ndarray] = None
    for label_idx, label_name in enumerate(label_names):
        # Ignore background class
        if label_idx == 0:
            continue

        store_p = input_dir / image_id / "segmentations" / f"{label_name}.nii.gz"
        try:
            binary_seg = sitk.ReadImage(str(store_p))
        except RuntimeError:
            print("Converting", image_id, label_name)
            store_p = change_info(store_p)
            binary_seg = sitk.ReadImage(str(store_p))
        if segmentation is None:
            # First label has already label idx 1
            segmentation = sitk.GetArrayFromImage(binary_seg).astype(np.uint8)
        else:
            binary_seg_data = sitk.GetArrayViewFromImage(binary_seg)
            segmentation[binary_seg_data != 0] = label_idx

    seg_image = sitk.GetImageFromArray(segmentation)
    seg_image.CopyInformation(binary_seg)
    new_path_segs.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(seg_image, str(new_path_segs / f"{image_id}.nii.gz"))

    ct_file = input_dir / image_id / "ct.nii.gz"
    try:
        sitk.ReadImage(ct_file)
    except RuntimeError:
        traceback.print_exc()
        print(split, "Converting", image_id)
        ct_file = change_info(ct_file)
    # print(ct_file)
    new_path_images.mkdir(parents=True, exist_ok=True)
    copy2(ct_file, new_path_images / f"{image_id}.nii.gz")


def main(args: Namespace) -> None:
    with Path("preprocessing_scripts/labels_totalseg.txt").open() as ifile:
        label_names = ifile.read().splitlines()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = pd.read_csv(args.input_dir / "meta.csv", sep=";")

    with Pool(processes=8) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(
                    handle_case,
                    label_names=label_names,
                    input_dir=args.input_dir,
                    output_dir=args.output_dir,
                ),
                [
                    ("test" if args.all_test_split else row.split, row.image_id)
                    for row in info.itertuples()
                ],
            ),
            total=len(info),
        ):
            pass

    with (args.output_dir / "labels.txt").open("w") as ofile:
        ofile.write("\n".join(label_names))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--all-test-split", default=False, action="store_true")
    args = parser.parse_args()

    main(args)
