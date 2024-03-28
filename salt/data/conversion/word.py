import argparse
import multiprocessing as mp
import pathlib
from functools import partial

import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm

from salt.data.conversion.utils import (
    Label,
    enforce_orientation,
    plot_volume,
    write_label_files,
)

LABELS = [
    Label(),  # background
    Label("liver", ("body", "abdominal_cavity", "liver")),
    Label("spleen", ("body", "abdominal_cavity", "spleen")),
    Label("left kidney", ("body", "abdominal_cavity", "kidneys", "kidney_left")),
    Label("right kidney", ("body", "abdominal_cavity", "kidneys", "kidney_right")),
    Label("stomach", ("body", "abdominal_cavity", "stomach")),
    Label("gallbladder", ("body", "abdominal_cavity", "gallbladder")),
    Label("esophagus", None),
    Label("pancreas", ("body", "abdominal_cavity", "pancreas")),
    Label("duodenum", ("body", "abdominal_cavity", "duodenum")),
    Label("colon", ("body", "abdominal_cavity", "colon")),
    Label("intestine", ("body", "abdominal_cavity", "small_bowel")),
    Label(
        "adrenal glands",
        ("body", "abdominal_cavity", "adrenal_glands"),
    ),
    Label("rectum", None),
    Label("bladder", ("body", "abdominal_cavity", "urinary_bladder")),
    Label("left head of femur", None),
    Label("right head of femur", None),
]

NUM_LABELS = len(LABELS)
LABELS_STR = [label.original for label in LABELS]


def handle_example(image_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    # Derive image ID and image path
    image_id = image_path.name.replace(".nii.gz", "")

    if image_path.parent.name == "imagesTr":
        phase_folder = "Tr"
    elif image_path.parent.name == "imagesLiTSTs":
        phase_folder = "LiTSTs"
        image_id = image_id.replace("volume-", "liver_").replace(".nii", "_word_label")
    else:
        phase_folder = "Val"

    seg_path = image_path.parent.parent / f"labels{phase_folder}" / f"{image_id}.nii.gz"

    # Load data
    image = sitk.ReadImage(str(image_path), sitk.sitkInt16)
    seg = sitk.ReadImage(str(seg_path), sitk.sitkUInt8)

    # Normalize orientation
    image = enforce_orientation(image)
    seg = enforce_orientation(seg)

    seg_data = sitk.GetArrayFromImage(seg)
    seg_data[seg_data == LABELS_STR.index("rectum")] = LABELS_STR.index("colon")

    new_seg = sitk.GetImageFromArray(seg_data)
    new_seg.CopyInformation(seg)
    # Save to output directory
    output_image_path = output_dir / "test" / "images" / f"{image_id}.nii.gz"
    sitk.WriteImage(image, str(output_image_path))
    output_seg_path = output_dir / "test" / "labels" / f"{image_id}.nii.gz"
    sitk.WriteImage(new_seg, str(output_seg_path))
    plot_volume(image, new_seg).savefig(
        output_dir / "test" / "previews" / f"{image_id}.png"
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    # Create output directory structure
    (args.output_dir / "test" / "images").mkdir(exist_ok=True, parents=True)
    (args.output_dir / "test" / "labels").mkdir(exist_ok=True, parents=True)
    (args.output_dir / "test" / "previews").mkdir(exist_ok=True, parents=True)

    # Process all examples
    train_paths = sorted((args.dataset_dir / "imagesTr").rglob("*.nii.gz"))
    val_paths = sorted((args.dataset_dir / "imagesVal").rglob("*.nii.gz"))
    test_paths = sorted((args.dataset_dir / "imagesLiTSTs").rglob("*.nii"))

    seg_paths = test_paths + train_paths + val_paths
    with mp.Pool(processes=20) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(handle_example, output_dir=args.output_dir), seg_paths
            ),
            total=len(seg_paths),
            desc="Processing WORD samples",
        ):
            pass

    # Generate label files
    write_label_files(args.output_dir, LABELS)
