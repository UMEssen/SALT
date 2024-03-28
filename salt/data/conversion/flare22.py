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
    split_long_organs,
    write_label_files,
)

LABELS = [
    Label(),
    Label("liver", ("body", "abdominal_cavity", "liver")),
    Label("right_kidney", ("body", "abdominal_cavity", "kidneys", "kidney_right")),
    Label("spleen", ("body", "abdominal_cavity", "spleen")),
    Label("pancreas", ("body", "abdominal_cavity", "pancreas")),
    # What will remain of the aorta after splitting will be aorta_thoracica_pass_mediastinum
    Label(
        "aorta",
        (
            "body",
            "thoracic_cavity",
            "mediastinum",
            "aorta_thoracica_pass_mediastinum",
        ),
    ),
    # What will remain of the vci after splitting will be vci_pass_thoracica
    Label(
        "inferior_vena_cava",
        (
            "body",
            "thoracic_cavity",
            "mediastinum",
            "vci_pass_thoracica",
        ),
    ),
    Label(
        "right_adrenal_gland",
        ("body", "abdominal_cavity", "adrenal_glands", "adrenal_gland_right"),
    ),
    Label(
        "left_adrenal_gland",
        ("body", "abdominal_cavity", "adrenal_glands", "adrenal_gland_left"),
    ),
    Label("gallbladder", ("body", "abdominal_cavity", "gallbladder")),
    Label("esophagus", None),
    Label("stomach", ("body", "abdominal_cavity", "stomach")),
    Label("duodenum", ("body", "abdominal_cavity", "duodenum")),
    Label("left_kidney", ("body", "abdominal_cavity", "kidneys", "kidney_left")),
]

NUM_LABELS = len(LABELS)

LABELS += [
    Label("aorta abdominalis", ("body", "abdominal_cavity", "aorta_abdominalis")),
    Label(
        "aorta thoracica",
        (
            "body",
            "thoracic_cavity",
            "mediastinum",
            "pericardium",
            "aorta_thoracica_pass_pericardium",
        ),
    ),
    Label("vci pass abdominalis", ("body", "abdominal_cavity", "vci_pass_abdominalis")),
]


def handle_example(image_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    # Derive image ID and image path
    image_id = image_path.name.replace("_0000.nii.gz", "")

    seg_path = image_path.parent.parent / "labels" / f"{image_id}.nii.gz"

    # Load data
    image = sitk.ReadImage(str(image_path), sitk.sitkInt16)
    seg = sitk.ReadImage(str(seg_path), sitk.sitkUInt8)

    # Normalize orientation
    image = enforce_orientation(image)
    seg = enforce_orientation(seg)

    br_image = sitk.ReadImage(
        str(
            image_path.parent.parent
            / "bca"
            / (image_id + "_0000")
            / "body-regions.nii.gz"
        )
    )
    br_image = enforce_orientation(br_image)

    real_labels = [ll.original for ll in LABELS[:NUM_LABELS]]
    seg = split_long_organs(seg_ts=seg, seg_br=br_image, label_names=real_labels)

    # Resample to uniform slice thickness of 5mm
    # image, seg = resample_to_thickness(image, seg, 5.0)

    # Save to output directory
    output_image_path = output_dir / "test" / "images" / f"{image_id}.nii.gz"
    sitk.WriteImage(image, str(output_image_path))
    output_seg_path = output_dir / "test" / "labels" / f"{image_id}.nii.gz"
    sitk.WriteImage(seg, str(output_seg_path))
    plot_volume(image, seg).savefig(
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
    seg_paths = list((args.dataset_dir / "images").rglob("*.nii.gz"))
    # seg_paths = seg_paths[:1]
    with mp.Pool() as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(handle_example, output_dir=args.output_dir), seg_paths
            ),
            total=len(seg_paths),
            desc="Processing FLARE22 samples",
        ):
            pass

    # Generate label files
    write_label_files(args.output_dir, LABELS)
