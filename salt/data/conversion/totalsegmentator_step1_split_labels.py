import shutil
from argparse import ArgumentParser, Namespace
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def remove_face_class(segmentation: np.ndarray, label_names: List[str]) -> None:
    face_label_idx = label_names.index("face")
    face_mask = segmentation == face_label_idx
    replacement_mask = segmentation == len(label_names) - 1
    segmentation[face_mask] = 0
    segmentation[replacement_mask] = face_label_idx


def handle_case(
    segmentation_path: Path,
    bca_dir: Path,
    label_names: List[str],
    output_dir: Path,
):
    split_name = segmentation_path.parent.parent.name
    (args.output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)
    seg_ts = sitk.ReadImage(str(segmentation_path))
    seg_br = sitk.ReadImage(
        str(
            # Go back to the folder root
            bca_dir
            / split_name
            / "labels"
            / segmentation_path.name
        )
    )
    assert seg_ts.GetSize() == seg_br.GetSize(), "Segmentations have different sizes"

    seg_ts_data = sitk.GetArrayFromImage(seg_ts)

    remove_face_class(seg_ts_data, label_names)
    # Index is -1 because the urinary bladder gets to the face spot
    last_label_index = len(label_names) - 1

    seg_br_data = sitk.GetArrayViewFromImage(seg_br)

    abdominal_cavity_mask = np.equal(seg_br_data, 3)
    pericardium_mask = np.equal(seg_br_data, 7)
    # mediastinum_mask = np.equal(seg_br_data, 9)

    # Split aorta into abdominal and thoracic part
    aorta_idx = label_names.index("aorta")
    aorta_mask = np.equal(seg_ts_data, aorta_idx)
    aorta_abdominalis_mask = np.logical_and(aorta_mask, abdominal_cavity_mask)
    aorta_thoracica_pass_pericardium = np.logical_and(aorta_mask, pericardium_mask)
    seg_ts_data[aorta_abdominalis_mask] = last_label_index
    seg_ts_data[aorta_thoracica_pass_pericardium] = last_label_index + 1

    # Split inferior vena cava into abdominal and thoracic part
    ivc_idx = label_names.index("inferior_vena_cava")
    ivc_mask = np.equal(seg_ts_data, ivc_idx)
    ivc_abdominalis_mask = np.logical_and(ivc_mask, abdominal_cavity_mask)
    seg_ts_data[ivc_abdominalis_mask] = last_label_index + 2

    # Filter iliac artery/vena to be only present in abdominal cavity
    for label_name in {"iliac_artery", "iliac_vena"}:
        for side in {"left", "right"}:
            label_idx = label_names.index(f"{label_name}_{side}")
            seg_ts_data[
                np.logical_and(np.equal(seg_ts_data, label_idx), ~abdominal_cavity_mask)
            ] = 0

    # Split pulmunary artery into mediastinum and pericardium part
    pulmonary_artery_idx = label_names.index("pulmonary_artery")
    pulmonary_artery_mask = np.equal(seg_ts_data, pulmonary_artery_idx)
    pulmonary_artery_pass_pericardium_mask = np.logical_and(
        pulmonary_artery_mask, pericardium_mask
    )
    seg_ts_data[pulmonary_artery_pass_pericardium_mask] = last_label_index + 3

    # Save updates segmentation
    new_seg = sitk.GetImageFromArray(seg_ts_data)
    new_seg.CopyInformation(seg_ts)
    sitk.WriteImage(
        new_seg, str(output_dir / split_name / "labels" / segmentation_path.name)
    )
    (args.output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        segmentation_path.parent.parent / "images" / segmentation_path.name,
        output_dir / split_name / "images" / segmentation_path.name,
    )


def main(args: Namespace) -> None:
    with (args.input_dir / "labels.txt").open() as ifile:
        label_names = ifile.read().splitlines()

    segmentations = sorted(
        [p for p in args.input_dir.rglob("*.nii.gz") if p.parent.name != "images"]
    )
    with Pool() as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(
                    handle_case,
                    bca_dir=args.bca_dir,
                    label_names=label_names,
                    output_dir=args.output_dir,
                ),
                segmentations,
            ),
            total=len(segmentations),
        ):
            pass

    # Substitute face with urinary bladder
    label_names[label_names.index("face")] = label_names[-1]

    # Remove last label
    label_names = label_names[:-1]

    # Update labels
    label_names[label_names.index("aorta")] = "aorta_thoracica_pass_mediastinum"
    label_names.append("aorta_abdominalis")
    label_names.append("aorta_thoracica_pass_pericardium")
    label_names[label_names.index("inferior_vena_cava")] = "vci_pass_thoracica"
    label_names.append("vci_pass_abdominalis")
    label_names[
        label_names.index("pulmonary_artery")
    ] = "pulmonary_artery_pass_mediastinum"
    label_names.append("pulmonary_artery_pass_pericardium")

    with (args.output_dir / "labels.txt").open("w") as ofile:
        ofile.write("\n".join(label_names))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bca-dir", type=Path, required=True)
    args = parser.parse_args()

    main(args)
