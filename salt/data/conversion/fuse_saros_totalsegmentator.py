from argparse import ArgumentParser, Namespace
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from shutil import copy2
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def _read_image(working_dir: Path) -> Tuple[sitk.Image, np.ndarray]:
    img = sitk.ReadImage(str(working_dir))  # type: ignore
    arr = sitk.GetArrayFromImage(img)  # type: ignore
    return img, arr


def build_map(arr):
    unique, counts = np.unique(arr, return_counts=True)
    return {int(u): int(c) for u, c in zip(list(unique), list(counts))}


def build_labels(filename: Path):
    with filename.open("r") as f:
        label_dict = [(i, line.strip()) for i, line in enumerate(f.readlines())]
    return label_dict


OTHER_LABELS = {
    "abdominal_cavity",
    "pericardium",
    "mediastinum",
    "thoracic_cavity",
    "muscles",
    "bones",
}


SAROS_IGNORE_INDICES = {
    6,  # Remove glands
    10,  # Remove, use brain from Totalsegmentator
    8,  # Remove breast implant
    12,  # Remove glands
    13,  # Remove glands
}


def handle_case(
    ct_path: Path,
    saros_root: Path,
    totalseg_root: Path,
    output_dir: Path,
    total_labels,
    existing_labels: List[str],
):
    phase = ct_path.parent.parent.name
    saros_img, saros_seg = _read_image(saros_root / phase / "labels" / ct_path.name)
    total_img, total_seg = _read_image(totalseg_root / phase / "labels" / ct_path.name)

    assert np.isclose(saros_img.GetSize(), total_img.GetSize()).all(), (  # type: ignore
        f"SAROS image and total image do not have the same size: "  # type: ignore
        f"{saros_img.GetSize()} vs. {total_img.GetSize()}"  # type: ignore
    )
    assert np.isclose(saros_img.GetSpacing(), total_img.GetSpacing()).all(), (  # type: ignore
        f"SAROS image and total image do not have the same spacing: "  # type: ignore
        f"{saros_img.GetSpacing()} vs. {total_img.GetSpacing()}"  # type: ignore
    )

    for index in SAROS_IGNORE_INDICES:
        saros_seg[saros_seg == index] = 255

    # brain_label = [i for i, name in total_labels if name == "body,brain"]
    # assert len(brain_label) == 1
    # saros_seg[saros_seg == 10] = existing_labels + brain_label[0]

    # old_saros_counts = build_map(saros_seg)
    # total_counts = build_map(total_seg)
    for i, name in total_labels[1:]:
        if name == "-":
            # print(i, "will be ignored")
            continue
        # print(i, existing_labels + i, name)
        saros_seg[total_seg == i] = existing_labels + i

    # for i, count in build_map(saros_seg).items():
    #     if i == 255:
    #         continue
    #     else:
    #         name = [name for j, name in enumerate(new_labels) if j == i]
    #         assert len(name) == 1
    #         name = name[0]
    #     if i < existing_labels + 1:
    #         if name.split(",")[-1] not in OTHER_LABELS:
    #             if old_saros_counts[i] != count:
    #                 print(
    #                     i,
    #                     name,
    #                     count,
    #                     old_saros_counts[i],
    #                 )
    #                 assert count < old_saros_counts[i]
    #     else:
    #         if count != total_counts[i - existing_labels]:
    #             print(
    #                 i,
    #                 name,
    #                 count,
    #                 total_counts[i - existing_labels],
    #             )

    new_dataset = output_dir / phase
    (new_dataset / "images").mkdir(parents=True, exist_ok=True)
    (new_dataset / "labels").mkdir(parents=True, exist_ok=True)
    copy2(ct_path, new_dataset / "images" / ct_path.name)

    new_seg = sitk.GetImageFromArray(saros_seg)
    new_seg.CopyInformation(saros_img)  # type: ignore
    sitk.WriteImage(new_seg, new_dataset / "labels" / ct_path.name, useCompression=True)


def main(args: Namespace) -> None:
    total_labels = build_labels(args.totalseg_input_dir / "tree-labels.txt")
    saros_labels = build_labels(args.saros_input_dir / "tree-labels.txt")
    existing_labels = len(saros_labels) - 1

    new_labels = ["-" if a in SAROS_IGNORE_INDICES else b for a, b in saros_labels] + [
        b for _, b in total_labels[1:]
    ]

    cts = [
        ct
        for phase in args.saros_input_dir.glob("*")
        if phase.is_dir()
        for ct in phase.glob("images/*.nii.gz")
    ]

    with Pool() as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(
                    handle_case,
                    saros_root=args.saros_input_dir,
                    totalseg_root=args.totalseg_input_dir,
                    output_dir=args.output_dir,
                    total_labels=total_labels,
                    existing_labels=existing_labels,
                ),
                cts,
            ),
            total=len(cts),
            desc="Fusing SAROS with TotalSegmentator",
        ):
            pass

    with (args.output_dir / "tree-labels.txt").open("w") as f:
        for label in new_labels:
            if label.split(",")[-1] in OTHER_LABELS:
                f.write(label + ",other" + "\n")
            else:
                f.write(label + "\n")

    with (args.output_dir / "labels.txt").open("w") as f:
        for label in new_labels:
            f.write(label.split(",")[-1] + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saros-input-dir", type=Path, required=True)
    parser.add_argument("--totalseg-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    main(args)
