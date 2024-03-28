import argparse
import multiprocessing as mp
import pathlib
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon
from tqdm import tqdm

from salt.data.conversion.utils import (
    Label,
    enforce_orientation,
    plot_volume,
    write_label_files,
)

LABELS = [
    Label(),
    Label("spinal_cord", ("body", "spinal_cord")),
    Label("lung_right", ("body", "thoracic_cavity", "lungs", "lung_right")),
    Label("lung_left", ("body", "thoracic_cavity", "lungs", "lung_left")),
    Label("heart", ("body", "thoracic_cavity", "mediastinum", "pericardium")),
    Label("esophagus", None),
]


def handle_example(rtdcm_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    # Derive image ID and image path
    image_id = rtdcm_path.parent.parent.parent.name
    image_dir = [
        x
        for x in rtdcm_path.parent.parent.iterdir()
        if x.name != rtdcm_path.parent.name
    ][0]

    # Load data
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(image_dir)))
    image = reader.Execute()
    seg = _convert_seg_from_rtdcm(image, rtdcm_path)

    # Normalize orientation
    image = enforce_orientation(image)
    seg = enforce_orientation(seg)

    # Resample to uniform slice thickness of 5mm
    # image, seg = resample_to_thickness(image, seg, 5.0)

    # Save to output directory
    split_name = "test"  # "train" if "Train" in image_id else "test"
    output_image_path = output_dir / split_name / "images" / f"{image_id}.nii.gz"
    sitk.WriteImage(image, str(output_image_path))
    output_seg_path = output_dir / split_name / "labels" / f"{image_id}.nii.gz"
    sitk.WriteImage(seg, str(output_seg_path))
    plot_volume(image, seg).savefig(
        output_dir / split_name / "previews" / f"{image_id}.png"
    )
    plt.close()


def _convert_seg_from_rtdcm(image: sitk.Image, rtdcm_path: pathlib.Path) -> sitk.Image:
    dcm = pydicom.dcmread(rtdcm_path)
    seg_data = np.zeros(image.GetSize()[::-1], dtype=np.uint8)

    possible_names = [None, "SpinalCord", "Lung_R", "Lung_L", "Heart", "Esophagus"]
    roi_number_to_class_idx = {
        x.ROINumber: possible_names.index(x.ROIName)
        for x in dcm.StructureSetROISequence
    }

    for roi in dcm.ROIContourSequence:
        class_idx = roi_number_to_class_idx[roi.ReferencedROINumber]
        for contour in roi.ContourSequence:
            assert contour.ContourGeometricType == "CLOSED_PLANAR"
            data = np.asarray(contour.ContourData)
            data = data.reshape((-1, 3))
            data = np.asarray(
                [image.TransformPhysicalPointToContinuousIndex(x)[::-1] for x in data]
            )
            slice_idx = int(np.round(data[0, 0]))
            rr, cc = polygon(data[:, 1], data[:, 2], seg_data.shape[1:])
            seg_data[slice_idx, rr, cc] = class_idx

    empty_slices = np.equal(seg_data, 0).all(axis=(1, 2))
    seg_data[empty_slices, ...] = 255

    seg = sitk.GetImageFromArray(seg_data)
    seg.CopyInformation(image)

    return seg


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

    # # Process all examples
    rtdcm_paths = list(args.dataset_dir.rglob("1-1.dcm"))
    with mp.Pool() as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(handle_example, output_dir=args.output_dir), rtdcm_paths
            ),
            total=len(rtdcm_paths),
            desc="Processing LCTSC samples",
        ):
            pass

    # Generate label files
    write_label_files(args.output_dir, LABELS)
