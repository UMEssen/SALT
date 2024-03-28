import multiprocessing
import pickle
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from standalone.utils import LabelQuery
from surface_distance.metrics import (
    compute_average_surface_distance,
    compute_robust_hausdorff,
    compute_surface_dice_at_tolerance,
    compute_surface_distances,
)
from tqdm import tqdm

JOINED_REGIONS = [
    "aorta",
    "vci",
    "pulmonary_artery",
    "body>abdominal_cavity>kidneys",
    "body>thoracic_cavity>lungs",
    "body>thoracic_cavity>lungs>lung_left",
    "body>thoracic_cavity>lungs>lung_right",
    "body>abdominal_cavity>adrenal_glands",
]

DATASETS = [
    "luna16lobes",  # Take
    "saros",  # Take
    "word",  # Take
    "ctorg",  # Take
    # "amos22",
    # "saros+totalseg",
    "flare22",
    "lctsc",
    # "totalseg",
]


metrics = [
    "precision",
    "recall",
    "dice",
    "surface_distance_3mm",
]

SAROS_MAPPING = {
    0: [0],  # background
    1: [1],  # subcutaneous
    2: [2],  # muscles
    3: [3],  # abdominal cavity
    4: [
        4,
        7,
        9,
    ],  # thoracic cavity has mediastinum and pericardium
    5: [5],  # bones
    6: [6],  # parotid glands
    7: [7],  # pericardium
    8: [8],  # breast implant
    9: [7, 9],  # mediastinum has pericardium
    10: [10],  # brain
    11: [11],  # spinal cord
    12: [12],  # thyroid glands
    13: [13],  # submandibular glands
}


def compute_metrics(gt: np.ndarray, pred: np.ndarray, spacing: Tuple[int]):
    # There is no GT and no prediction
    if not gt.max() and not pred.max():
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
    # There is GT but no prediction
    elif gt.max() and not pred.max():
        return {
            "tp": 0,
            "fp": 0,
            "fn": gt.sum(),
            **{m: 0 for m in metrics},
        }
    # There is prediction but no GT
    elif pred.max() and not gt.max():
        return {
            "tp": 0,
            "fp": pred.sum(),
            "fn": 0,
            **{m: 0 for m in metrics},
        }
    else:
        tp = (gt & pred).sum()
        fp = (~gt & pred).sum()
        fn = (gt & ~pred).sum()
        sd = compute_surface_distances(gt, pred, spacing)
        avg_gt_to_pred, avg_pred_to_gt = compute_average_surface_distance(sd)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "dice": tp / (tp + 0.5 * fp + 0.5 * fn),
            "hausdorff_95": compute_robust_hausdorff(sd, 95.0),
            "avg_surface_distance": (avg_gt_to_pred + avg_pred_to_gt) / 2,
            "surface_distance_1mm": compute_surface_dice_at_tolerance(sd, 1.0),
            "surface_distance_2mm": compute_surface_dice_at_tolerance(sd, 2.0),
            "surface_distance_3mm": compute_surface_dice_at_tolerance(sd, 3.0),
        }


def _worker(
    case_path: Path,
    data_dir: Path,
    label_names: List[str],
    adjacency_matrix: np.ndarray,
    joined_regions: Dict[str, List[int]],
):
    dataset_name = case_path.parent.parent.name
    ground_truth_path = (
        data_dir
        / dataset_name
        / "test"
        / "labels"
        / case_path.name.replace("_pred", "")
    )
    sitk_label = sitk.ReadImage(str(ground_truth_path))
    sitk_pred = sitk.ReadImage(str(case_path))

    with (data_dir / dataset_name / "tree-labels.txt").open("r") as of:
        tree_labels = [tuple(lab.split(",")) for lab in of.read().split("\n")]

    label_data = sitk.GetArrayViewFromImage(sitk_label)
    pred_data = sitk.GetArrayFromImage(sitk_pred)
    if label_data.max() == 255:
        # print(f"{ground_truth_path.name} contains ignore labels.")
        label_data = label_data.copy()
        pred_data = pred_data.copy()
        mask_values = label_data == 255
        label_data[mask_values] = 0
        pred_data[mask_values] = 0
    lq = LabelQuery(pred_data, adjacency_matrix, pruned=True)

    results = []
    base_dict = {
        "id": case_path.name.replace("_pred.nii.gz", ""),
        "dataset": dataset_name,
    }
    metric_based_dict = {}
    for idx, label in enumerate(tree_labels):
        # We don't eval 0
        if label[0] == "-" or idx == 0:
            continue
        if label in label_names:
            label_id = label_names.index(label)
        else:
            print(f"Label {label} not found in label names")
            continue
        if dataset_name == "saros":
            gt = np.isin(label_data, SAROS_MAPPING[idx])
        else:
            gt = label_data == idx
        pred = lq.get_mask(label_id)
        for metric, value in compute_metrics(gt, pred, sitk_label.GetSpacing()).items():
            if metric not in metric_based_dict:
                metric_based_dict[metric] = {
                    ">".join(label): value,
                }
            else:
                metric_based_dict[metric][">".join(label)] = value

    for region in JOINED_REGIONS:
        gt_region_name = region if region[-1] != "s" else region.split(">")[-1][:-1]
        if "lung_left" in region:
            gt_ids = [
                i
                for i, lab in enumerate(tree_labels)
                if "lung" in lab[-1] and "left" in lab[-1]
            ]
        elif "lung_right" in region:
            gt_ids = [
                i
                for i, lab in enumerate(tree_labels)
                if "lung" in lab[-1] and "right" in lab[-1]
            ]
        else:
            gt_ids = [
                i for i, lab in enumerate(tree_labels) if gt_region_name in lab[-1]
            ]
        if len(gt_ids) < 2:
            continue
        pred_ids = joined_regions[region]
        # print(region, gt_ids, pred_ids)
        gt = np.isin(label_data, gt_ids)
        pred = lq.get_mask(pred_ids)
        for metric, value in compute_metrics(gt, pred, sitk_label.GetSpacing()).items():
            metric_based_dict[metric][region] = value
    for metric, values in metric_based_dict.items():
        # if metric == "dice":
        #     print(base_dict)
        #     for i, val in values.items():
        #         print(i, val)
        results.append({**base_dict, "metric": metric, **values})
    return results


def main(args: Namespace) -> None:
    args.output_dir.mkdir(exist_ok=True, parents=True)
    with args.config_file.open(
        "rb",
    ) as ifile:
        config = pickle.load(ifile)

    # for i, label in enumerate(config["label_names"]):
    #     print(i, ">".join(label))

    cases = sorted(
        [
            f
            for f in args.predictions_dir.rglob("*.nii.gz")
            if f.parent.parent.name in DATASETS
        ]
    )
    # cases = sorted(args.predictions_dir.rglob("*.nii.gz"))[:1]

    joined_regions = {}
    for region in JOINED_REGIONS:
        joined_regions[region] = [
            i
            for i, lab in enumerate(config["label_names"])
            if region.split(">")[-1] in lab[-1]
        ]

    results = []
    with multiprocessing.Pool(processes=20) as pool:
        for res in tqdm(
            pool.imap_unordered(
                partial(
                    _worker,
                    data_dir=args.data_dir,
                    label_names=config["label_names"],
                    adjacency_matrix=config["adjacency_matrix"],
                    joined_regions=joined_regions,
                ),
                cases,
            ),
            total=len(cases),
        ):
            results += res
    pd.DataFrame(results).to_csv(args.output_dir / "classic-scores.csv", index=False)


if __name__ == "__main__":
    cli = ArgumentParser()
    cli.add_argument("--data-dir", type=Path, required=True)
    cli.add_argument("--predictions-dir", type=Path, required=True)
    cli.add_argument("--config-file", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()

    main(args)
