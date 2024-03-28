import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from salt.core.label_tree import LabelTree

logger = logging.getLogger(__name__)


@dataclass(init=False)
class DatasetInfo:
    data_dir: Path
    labels: List[Optional[Tuple[str, ...]]]
    label_lut: np.ndarray
    class_mask: np.ndarray
    splits: Dict[str, List[Dict[str, Path]]]


@dataclass
class DataConfig:
    dataset_infos: List[DatasetInfo]
    num_classes: int
    labels: List[Tuple[str, ...]]
    leave_labels: List[Tuple[str, ...]]
    adjacency_matrix: np.ndarray
    sink_mask: np.ndarray


def find_datasets(dataset_dir: Path) -> DataConfig:
    label_paths = list(dataset_dir.rglob("tree-labels.txt"))
    result = []

    for label_path in label_paths:
        logger.info(f"Found dataset: {label_path.parent.name}")
        info = DatasetInfo()
        result.append(info)
        info.data_dir = label_path.parent
        info.labels = [
            tuple(label.split(",")) if label and label != "-" else None
            for label in label_path.read_text().splitlines()
        ]

        info.splits = {}
        for split_path in info.data_dir.iterdir():
            if not split_path.is_dir():
                continue

            image_files = sorted(list((split_path / "images").glob("*.nii.gz")))
            label_files = [split_path / "labels" / file.name for file in image_files]

            info.splits[split_path.name] = [
                {"image": image_file, "label": label_file}
                for image_file, label_file in zip(image_files, label_files)
            ]

    builder = LabelTree()
    for info in result:
        for label in info.labels:
            if label:
                builder.add(*label)

    sinks: Set[Tuple[str, ...]] = set()
    additional_label_path = dataset_dir / "tree-additional-labels.txt"
    if additional_label_path.exists():
        for label_line in additional_label_path.read_text().splitlines():
            label = tuple(label_line.split(","))
            sinks.add(label)
            builder.add(*label)

    builder.optimize()
    for info in result:
        info.label_lut, info.class_mask = builder.create_label_lut(info.labels)

    sink_mask = np.zeros((builder.num_classes,), dtype=np.uint8)
    for sink_label in sinks:
        idx = builder.labels.index(sink_label)
        sink_mask[idx] = 1

    return DataConfig(
        dataset_infos=result,
        num_classes=builder.num_classes,
        labels=builder.labels,
        leave_labels=builder.leave_names,
        adjacency_matrix=builder.adjacency_matrix,
        sink_mask=sink_mask,
    )
