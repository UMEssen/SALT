import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from salt.core.label_tree import LabelTree

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    data_dir: Path
    labels: List[Optional[Tuple[str, ...]]]
    label_lut: np.ndarray
    class_mask: np.ndarray
    splits: Dict[str, List[Dict[str, Path]]]

    @classmethod
    def from_tree_labels(cls, tree_labels_path: Path) -> "DatasetInfo":
        '''
        Load labels from a text file describing labels in a dataset. Each line
        describes one integer label, starting with 0.  Unused labels can be
        marked with a single "-" or an empty line.  Other lines should contain
        comma-separated hierarchical labels, e.g.
        "body,thoracic_cavity,lungs,lung_left" would describe the meaning of a
        single integer label in this dataset.
        '''
        labels = [
            tuple(label.split(",")) if label and label != "-" else None
            for label in tree_labels_path.read_text().splitlines()
        ]
        result = cls(
            data_dir=tree_labels_path.parent,
            labels=labels,
            splits={},
            # placeholder values that could be used if this dataset is the only
            # one used; will be replaced later with the results of
            # LabelTree.create_label_lut():
            label_lut=np.arange(len(labels), dtype=np.uint8),
            class_mask=np.ones((len(labels),), dtype=np.uint8),
        )

        for split_path in result.data_dir.iterdir():
            if not split_path.is_dir():
                continue

            image_files = sorted(list((split_path / "images").glob("*.nii.gz")))
            label_files = [split_path / "labels" / file.name for file in image_files]

            result.splits[split_path.name] = [
                {"image": image_file, "label": label_file}
                for image_file, label_file in zip(image_files, label_files)
            ]

        return result


@dataclass
class DataConfig:
    dataset_infos: List[DatasetInfo]
    num_classes: int
    labels: List[Tuple[str, ...]]
    leaf_labels: List[Tuple[str, ...]]
    adjacency_matrix: np.ndarray
    sink_mask: np.ndarray


def find_datasets(dataset_dir: Path) -> DataConfig:
    label_paths = list(dataset_dir.rglob("tree-labels.txt"))
    result = []

    for label_path in label_paths:
        logger.info(f"Found dataset: {label_path.parent.name}")
        result.append(DatasetInfo.from_tree_labels(label_path))

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
        leaf_labels=builder.leaf_names,
        adjacency_matrix=builder.adjacency_matrix,
        sink_mask=sink_mask,
    )
