import logging
import pickle
from hashlib import sha256
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import itk
import numpy as np
import pytorch_lightning as pl
import torch
from monai.data import (
    DataLoader,
    Dataset,
    PersistentDataset,
    partition_dataset,
    select_cross_validation_folds,
)
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from salt.data.discovery import DataConfig
from salt.input_pipeline import (
    IntensityProperties,
    get_train_transforms,
    get_validation_transforms,
)

logger = logging.getLogger(__name__)


class MultiSourceDataset(Dataset):
    def __init__(self, data: List[Dataset], equal_sampling: bool = False) -> None:
        super().__init__(data)
        self._datasets = data
        self._equal_sampling = equal_sampling

        self._indices: List[Tuple[int, int]] = []
        if equal_sampling:
            max_samples_per_dataset = max(len(x) for x in data)
            for ds_idx, ds in enumerate(data):
                sample_count = len(ds)
                for sample_idx in range(max_samples_per_dataset):
                    self._indices.append(
                        (ds_idx, sample_idx if sample_idx < sample_count else -1)
                    )
        else:
            for ds_idx, ds in enumerate(data):
                for sample_idx in range(len(ds)):
                    self._indices.append((ds_idx, sample_idx))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Any:
        ds_idx, sample_idx = self._indices[index]
        if sample_idx == -1:
            sample_idx = int(torch.randint(0, len(self._datasets[ds_idx]), ()))
        return self._datasets[ds_idx][sample_idx]


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: DataConfig,
        roi_size: Sequence[int],
        spacing: Sequence[float],
        cv_index: Optional[int] = None,
        debug: bool = False,
        intensity_properties: Optional[IntensityProperties] = None,
        skip_intensity_properties: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.cv_index = cv_index
        self.debug = debug
        self.roi_size = roi_size
        self.spacing = spacing
        self.cache_dir = Path("/storage/tree-softmax-cache")
        if self.cache_dir.parent.exists():
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.intensity_properties = (
            intensity_properties if not skip_intensity_properties else None
        )
        self.skip_intensity_properties = skip_intensity_properties
        self.name = "" if name is None else name

    def setup(self, stage: str) -> None:
        train_examples = [
            (i, info.splits["train"])
            for i, info in enumerate(self.config.dataset_infos)
            if "train" in info.splits
        ]
        logger.info(
            f"Found {len(train_examples)} training dataset "
            f"with a total of {sum([len(info) for _, info in train_examples])} samples."
        )
        if self.cv_index is not None:
            # TODO Implement stratified splits
            # TODO Add dataset index to train and val examples
            cv_splits = [partition_dataset(d, num_partitions=5) for d in train_examples]
            train_examples = [
                select_cross_validation_folds(
                    d, [i for i in range(5) if i != self.cv_index]
                )
                for d in cv_splits
            ]
            val_examples = [
                select_cross_validation_folds(d, self.cv_index) for d in cv_splits
            ]
        else:
            val_examples = [
                (i, info.splits["val"])
                for i, info in enumerate(self.config.dataset_infos)
                if "val" in info.splits
            ]
            logger.info(
                f"Found {len(val_examples)} validation dataset "
                f"with a total of {sum([len(info) for _, info in val_examples])} samples."
            )
        test_examples = [
            (i, info.splits["test"])
            for i, info in enumerate(self.config.dataset_infos)
            if "test" in info.splits
        ]
        logger.info(
            f"Found {len(test_examples)} test dataset "
            f"with a total of {sum([len(info) for _, info in test_examples])} samples."
        )
        if (
            len(train_examples) > 0
            and self.intensity_properties is None
            and not self.skip_intensity_properties
        ):
            self.intensity_properties = self._load_or_compute_intensity_properties(
                list(zip(*train_examples))[1]
            )
        else:
            logger.info("No intensity properties will be used for this dataset.")

        datasets = [
            PersistentDataset(
                examples[:10] if self.debug else examples,
                get_train_transforms(
                    info=self.config.dataset_infos[ds_idx],
                    roi_size=self.roi_size,
                    spacing=self.spacing,
                    intensity_properties=self.intensity_properties,
                ),
                cache_dir=self._get_cache_dir_for_examples(examples, "train"),
            )
            for ds_idx, examples in train_examples
        ]
        self.train_ds = MultiSourceDataset(datasets, equal_sampling=False)
        self.val_ds = ConcatDataset(
            [
                PersistentDataset(
                    examples[:10] if self.debug else examples,
                    get_validation_transforms(
                        info=self.config.dataset_infos[ds_idx],
                        spacing=self.spacing,
                        intensity_properties=self.intensity_properties,
                    ),
                    cache_dir=self._get_cache_dir_for_examples(examples, "val"),
                )
                for ds_idx, examples in val_examples
            ]
        )
        self.test_ds = ConcatDataset(
            [
                PersistentDataset(
                    examples[:10] if self.debug else examples,
                    get_validation_transforms(
                        info=self.config.dataset_infos[ds_idx],
                        spacing=self.spacing,
                        intensity_properties=self.intensity_properties,
                    ),
                    cache_dir=self._get_cache_dir_for_examples(examples, "test"),
                )
                for ds_idx, examples in test_examples
            ]
        )

    def _load_or_compute_intensity_properties(
        self, examples: List[List[Dict[str, Path]]]
    ) -> IntensityProperties:
        hexdigest = sha256(self.digest(examples).encode()).hexdigest()
        cache_path = self.cache_dir / f"intensity_properties_{hexdigest}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as ifile:
                return pickle.load(ifile)  # type: ignore
        else:
            logger.info("Computing intensity properties for training dataset...")
            with Pool() as pool:
                all_examples: List[Dict[str, Path]] = sum(examples, [])
                all_voxels: Sequence[int] = []
                for voxels_per_example in tqdm(
                    pool.imap(self._get_voxels_from_example, all_examples),
                    total=len(all_examples),
                ):
                    all_voxels += voxels_per_example

            properties = IntensityProperties()
            (
                properties.min,
                properties.robust_lower_bound,
                properties.median,
                properties.robust_upper_bound,
                properties.max,
            ) = np.percentile(all_voxels, (0.0, 0.5, 50, 99.5, 100.0))
            properties.mean = np.mean(all_voxels)
            properties.std = np.std(all_voxels)
            logger.info(properties)

            with cache_path.open("wb") as ofile:
                pickle.dump(properties, ofile)

            return properties

    def _get_voxels_from_example(self, example: Dict[str, Path]) -> List[int]:
        image = itk.imread(example["image"])
        image_data = itk.GetArrayViewFromImage(image)

        seg = itk.imread(example["label"])
        seg_data = itk.GetArrayViewFromImage(seg)

        # Only include voxels with feasible Hounsfield values, because some scanners
        # mask out image areas with intensity values larger than common HU and
        # extremely dense materials are not that important.
        # mask = np.logical_and(image_data > -1024, image_data < 3072)
        mask = np.logical_and(seg_data > 0, seg_data < 255)
        # Subsample voxels for estimation of the intensity properties
        return list(image_data[mask].ravel()[::8])

    def digest(self, examples: List[Dict[str, Path]]):
        return sha256((str(examples) + self.name).encode()).hexdigest()

    def _get_cache_dir_for_examples(
        self, examples: List[Dict[str, Path]], prefix: str
    ) -> Path:
        if not self.cache_dir.exists():
            return None
        # TODO Include transform pipeline into checksum
        path = self.cache_dir / f"{prefix}_{self.name}_{self.digest(examples)}"
        logger.info(f"Using cache directory: {path}")
        return path

    def train_dataloader(self, batch_size: int) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds,
            num_workers=8,
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds,
            num_workers=4,
            shuffle=False,
            batch_size=1,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_ds,
            num_workers=4,
            shuffle=False,
            batch_size=1,
            drop_last=False,
            pin_memory=True,
        )
