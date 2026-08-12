import logging
import pickle
import time
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.transforms.utils import allow_missing_keys_mode

from salt.input_pipeline import (
    IntensityProperties,
    get_postprocess_transforms,
    get_validation_transforms,
)
from salt.utils.inference import sliding_window_inference_with_reduction

logger = logging.getLogger(__name__)


def argmax_leafs(
    inputs: torch.Tensor,
    adjacency_matrix: np.ndarray,
    dim: int = 1,
    pruned: bool = True,
) -> torch.Tensor:
    leaf_nodes = np.where(adjacency_matrix[1:, 1:].sum(axis=1) == 0)[0]
    indices = np.arange(adjacency_matrix.shape[0] - 1, dtype=np.int32)
    indices = indices[leaf_nodes]
    y_pred_leafs = inputs[:, leaf_nodes]
    y_pred_leaf_idx = torch.argmax(y_pred_leafs, axis=dim)
    if pruned:
        return y_pred_leaf_idx
    return torch.tensor(indices).to(inputs.device)[y_pred_leaf_idx]


def main(args: Namespace) -> None:
    logger.info("Loading model...")
    with (args.config_file).open(
        "rb",
    ) as ifile:
        config = pickle.load(ifile)

    model = torch.jit.load(args.model_file)

    pre_processing = get_validation_transforms(
        spacing=config["model"]["voxel_spacing"],
        info=None,
        intensity_properties=(
            IntensityProperties(
                mean=config["intensity_properties"]["mean"],
                std=config["intensity_properties"]["std"],
            )
            if config["intensity_properties"] is not None
            else None
        ),
    )
    model.cuda()
    model.eval()

    # HACK Disable JIT profiling to speed up first inference rounds
    # See: https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)

    if args.data_dir.is_dir():
        inputs = sorted(args.data_dir.glob("*.nii.gz"))
    else:
        inputs = [args.data_dir]
    times = []
    for i, input_path in enumerate(inputs, start=1):
        logger.info(f"{i}/{len(inputs)}: Loading image {input_path.name}...")
        with allow_missing_keys_mode(pre_processing):
            example = pre_processing({"image": input_path})
        logger.info(f"Computing model output for {input_path.name}...")
        base_dict = {"image": input_path.name.replace(".nii.gz", "")}
        start = time.time()
        with torch.cuda.amp.autocast(), torch.no_grad():
            pred = (
                sliding_window_inference_with_reduction(
                    inputs=example["image"].unsqueeze(0).cuda(),
                    roi_size=config["model"]["roi_size"],
                    sw_batch_size=2,
                    predictor=model,
                    progress=True,
                    overlap=0.5,
                    mode="gaussian",
                    cval=(
                        (-1024 - config["intensity_properties"]["mean"])
                        / config["intensity_properties"]["std"]
                        if config["intensity_properties"] is not None
                        else 0.0
                    ),
                    reduction_fn=partial(
                        argmax_leafs, adjacency_matrix=config["adjacency_matrix"]
                    ),
                    # device="cpu",
                )
                .cpu()
                .to(torch.int64)
            )
        base_dict["prediction_time"] = time.time() - start
        logger.info("Applying post-processing and saving image")
        postprocess_time = time.time()
        pred = get_postprocess_transforms(output_dir=args.output_dir)(
            {"pred": pred, "image_meta_dict": example["image_meta_dict"]}
        )
        base_dict["postprocess_time"] = time.time() - postprocess_time
        base_dict["total_time"] = time.time() - start
        base_dict["shape"] = tuple(pred["pred"][0].shape)
        times.append(base_dict)
    pd.DataFrame(times).to_csv(args.output_dir / "times.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("models/foobar-31/model.pt"),
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("models/foobar-31/config.pkl"),
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()

    main(args)
