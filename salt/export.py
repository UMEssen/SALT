import logging
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Tuple

import torch
from monai.networks.nets import DynUNet

from salt.core.activations import TreeSoftmax

logger = logging.getLogger(__name__)


class TSMEnsemble(torch.nn.Module):
    def __init__(self, models: List[torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        results = self.models[0](x)
        for model in self.models[1:]:
            results += model(x)
        return results / len(self.models)


def trace_model(model: torch.nn.Module, roi_size: Tuple, output_file: Path):
    traced_model = torch.jit.trace(model, torch.rand(1, 1, *roi_size, device="cuda"))
    traced_model = torch.jit.optimize_for_inference(traced_model)
    traced_model.save(output_file)


def main(args: Namespace) -> None:
    if args.cv is not None:
        cvs = range(1, args.cv + 1)
    else:
        cvs = [None]

    models = []
    out_config = None
    for cv in cvs:
        if cv is None:
            config_path = args.train_dir / "config.pkl"
            model_path = args.train_dir / "train" / f"model-{args.weights_type}.pt"
        else:
            config_path = args.train_dir / f"cv-{cv}" / "config.pkl"
            model_path = (
                args.train_dir / f"cv-{cv}" / "train" / f"model-{args.weights_type}.pt"
            )
        logger.info(f"Loading model {model_path}")
        with config_path.open("rb") as ifile:
            config = pickle.load(ifile)

        if out_config is None:
            out_config = config
        else:
            for k, v in config.items():
                if k in {"intensity_properties", "args"}:
                    if isinstance(out_config[f"{k}"], List):
                        out_config[f"{k}"].append(v)
                    else:
                        out_config[f"{k}"] = [out_config[k], v]
                    continue
                check = v == out_config[k]
                error = f"The config {k} are not the same: {v} vs. {out_config[k]}"
                if not isinstance(check, bool):
                    assert check.all(), error
                else:
                    assert check, error
        base_model = torch.nn.Sequential(
            DynUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=config["model"]["num_classes"],
                kernel_size=config["model"]["kernel_sizes"],
                strides=config["model"]["strides"],
                upsample_kernel_size=config["model"]["strides"][1:],
                res_block=True,
            ),
            TreeSoftmax(
                adjacency_matrix=config["adjacency_matrix"],
            ),
        )
        base_model.load_state_dict(
            {
                k.replace("base_model.", ""): x
                for k, x in torch.load(model_path)["model"].items()
            }
        )
        base_model.eval()
        base_model.cuda()
        models.append(base_model)

    if len(models) == 1:
        export_model = models[0]
        if out_config["intensity_properties"] is not None:
            out_config["intensity_properties"] = {
                "mean": out_config["intensity_properties"].mean,
                "std": out_config["intensity_properties"].std,
            }
    else:
        out_config["intensity_properties"] = None
        export_model = TSMEnsemble(models)

    args.output_dir.mkdir(exist_ok=True, parents=True)
    with (args.output_dir / "config.pkl").open("wb") as ofile:
        pickle.dump(out_config, ofile)

    if args.ensemble:
        trace_model(
            export_model, out_config["model"]["roi_size"], args.output_dir / "model.pt"
        )
    elif len(models) == 1:
        trace_model(
            export_model,
            out_config["model"]["roi_size"],
            args.output_dir / "model.pt",
        )
    else:
        for i, m in enumerate(models, start=1):
            trace_model(
                m, out_config["model"]["roi_size"], args.output_dir / f"model_{i}.pt"
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--weights-type", default="best", choices=["best", "latest"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cv", type=int, default=None)
    parser.add_argument("--ensemble", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
