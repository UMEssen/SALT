import argparse
import pathlib
import pickle
import time
from collections import deque
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from monai.losses import DiceLoss
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from tensorboardX import SummaryWriter

from salt.core.activations import TreeSoftmax
from salt.core.adjacency_matrix import (
    bitpack_tree,
    reachability_from_adjacency_matrix,
    sibling_from_adjacency_matrix,
)
from salt.data.discovery import find_datasets
from salt.data.module import DataModule
from salt.dynunet import DynUNet
from salt.utils.export import create_itksnap_label_definition
from salt.utils.inference import sliding_window_inference_with_reduction
from salt.utils.visualization import make_palette


class TrainingModel(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        activation_fn: torch.nn.Module,
        adjacency_matrix: np.ndarray,
        num_classes: int,
        sink_mask: np.ndarray,
        sink_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.activation_fn = activation_fn
        self.adjacency_matrix = adjacency_matrix
        self.num_classes = num_classes
        self.sink_mask = sink_mask
        self.sink_weight = sink_weight

        self.reachability_matrix = reachability_from_adjacency_matrix(adjacency_matrix)

        # Create bitmasks from adjacency matrix
        self.bitmask_classes, self.bitmask_groups = bitpack_tree(adjacency_matrix)

        # Remove root node from encodings
        self.bitmask_classes = torch.as_tensor(self.bitmask_classes[1:])
        self.bitmask_groups = torch.as_tensor(self.bitmask_groups[1:])

    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        yhats = self.base_model(images)
        is_deep_supervision = len(yhats.shape) == 6
        if is_deep_supervision:
            yhats = self.activation_fn(yhats, dim=2)
        else:
            yhats = self.activation_fn(yhats, dim=1)

        with torch.no_grad():
            yhat_preds = self._argmax_leafs(yhats, dim=2 if is_deep_supervision else 1)

        if is_deep_supervision:
            loss = torch.stack(
                [
                    torch.stack(
                        [
                            0.5**ds_idx
                            * torch.stack(
                                self._compute_loss(
                                    yhat_ds,
                                    yhat_pred_ds,
                                    target,
                                    mask,
                                )
                            )
                            for ds_idx, (yhat_ds, yhat_pred_ds) in enumerate(
                                zip(yhat.unbind(0), yhat_pred.unbind(0))
                            )
                        ],
                        dim=0,
                    ).sum(0)
                    for yhat, yhat_pred, target, mask in zip(
                        yhats, yhat_preds, targets, masks
                    )
                ]
            )
        else:
            loss = torch.stack(
                [
                    torch.stack(
                        self._compute_loss(
                            yhat,
                            yhat_pred,
                            target,
                            mask,
                        )
                    )
                    for yhat, yhat_pred, target, mask in zip(
                        yhats, yhat_preds, targets, masks
                    )
                ]
            )

        # Compute metric
        with torch.no_grad():
            cm_update = torch.stack(
                [
                    self._compute_confusion_matrix(yhat_pred, target, mask)
                    for yhat_pred, target, mask in zip(
                        yhat_preds[:, 0] if is_deep_supervision else yhat_preds,
                        targets,
                        masks,
                    )
                ]
            )
        return (
            loss,
            cm_update,
            yhat_preds[:, 0] if is_deep_supervision else yhat_preds,
        )

    def val_step(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        roi_size: Sequence[int],
        batch_size: int,
        gpus: Sequence[int],
        cval: float = -1.0,
    ) -> torch.Tensor:
        yhats = sliding_window_inference_with_reduction(
            inputs=images,
            roi_size=roi_size,
            sw_batch_size=batch_size,
            predictor=torch.nn.DataParallel(
                module=torch.nn.Sequential(
                    self.base_model,
                    self.activation_fn,
                ),
                device_ids=gpus,
            ),  # self.base_model,
            overlap=0.25,
            cval=cval,
            reduction_fn=self._argmax_leafs,
        ).to(
            torch.int32
        )  # HACK MONAI casts to float???
        return torch.stack(
            [
                self._compute_confusion_matrix(yhat, target, mask)
                for yhat, target, mask in zip(yhats, targets, masks)  # [0]
            ]
        )

    def _compute_loss(
        self,
        yhat: torch.Tensor,
        yhat_pred: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-7,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = y.reshape(-1)
        yhat_pred = yhat_pred.reshape(-1)
        yhat = yhat.reshape(yhat.shape[0], -1).transpose(0, 1)

        # Filter ignored voxels
        valid_mask = y != 255
        y = y[valid_mask]
        yhat = yhat[valid_mask]

        class_indices = torch.nonzero(mask)[:, 0]

        # Sink class loss
        yhat_bm = torch.index_select(
            self.bitmask_classes.to(y.device),
            0,
            yhat_pred[valid_mask],
        )
        sink_y: List[torch.Tensor] = []
        sink_yhat: List[torch.Tensor] = []
        # print("class", class_indices)
        if self.sink_weight > 0.0:
            for sink_idx in np.nonzero(self.sink_mask)[0]:
                # Get foreground mask as fusion of all sibling masks
                siblings = sibling_from_adjacency_matrix(self.adjacency_matrix)[1:, 1:][
                    sink_idx
                ]
                siblings[sink_idx] = 0
                sibling_indices = torch.as_tensor(
                    np.nonzero(siblings)[0], device=y.device
                )
                existing_siblings = torch.isin(sibling_indices, class_indices)
                sibling_mask = torch.isin(y, sibling_indices)
                # print(
                #     sink_idx,
                #     "siblings",
                #     sibling_indices,
                #     "existing",
                #     sibling_indices[existing_siblings],
                #     "not existing",
                #     sibling_indices[~existing_siblings],
                # )

                if not torch.all(existing_siblings):
                    for idx in sibling_indices[~existing_siblings]:
                        sibling_mask = torch.logical_or(
                            sibling_mask,
                            yhat_pred[valid_mask] == idx,
                        )
                        # print(
                        #     sink_idx,
                        #     "predicted sibling",
                        #     idx,
                        #     torch.sum(new_mask),
                        # )
                # print("sibling mask", torch.sum(sibling_mask))
                if torch.sum(sibling_mask) == 0:
                    # print(sink_idx, "skipping sink")
                    continue
                # Get parent mask
                parent_idx = self.adjacency_matrix[1:, 1:][:, sink_idx].argmax()
                parent_mask = self._bitwise_mask_equal(
                    yhat_bm,
                    self.bitmask_classes.to(y.device)[parent_idx],
                    self.bitmask_groups.to(y.device)[parent_idx],
                )

                # get label and prediction for sink class
                sink_y.append(torch.logical_and(parent_mask, ~sibling_mask).detach())
                sink_yhat.append(yhat[:, sink_idx])

        if len(sink_y) == 0:
            sink_dice = torch.as_tensor(0.0, device=y.device)
        else:
            sink_dice = (
                self.sink_weight
                * DiceLoss()(  # TODO Use efficient Dice computation
                    torch.stack(sink_yhat, dim=1).unsqueeze(0).transpose(1, 2),
                    torch.stack(sink_y, dim=1).unsqueeze(0).transpose(1, 2),
                )
            )
            # TODO Add cross entropy loss for sink classes?

        # Filter classes
        class_indices = torch.nonzero(mask)[:, 0]
        yhat = yhat[:, class_indices]

        # Create dense label encoding based on reachability matrix
        rmt = torch.tensor(self.reachability_matrix[1:, 1:].T, dtype=yhat.dtype).to(
            y.device
        )
        y = rmt[:, class_indices][y.type(torch.long)]

        # Dice Loss
        dice = DiceLoss(smooth_nr=0.0)(  # TODO Use efficient Dice computation
            yhat.unsqueeze(0).transpose(1, 2), y.unsqueeze(0).transpose(1, 2)
        )

        # Cross Entropy
        ce_loss = -y * torch.log(torch.clip(yhat, eps, 1 - eps))
        ce_loss = ce_loss.sum(dim=1)
        ce_loss = ce_loss.mean()
        return ce_loss, dice, sink_dice

    def _compute_confusion_matrix(
        self, yhat_pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        y = y.reshape(-1)
        yhat_pred = yhat_pred.reshape(-1)

        # Filter ignored voxels
        valid_mask = y != 255
        y = y[valid_mask]
        yhat_pred = yhat_pred[valid_mask]

        # Translate class indices to bitmasks with path encodings
        y_bm = torch.index_select(
            torch.tensor(self.bitmask_classes).to(y.device), 0, y.type(torch.int32)
        )
        yhat_bm = torch.index_select(
            torch.tensor(self.bitmask_classes).to(yhat_pred.device), 0, yhat_pred
        )

        cm_update = torch.zeros(
            (self.num_classes, 3), dtype=torch.float64, device=y.device
        )
        for class_idx in range(self.num_classes):
            if mask[class_idx]:
                cm_update[class_idx] = self._get_stats_by_label(
                    yhat_bm,
                    y_bm,
                    torch.tensor(self.bitmask_classes[class_idx]).to(y.device),
                    torch.tensor(self.bitmask_groups[class_idx]).to(y.device),
                )

        return cm_update

    def _get_stats_by_label(
        self,
        yhat: torch.Tensor,
        y: torch.Tensor,
        bm_class: np.ndarray,
        bm_group: np.ndarray,
    ) -> torch.Tensor:
        y = self._bitwise_mask_equal(y, bm_class, bm_group)
        yhat = self._bitwise_mask_equal(yhat, bm_class, bm_group)

        tp = torch.logical_and(y, yhat).sum()
        fp = torch.logical_and(torch.logical_not(y), yhat).sum()
        fn = torch.logical_and(y, torch.logical_not(yhat)).sum()

        return torch.stack([tp, fp, fn], axis=-1)

    def _bitwise_mask_equal(
        self, x: torch.Tensor, bm_class: torch.Tensor, bm_group: torch.Tensor
    ) -> torch.Tensor:
        comp = torch.eq(torch.bitwise_and(x, bm_group), bm_class)
        if comp.ndim == 1:
            return comp
        assert comp.ndim == 2
        return torch.all(comp, axis=1)

    def _argmax_leafs(
        self, yhat: torch.Tensor, dim: int = 1, inplace: bool = False
    ) -> torch.Tensor:
        assert dim in {1, 2}
        leaf_nodes = np.where(self.adjacency_matrix[1:, 1:].sum(axis=1) == 0)[0]

        if inplace:
            if dim == 1:
                yhat[:, ~leaf_nodes] = 0.0
            else:
                yhat[:, :, ~leaf_nodes] = 0.0
            return torch.argmax(yhat, axis=dim)

        indices = np.arange(self.num_classes, dtype=np.int32)
        indices = indices[leaf_nodes]
        y_pred_leafs = yhat[:, leaf_nodes] if dim == 1 else yhat[:, :, leaf_nodes]
        y_pred_leaf_idx = torch.argmax(y_pred_leafs, axis=dim)
        return torch.tensor(indices).to(yhat.device)[y_pred_leaf_idx]


def blend_images(
    image: torch.Tensor, seg: torch.Tensor, alpha: float = 0.25
) -> np.ndarray:
    lut = make_palette(256).astype(np.float32) / 255.0
    image = image.numpy()
    seg_rgb = lut[seg.numpy().squeeze(1)]
    seg_rgb = np.moveaxis(seg_rgb, -1, 1)

    blended = (1.0 - alpha) * image + alpha * seg_rgb
    result: np.ndarray = np.where(seg == 0, image, blended)
    result = np.clip(result, 0.0, 1.0)
    return result


def _get_kernels_strides(
    sizes: Sequence[int], spacings: Sequence[float]
) -> Tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]:
    """Adapted from nnUnet"""
    input_size = sizes
    modsized: Sequence[Union[float, int]] = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, modsized)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(modsized, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        modsized = [i / j for i, j in zip(modsized, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


def main(args: argparse.Namespace) -> None:
    assert args.batch_size % len(args.gpus) == 0

    data_config = find_datasets(args.data_dir)
    print("Number of classes:", data_config.num_classes)
    for label_idx, label in enumerate(data_config.labels):
        print(label_idx, ">".join(label))

    device = torch.device("cuda")

    roi_size = (192, 192, 48)
    # voxel_spacing = (1.5, 1.5, 5.0)
    # roi_size = (128, 128, 128)
    voxel_spacing = (1.5, 1.5, 1.5)
    kernel_sizes, strides = _get_kernels_strides(roi_size, voxel_spacing)

    # TODO Add Deep Supervision support
    # assert not args.deep_supervision
    base_model = torch.nn.Sequential(
        # VNet(spatial_dims=3, in_channels=1, out_channels=data_config.num_classes),
        DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=data_config.num_classes,
            kernel_size=kernel_sizes,
            strides=strides,
            upsample_kernel_size=strides[1:],
            deep_supervision=args.deep_supervision,
            deep_supr_num=1,
            res_block=True,
        ),
    )
    model = TrainingModel(
        base_model=base_model,
        activation_fn=TreeSoftmax(adjacency_matrix=data_config.adjacency_matrix),
        adjacency_matrix=data_config.adjacency_matrix,
        num_classes=data_config.num_classes,
        sink_mask=data_config.sink_mask,
        sink_weight=args.sink_weight,
    ).to(device)
    model_parallel = torch.nn.DataParallel(model, args.gpus)

    data_module = DataModule(
        config=data_config,
        roi_size=roi_size,
        spacing=voxel_spacing,
        cv_index=args.cross_validation_index,
        debug=args.debug,
        intensity_properties=None,
        skip_intensity_properties=True,
        name=args.train_dir.name,
    )
    data_module.setup("fit")
    train_dl = data_module.train_dataloader(batch_size=args.batch_size)
    val_dl = data_module.val_dataloader()

    # Save essential parameters to training dir
    args.train_dir.mkdir(parents=True, exist_ok=True)
    with (args.train_dir / "itksnap_labels.txt").open("w") as ofile:
        ofile.write(create_itksnap_label_definition(data_config.labels))
    with (args.train_dir / "itksnap_labels_pruned.txt").open("w") as ofile:
        ofile.write(create_itksnap_label_definition(data_config.leaf_labels))
    with (args.train_dir / "config.pkl").open("wb") as ofile:
        pickle.dump(
            {
                "adjacency_matrix": data_config.adjacency_matrix,
                "intensity_properties": data_module.intensity_properties,
                "label_names": data_config.labels,
                "model": {
                    "num_classes": data_config.num_classes,
                    "kernel_sizes": kernel_sizes,
                    "strides": strides,
                    "roi_size": roi_size,
                    "voxel_spacing": voxel_spacing,
                },
                "args": args,
            },
            ofile,
        )

    # optimizer = torch.optim.RMSprop(model.parameters(), 2.5e-4)  # , weight_decay=1e-5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=5e-5
    )
    if args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()

    # dice_metric = ...  # DiceMetric(include_background=True, reduction="mean")
    # dice_metric_batch = (
    #     ...
    # )  # DiceMetric(include_background=True, reduction="mean_batch")

    # post_trans = Compose(
    #     [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    # )

    initial_epoch = 0
    best_model_dice = 0.0
    if args.resume is not None:
        state = torch.load(args.resume)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        initial_epoch = state["epoch"]
        best_model_dice = state["best_metric"]

    writer_train = SummaryWriter(log_dir=args.train_dir / "train")
    writer_val = SummaryWriter(log_dir=args.train_dir / "val")

    output_labels = ["/".join(label) for label in data_config.labels]
    train_timings: deque[float] = deque(maxlen=10)
    for epoch in range(initial_epoch, args.epochs):
        model.train()  # model_parallel.train()
        epoch_dloss = 0.0
        epoch_celoss = 0.0
        epoch_sloss = 0.0
        epoch_cm = torch.zeros((data_config.num_classes, 3), dtype=torch.float64)
        for batch_idx, batch_data in enumerate(train_dl):
            batch_started = time.time()

            batch_images = batch_data["image"]
            batch_labels = batch_data["label"]
            batch_masks = batch_data["mask"]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                loss, cm_update, yhats = model_parallel(  # model_parallel(
                    batch_images, batch_labels, batch_masks
                )
                loss = loss.nanmean(0)
                celoss = loss[0]
                dloss = loss[1]
                sloss = loss[2]
                loss = loss.sum()

            if args.mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if batch_idx == 0:
                for label_name, label_weights in zip(
                    output_labels, base_model[0].output_block.conv.conv.weight
                ):
                    writer_train.add_histogram(
                        "weights_logits/" + label_name, label_weights, epoch + 1
                    )

                # image_for_plotting = (
                #     batch_images[:1] * data_module.intensity_properties.std
                #     + data_module.intensity_properties.mean
                # )
                # image_for_plotting = torch.clip(image_for_plotting, -1024, 1024) / 1024
                image_for_plotting = batch_images[:1]
                overview_image = np.concatenate(
                    [
                        np.repeat(image_for_plotting, 3, axis=1),  # / 2 + 0.5,
                        blend_images(image_for_plotting, batch_labels[:1]),
                        blend_images(image_for_plotting, yhats[:1].cpu().unsqueeze(1)),
                    ],
                    axis=2,
                ).swapaxes(2, 3)
                plot_2d_or_3d_image(
                    overview_image,
                    epoch + 1,
                    writer_train,
                    max_channels=3,
                    frame_dim=-1,
                    max_frames=5,
                    tag="preview",
                )
            del yhats

            # add metric computation
            epoch_dloss += dloss.item()
            epoch_celoss += celoss.item()
            epoch_sloss += sloss.item()
            epoch_cm += cm_update.cpu().sum(0)

            # Update timings
            train_timings.append(time.time() - batch_started)

            print(
                f"Epoch {epoch+1}/{args.epochs}",
                f"Batch {batch_idx+1}",
                "ce_loss",
                f"{epoch_celoss / (batch_idx + 1):.5f}",
                "dice_loss",
                f"{epoch_dloss / (batch_idx + 1):.5f}",
                "sink_loss",
                f"{epoch_sloss / (batch_idx + 1):.5f}",
                "sec/batch",
                f"{np.mean(train_timings):.3f}",
            )

        # Write tensorboard summaries
        dice_scores = epoch_cm[:, 0] / (
            epoch_cm[:, 0] + 0.5 * epoch_cm[:, 1] + 0.5 * epoch_cm[:, 2]
        )
        for label_name, label_dice in zip(output_labels, dice_scores):
            label_dice_value = label_dice.item()
            writer_train.add_scalar(
                "dice/by-label/" + label_name,
                0.0 if np.isnan(label_dice_value) else label_dice_value,
                epoch + 1,
            )
        writer_train.add_scalar("dice/macro", dice_scores.nanmean().item(), epoch + 1)
        writer_train.add_scalar("loss/ce", epoch_celoss / (batch_idx + 1), epoch + 1)
        writer_train.add_scalar("loss/dice", epoch_dloss / (batch_idx + 1), epoch + 1)
        writer_train.add_scalar("loss/sink", epoch_sloss / (batch_idx + 1), epoch + 1)
        writer_train.add_scalar("learning_rate", lr_scheduler.get_last_lr(), epoch + 1)

        # Update learning rate
        lr_scheduler.step()

        # Evaluate the current model state
        if (epoch + 1) % args.evaluation_interval == 0:
            torch.cuda.empty_cache()
            model.eval()
            epoch_dloss = 0.0
            epoch_celoss = 0.0
            epoch_cm = torch.zeros((data_config.num_classes, 3), dtype=torch.float64)
            batch_idx = 0
            for batch_data in val_dl:
                torch.cuda.empty_cache()
                with torch.cuda.amp.autocast(
                    enabled=args.mixed_precision
                ), torch.no_grad():
                    cm_update = model.val_step(
                        batch_data["image"].to(device),
                        batch_data["label"].to(device),
                        batch_data["mask"].to(device),
                        roi_size=roi_size,
                        batch_size=args.batch_size,
                        gpus=args.gpus,
                        cval=(
                            (-1024 - data_module.intensity_properties.mean)
                            / data_module.intensity_properties.std
                            if data_module.intensity_properties is not None
                            else 0.0
                        ),
                    )

                epoch_cm += cm_update.cpu().sum(0)
                batch_idx += 1

            # Write tensorboard summaries
            dice_scores = epoch_cm[:, 0] / (
                epoch_cm[:, 0] + 0.5 * epoch_cm[:, 1] + 0.5 * epoch_cm[:, 2]
            )
            for label_name, label_dice in zip(output_labels, dice_scores):
                label_dice_value = label_dice.item()
                writer_val.add_scalar(
                    "dice/by-label/" + label_name,
                    0.0 if np.isnan(label_dice_value) else label_dice_value,
                    epoch + 1,
                )
            eval_dice = dice_scores.nanmean().item()
            writer_val.add_scalar("dice/macro", eval_dice, epoch + 1)

            if eval_dice > best_model_dice:
                print(
                    f"New best model found, Dice score improved from {best_model_dice:.4f} to {eval_dice:.4f}"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch + 1,
                        "best_metric": eval_dice,
                    },
                    args.train_dir / "train" / "model-best.pt",
                )
                best_model_dice = eval_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_metric": best_model_dice,
                },
                args.train_dir / "train" / "model-latest.pt",
            )


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--cross-validation-index", type=int)
    cli.add_argument("--data-dir", type=pathlib.Path, required=True)
    cli.add_argument("--debug", "-d", default=False, action="store_true")
    cli.add_argument("--epochs", type=int, default=100)
    cli.add_argument("--evaluation-interval", type=int, default=5)
    cli.add_argument("--mixed-precision", default=False, action="store_true")
    cli.add_argument("--resume", type=pathlib.Path)
    cli.add_argument("--train-dir", type=pathlib.Path, required=True)
    cli.add_argument("--deep-supervision", default=False, action="store_true")
    cli.add_argument("--learning-rate", type=float, default=2.5e-4)
    cli.add_argument("--batch-size", type=int, default=4)
    cli.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    cli.add_argument("--weight-decay", type=float, default=1e-5)
    cli.add_argument("--sink-weight", type=float, default=0.1)
    args = cli.parse_args()

    main(args)
