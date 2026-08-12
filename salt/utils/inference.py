from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import (
    compute_importance_map,
    dense_patch_slices,
    get_valid_patch_size,
)
from monai.inferers.utils import _get_scan_interval
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    fall_back_tuple,
    look_up_option,
)
from tqdm import tqdm


def sliding_window_inference_with_reduction(  # noqa: C901
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Optional[Union[torch.device, str]] = None,
    device: Optional[Union[torch.device, str]] = None,
    reduction_fn: Callable[..., torch.Tensor] = torch.argmax,
    reduction_dim: int = 1,
    output_dtype: torch.dtype = torch.uint8,
    progress: bool = False,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    # Sometimes the scans have a lot of empty slices at the beginning and at the end.
    # We skip these slices for the prediction
    start_point = 0
    while inputs[:, :, :, :, start_point].std() == 0:
        start_point += 1
    end_point = inputs.shape[-1]
    while inputs[:, :, :, :, end_point - 1].std() == 0:
        end_point -= 1
    original_shape = inputs.shape
    inputs = inputs[:, :, :, :, start_point:end_point]
    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    # Identifiy outer and inner dimensions for the sliding window and aggregation
    outer_dim = 2  # TODO Find heuristic for this

    # TODO: Bug
    # If the scan is longer than the roi_size but not a multiple of the roi_size
    # there are problems putting the overlapping part of the input into the output
    # In this code when we use 0.5 all windows only overlap with one other window, except for the last one, which overlaps with two. When this happens, some weird artifacts appear on the overlap. This has to do with the gaussian importance map, although this also appears with the constant importance map. It is probably related to the fact that we use a buffer to store the probabilities and weights and overwrite it whenever we move to the next window.
    # Things I have tried:
    # - Use a different importance map function: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/sliding_window_prediction.py
    # - Change the steps of the outer window to be overlapping with two other windows all the time (in case overlap is 0.5). This is what nnUnet does, but it produces the same artifacts. Actually it's even worse, since now the artifacts are in between every window.

    # Current bad fix: Make the scan a multiple of the roi_size in the outer_dim
    outer_dim_pad = None
    if (
        inputs.shape[outer_dim + 2] > roi_size[outer_dim]
        and inputs.shape[outer_dim + 2] % roi_size[outer_dim] != 0
    ):
        divi = inputs.shape[outer_dim + 2] // roi_size[outer_dim]
        mul_padding = roi_size[outer_dim] * (divi + 1) - inputs.shape[outer_dim + 2]
        outer_dim_pad = [mul_padding // 2, mul_padding - (mul_padding // 2)]

    batch_size, _, *orig_image_size = inputs.shape
    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size_safe: Tuple[int] = fall_back_tuple(roi_size, orig_image_size)

    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size_safe[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if outer_dim_pad is not None:
        pad_size[(2 - outer_dim) * 2 : (2 - outer_dim) * 2 + 2] = outer_dim_pad

    if max(pad_size) > 0:
        inputs = F.pad(
            inputs,
            pad=pad_size,
            mode=look_up_option(padding_mode, PytorchPadMode),
            value=cval,
        )
    image_size = inputs.shape[2:]
    patch_size = get_valid_patch_size(image_size, roi_size_safe)

    importance_map = compute_importance_map(
        patch_size=patch_size,
        mode=mode,
        sigma_scale=sigma_scale,
        device=sw_device,
    )
    importance_map = torch.clamp(
        importance_map,
        min=max(importance_map[importance_map != 0].min().item(), 1e-3),
    )
    importance_map = convert_data_type(
        importance_map, torch.Tensor, sw_device, compute_dtype
    )[0]

    # Allocate buffers
    output = torch.empty(
        tuple(x for i, x in enumerate(inputs.shape) if i != reduction_dim),
        dtype=output_dtype,
        device=device,
    )

    slab_probabilities: Optional[torch.Tensor] = None
    slab_weights = torch.zeros(
        tuple(
            inputs.shape[i]
            if i != outer_dim + 2  # account for batch and channel dimensions
            else roi_size_safe[outer_dim]
            for i in range(inputs.ndim)
        ),
        dtype=compute_dtype,
        device=sw_device,
    )

    # Iterate over outer dimension and aggregate a full slab of reduced predictions
    outer_step_size = int(roi_size_safe[outer_dim] * (1 - overlap))
    outer_indices = list(
        range(
            0,
            inputs.shape[outer_dim + 2] - roi_size_safe[outer_dim] + 1,
            outer_step_size,
        )
    )
    if outer_indices[-1] != image_size[outer_dim] - roi_size_safe[outer_dim]:
        outer_indices.append(image_size[outer_dim] - roi_size_safe[outer_dim])
    last_outer_dim_idx = -1
    for outer_idx, outer_dim_idx in enumerate(
        tqdm(outer_indices, leaf=True, position=0) if progress else outer_indices
    ):
        # Move old probabilities and weights based on the actual step size of this slab
        if outer_idx > 0:
            assert slab_probabilities is not None
            actual_step_size = outer_dim_idx - last_outer_dim_idx
            assert 0 < actual_step_size <= outer_step_size
            new_slices = tuple(
                slice(None)
                if i != outer_dim + 2  # account only for batch dimension
                else slice(None, -actual_step_size)
                for i in range(slab_probabilities.ndim)
            )
            old_slices = tuple(
                slice(None)
                if i != outer_dim + 2  # account only for batch dimension
                else slice(actual_step_size, None)
                for i in range(slab_probabilities.ndim)
            )
            null_slices = tuple(
                slice(None)
                if i != outer_dim + 2  # account only for batch dimension
                else slice(-actual_step_size - 1, None)
                for i in range(slab_probabilities.ndim)
            )

            slab_probabilities[new_slices] = slab_probabilities[old_slices]
            slab_weights[new_slices] = slab_weights[old_slices]

            slab_probabilities[null_slices] = 0.0
            slab_weights[null_slices] = 0.0

        # Take slab of input images and apply padding
        slab_slices = tuple(
            slice(None)
            if i != outer_dim + 2  # account for batch and channel dimensions
            else slice(outer_dim_idx, outer_dim_idx + roi_size_safe[outer_dim])
            for i in range(inputs.ndim)
        )
        slab_input = inputs[slab_slices]

        # Compute crop locations in slab
        scan_interval = _get_scan_interval(
            slab_input.shape[2:], roi_size_safe, num_spatial_dims, overlap
        )
        slices = dense_patch_slices(slab_input.shape[2:], roi_size_safe, scan_interval)
        num_win = len(slices)
        total_slices = num_win * batch_size

        # Perform sliding window inference on slab
        slice_indices = list(range(0, total_slices, sw_batch_size))
        for slice_idx in (
            tqdm(slice_indices, leaf=False, position=1) if progress else slice_indices
        ):
            # Get crops from slices
            slice_range = range(slice_idx, min(slice_idx + sw_batch_size, total_slices))
            unravel_slice = [
                [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)]
                + list(slices[idx % num_win])
                for idx in slice_range
            ]
            window_data = torch.cat(
                [
                    convert_data_type(slab_input[win_slice], torch.Tensor)[0]
                    for win_slice in unravel_slice
                ]
            ).to(sw_device)

            # Compute probabilities and aggregate
            probabilities = predictor(window_data, *args, **kwargs)

            if slab_probabilities is None:
                output_classes = probabilities.shape[1]
                slab_probabilities = torch.zeros(
                    (batch_size, output_classes)
                    + tuple(
                        image_size[i] if i != outer_dim else patch_size[outer_dim]
                        for i in range(len(image_size))
                    ),
                    dtype=compute_dtype,
                    device=sw_device,
                )

            probabilities *= importance_map.unsqueeze(0).unsqueeze(0)
            for slice_idx, win_slice in enumerate(unravel_slice):
                slab_probabilities[win_slice] += probabilities[
                    slice_idx : slice_idx + 1
                ]
                slab_weights[win_slice] += importance_map

        # Apply reduction operation and move partial output to output buffer
        assert slab_probabilities is not None
        assert slab_weights is not None
        copy_size = (
            roi_size_safe[outer_dim]
            if outer_idx == len(outer_indices) - 1
            else outer_step_size
        )
        reduction_slices = tuple(
            slice(None)
            if i != outer_dim + 2  # account for batch and channel dimensions
            else slice(None, copy_size)
            for i in range(slab_probabilities.ndim)
        )
        predictions = reduction_fn(
            slab_probabilities[reduction_slices] / slab_weights[reduction_slices],
            dim=reduction_dim,
        )
        output[
            tuple(
                slice(None)
                if i != outer_dim + 1  # account for batch dimension
                else slice(outer_dim_idx, outer_dim_idx + copy_size)
                for i in range(output.ndim)
            )
        ] = predictions
        last_outer_dim_idx = outer_dim_idx

    # Crop to original image size
    pad_size = pad_size[::-1]
    output = output[
        [slice(None)]
        + [
            slice(pad_size[i * 2 + 1], -pad_size[i * 2])
            if pad_size[i * 2] != 0 or pad_size[i * 2 + 1] != 0
            else slice(None)
            for i in range(len(pad_size) // 2)
        ]
    ]

    # TODO: Do this with indexing in output instead of repadding
    repad = [start_point, original_shape[-1] - end_point]
    if max(repad) > 0:
        output = F.pad(
            output,
            pad=repad,
            mode="constant",
            value=0,
        )

    if isinstance(inputs, MetaTensor):
        return convert_to_dst_type(output, inputs, device=device)[0]

    return output
