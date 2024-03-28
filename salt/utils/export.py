from typing import List, Optional, Tuple

from salt.utils.visualization import make_palette


def create_itksnap_label_definition(
    labels: List[Tuple[str, ...]], label_indices: Optional[List[int]] = None
) -> str:
    if label_indices is None:
        label_indices = list(range(len(labels)))

    # Add an ignore label
    if 255 not in label_indices:
        labels = labels.copy()
        labels.append(("ignore",))
        label_indices = label_indices.copy()
        label_indices.append(255)

    template = """################################################
# ITK-SnAP Label Description File
# File format:
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Fields:
#    IDX:   Zero-based index
#    -R-:   Red color component (0..255)
#    -G-:   Green color component (0..255)
#    -B-:   Blue color component (0..255)
#    -A-:   Label transparency (0.00 .. 1.00)
#    VIS:   Label visibility (0 or 1)
#    IDX:   Label mesh visibility (0 or 1)
#  LABEL:   Label description
################################################
"""

    color_lut = make_palette(256)
    for label_idx, label in zip(label_indices, labels):
        pretty_name = ">".join(label)
        is_foreground = int(pretty_name != "background")
        template += (
            f"{label_idx:4d}{color_lut[label_idx, 0]:5d}"
            f"{color_lut[label_idx, 1]:5d}{color_lut[label_idx, 2]:5d}"
            f"{is_foreground:9d}{is_foreground:3d}{is_foreground:3d}"
            f'     "{pretty_name}"\n'
        )

    return template
