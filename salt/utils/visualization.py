import numpy as np


def make_palette(num_classes: int) -> np.ndarray:
    """
    From: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/data/pascal/convert_sbdd.py

    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= ((label >> 0) & 1) << (7 - i)
            palette[k, 1] |= ((label >> 1) & 1) << (7 - i)
            palette[k, 2] |= ((label >> 2) & 1) << (7 - i)
            label >>= 3
            i += 1
    return palette
