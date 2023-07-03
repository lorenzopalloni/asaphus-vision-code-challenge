"""
This module contains a function to compute the local integral image of a given
image.
"""
import numpy as np
from numpy.typing import NDArray


def compute_local_integral_image(
    global_integral_image: NDArray[np.float64],
    img_h: int,
    img_w: int,
    radius_h: int,
    radius_w: int,
) -> NDArray[np.float64]:
    """
    Compute the local integral image for a given global integral image.

    This function calculates the local integral image by summing the pixel
    values within a patch defined by radius_h and radius_w for each pixel
    in the image.

    Args:
        global_integral_image (NDArray[np.float64]): The global integral
            image of the original image.
        img_h (int): The height of the image.
        img_w (int): The width of the image.
        radius_h (int): The vertical radius of the patch.
        radius_w (int): The horizontal radius of the patch.

    Returns:
        NDArray[np.float64]: The local integral image.
    """
    local_integral_image = np.zeros((img_h, img_w), dtype=np.float64)
    for i in range(radius_h, img_h - radius_h):
        for j in range(radius_w, img_w - radius_w):
            local_integral_image[i, j] = (
                global_integral_image[i + radius_h + 1, j + radius_w + 1]
                - global_integral_image[i - radius_h, j + radius_w + 1]
                - global_integral_image[i + radius_h + 1, j - radius_w]
                + global_integral_image[i - radius_h, j - radius_w]
            )
    return local_integral_image
