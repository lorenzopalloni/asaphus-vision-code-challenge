"""
Asaphus Vision Code Challenge

This is a solution of the challenge written by Lorenzo Palloni.
email: palloni.lorenzo@gmail.com

Description of the challenge:
Write a Python script that reads an image from a file as grayscale, and finds
the four non-overlapping 5x5 patches with highest average brightness. Take the
patch centers as corners of a quadrilateral, calculate its area in pixels, and
draw the quadrilateral in red into the image and save it in PNG format. Use
the opencv-python package for image handling. Write test cases.

Usage:
```sh
solution.py [-h] \
    --input_image_path INPUT_IMAGE_PATH \
    [--output_image_path OUTPUT_IMAGE_PATH]
```
"""
from __future__ import annotations

from pathlib import Path
import argparse

import cv2
import numpy as np

# import matplotlib.pyplot as plt
from numpy.typing import NDArray



# def display_grayscale_img(img: NDArray[np.uint8], figsize=(5, 5)):
#     """Display a grayscale image using Matplotlib."""
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
#     ax.set_xticks([])
#     ax.set_yticks([])
#     return fig

def verify_pairwise_distances(
    points: NDArray[np.int64], patch_size: tuple[int, int]
) -> bool:
    min_point = points.min(axis=0)
    rescaled_points = points - min_point
    max_rescaled_point = rescaled_points.max(axis=0)
    region = np.ones(max_rescaled_point)
    ri, rj = patch_size[0] // 2, patch_size[1] // 2
    for i, j in rescaled_points:
        region[max(0, i - ri): i + ri + 1, max(0, j - rj): j + rj + 1] -= 1
        if np.sum(region[max(0, i - ri): i + ri, max(0, j - rj): j + rj]) < 0:
            return False
    return True

def verify_patch_centers(
    patch_centers: NDArray[np.int64],
    img: NDArray[np.uint8],
    patch_size: tuple[int, int] = (5, 5),
) -> bool:
    """
    Determine if patches extracted from an image are valid.

    This function first calculates a local integral image with the provided
    patch size. Then, for each patch, the function sets all the pixel values
    within the patch in the image to zero and records the value from the local
    integral image. The minimum value among all patches' integral values is
    then compared to the maximum pixel value in the image (after zeroing all
    patches). If the minimum patch value is greater than or equal to the
    maximum pixel value, the function returns True indicating all patches are
    valid; otherwise, it returns False.

    Args:
        patch_centers (NDArray[np.int64]): Numpy array containing the patch
            centers to be validated.
        img (NDArray[np.uint8]): Image from which the patches are extracted.
        patch_size (tuple[int, int], optional): Size of the patches.
            Defaults to (5, 5).

    Returns:
        bool: True if all patches are valid, False otherwise.
    """
    local_integral = compute_local_integral(img=img, patch_size=patch_size)
    img_copy = img.copy()
    patch_values = np.zeros(len(patch_centers), dtype=np.float64)
    for patch_id, (i, j) in enumerate(patch_centers):
        patch_values[patch_id] = local_integral[i, j]
        img_copy[
            max(0, i - (patch_size[0] - 1)) : i + patch_size[0],
            max(0, j - (patch_size[1] + 1)) : j + patch_size[1],
        ] = 0.0
    return patch_values.min() >= img_copy.max()


def get_corner_points(
    img: NDArray[np.uint8], patch_size: tuple[int, int] = (5, 5)
) -> NDArray[np.int64]:
    """Return corner points up to the radiuses of the `patch_size`."""
    img_h, img_w = img.shape[:2]
    radiuses = (patch_size[0] // 2, patch_size[1] // 2)
    return np.array(
        [
            [radiuses[0], radiuses[1]],
            [radiuses[0], (img_w - 1) - radiuses[1]],
            [(img_h - 1) - radiuses[0], (img_w - 1) - radiuses[1]],
            [(img_h - 1) - radiuses[0], radiuses[1]],
        ]
    )


def draw_red_polygon(
    img: NDArray[np.uint8], vertices: NDArray[np.int64]
) -> NDArray[np.uint8]:
    """Draw a polygon on image joining vertices."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(
        img,
        [np.flip(vertices, 1)],
        isClosed=True,
        color=(0, 0, 255),
        thickness=1,
    )
    return img


def sort_points_clockwise(points: NDArray[np.int64]) -> NDArray[np.int64]:
    """Sort points in clockwise order."""
    center = np.mean(points, axis=0)
    deltas = points - center
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    sorted_indices = np.argsort(angles)[::-1]
    return points[sorted_indices]


def calculate_area(sorted_vertices: NDArray[np.int64]) -> float:
    """Calculate the area of a polygon given its vertices.

    References:
    https://en.wikipedia.org/wiki/Shoelace_formula
    https://alexkritchevsky.com/2018/08/06/oriented-area.html
    https://rosettacode.org/wiki/Shoelace_formula_for_polygonal_area#Python
    https://scikit-spatial.readthedocs.io/en/stable/_modules/skspatial/measurement.html#area_signed
    """
    if sorted_vertices.ndim != 2:
        raise ValueError("The points must be 2D.")

    if len(sorted_vertices) < 3:
        raise ValueError("There must be at least 3 points.")

    sorted_vertices = np.append(sorted_vertices, [sorted_vertices[0]], axis=0)
    x = sorted_vertices[:, 0]
    y = sorted_vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def display_side_by_side(
    image1: NDArray[np.uint8],
    image2: NDArray[np.uint8],
    title: str = "Combined Image",
):
    """Display two images side-by-side using opencv-python."""
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image1 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((image1, image2))

    cv2.imshow(title, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def pad_with_edge_and_zero_top_left(
    arr: NDArray[np.float64], pad_width: tuple[int, int]
) -> NDArray[np.float64]:
    """
    Pad an arr with edge values and then zero out the top rows and left cols.

    Example:
        Given an input array:
            [
                [1, 2],
                [3, 4]
            ]
        and pad_width = (1, 1), the function will first pad the array with
        edge values:

            [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ]

    and then zero out the top row and the left column:

        [
            [0, 0, 0, 0],
            [0, 1, 2, 2],
            [0, 3, 4, 4],
            [0, 3, 4, 4]
        ]
    """
    padded_arr = np.pad(arr, pad_width=pad_width, mode="edge")
    padded_arr[: pad_width[0], :] = 0.0
    padded_arr[:, : pad_width[1]] = 0.0
    return padded_arr


def compute_local_integral(
    img: NDArray[np.uint8],
    patch_size: tuple[int, int] = (5, 5),
) -> NDArray[np.float64]:
    """Compute local integral of a given 2D numpy array."""

    img_h, img_w = img.shape

    kernel_size = (patch_size[0] + 2, patch_size[1] + 2)
    kernel = np.zeros(kernel_size, dtype=np.float64)
    kernel[0, 0] = 1
    kernel[0, -2] = -1
    kernel[-2, 0] = -1
    kernel[-2, -2] = 1

    # store radiuses
    ri, rj = kernel_size[0] // 2, kernel_size[1] // 2

    arr = img.astype(np.float64)
    np.cumsum(arr, axis=0, out=arr)
    np.cumsum(arr, axis=1, out=arr)
    arr = pad_with_edge_and_zero_top_left(arr, pad_width=(ri, rj))

    local_integral = np.zeros((img_h, img_w), dtype=np.float64)

    for i in range(img_h):
        for j in range(img_w):
            # store local_region's center
            ci, cj = i + ri, j + rj
            local_region = arr[ci - ri: ci + ri + 1, cj - rj: cj + rj + 1]
            local_integral[i, j] = np.multiply(local_region, kernel).sum()

    return local_integral


def find_patch_centers(
    img: NDArray[np.uint8],
    patch_size: tuple[int, int] = (5, 5),
    num_patches: int = 4,
) -> NDArray[np.int64]:
    """Find the non-overlapping patches with highest average brightness."""
    local_integral = compute_local_integral(img=img, patch_size=patch_size)

    indices = []
    arr_copy = local_integral.copy()

    for _ in range(num_patches):
        max_val_index = np.unravel_index(np.argmax(arr_copy), arr_copy.shape)
        indices.append(max_val_index)

        i, j = max_val_index[0], max_val_index[1]
        arr_copy[
            max(0, i - (patch_size[0] - 1)) : i + patch_size[0],
            max(0, j - (patch_size[1] + 1)) : j + patch_size[1],
        ] = -np.inf

    return np.array(indices)


def parse_arguments():
    """Parse script's input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_image_path",
        "-i",
        type=Path,
        required=True,
        help="The path to the input image file.",
    )
    parser.add_argument(
        "--output_image_path",
        "-o",
        type=Path,
        default="output.png",
        help="The path to the output image file.",
    )
    args = parser.parse_args()
    return args


def save_image(img: NDArray[np.uint8], output_image_path: Path):
    """Save the image to a file, ensuring the file is a PNG."""
    if output_image_path.suffix.lower() != ".png":
        raise ValueError("Extension of the output_image_path must be 'PNG'.")
    cv2.imwrite(filename=output_image_path.as_posix(), img=img)


def main():
    """Main that performs the following sequence of operations:
        1. Reads an input image in grayscale.
        2. Finds four non-overlapping 5x5 patches within the image that have
            the highest average brightness.
        3. Takes the patch centers as the corners of a quadrilateral.
        4. Calculates the area of this quadrilateral (in pixels).
        5. Draws the quadrilateral in red onto the original image.
        6. Saves the modified image in PNG format to a specified output path.

    The script also displays both the original and modified images
        side-by-side, and prints the calculated area to the console.
    """
    args = parse_arguments()
    input_image_path = args.input_image_path
    output_image_path = args.output_image_path

    img_grayscale = cv2.imread(
        input_image_path.as_posix(), cv2.IMREAD_GRAYSCALE
    )
    patch_centers = find_patch_centers(img=img_grayscale)
    verify_pairwise_distances(patch_centers, (5, 5))
    sorted_patch_centers = sort_points_clockwise(patch_centers)
    area = calculate_area(sorted_patch_centers)
    print("Area of the quadrilateral:", area)

    img_rgb = draw_red_polygon(img_grayscale, sorted_patch_centers)
    display_side_by_side(img_grayscale, img_rgb, title=f"Area: {area}")
    save_image(img=img_rgb, output_image_path=output_image_path)


if __name__ == "__main__":
    main()
