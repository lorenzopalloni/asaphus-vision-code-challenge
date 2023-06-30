# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,invalid-name
"""
Write a Python script that reads an image from a file as grayscale, and finds
the four non-overlapping 5x5 patches with highest average brightness. Take
the patch centers as corners of a quadrilateral, calculate its area in
pixels, and draw the quadrilateral in red into the image and save it in PNG
format. Use the opencv-python package for image handling. Write test cases.
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def display_grayscale_img(img: NDArray[np.uint8], figsize=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def draw_red_polygon(img, vertices):
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


def sort_points_clockwise(points: NDArray) -> NDArray:
    """Sort points in clockwise order."""
    center = np.mean(points, axis=0)
    deltas = points - center
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    sorted_indices = np.argsort(angles)[::-1]
    return points[sorted_indices]


def calculate_area(points: NDArray) -> float:
    """Calculate the area of a polygon given its vertices."""
    points = sort_points_clockwise(points)
    points = np.append(points, [points[0]], axis=0)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_corner_points(img: NDArray, patch_size: tuple[int, int] = (5, 5)) -> NDArray:
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


def display_side_by_side(
    image1: NDArray, image2: NDArray, title: str = "Combined Image"
):
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((image1, image2))

    cv2.imshow(title, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def custom_pad(
    arr: NDArray[np.float64], pad_width: tuple[int, int]
) -> NDArray[np.float64]:
    padded_arr = np.pad(arr, pad_width=pad_width, mode="edge")
    padded_arr[: pad_width[0], :] = 0.0
    padded_arr[:, : pad_width[1]] = 0.0
    return padded_arr


def compute_local_integral(
    img: NDArray[np.uint8],
    patch_size: tuple[int, int] = (5, 5),
) -> NDArray:
    img_h, img_w = img.shape
    arr = img.astype(np.float64)

    np.cumsum(arr, axis=0, out=arr)
    np.cumsum(arr, axis=1, out=arr)

    kernel_size = (patch_size[0] + 2, patch_size[1] + 2)
    kernel = np.zeros(kernel_size, dtype=np.float64)
    kernel[0, 0] = 1
    kernel[0, -2] = -1
    kernel[-2, 0] = -1
    kernel[-2, -2] = 1

    r1, r2 = kernel_size[0] // 2, kernel_size[1] // 2
    padded_arr = custom_pad(arr, pad_width=(r1, r2))

    local_integral = np.zeros((img_h, img_w), dtype=np.float64)

    for i in range(img_h):
        for j in range(img_w):
            icp, jcp = i + r1, j + r2  # center of padded_arr
            padded_region = padded_arr[icp - r1 : icp + r1 + 1, jcp - r2 : jcp + r2 + 1]
            local_integral[i, j] = np.multiply(padded_region, kernel).sum()

    return local_integral


def find_patches(
    img: NDArray,
    patch_size: tuple[int, int] = (5, 5),
) -> NDArray:

    local_integral = compute_local_integral(img, patch_size=patch_size)
    normalized_local_integral = local_integral / (patch_size[0] * patch_size[1])
    normalized_local_integral = normalized_local_integral.astype(np.uint8)

    indices = []
    arr_copy = local_integral.copy()

    for _ in range(4):
        max_val_index = np.unravel_index(np.argmax(arr_copy), arr_copy.shape)
        indices.append(max_val_index)

        i, j = max_val_index[0], max_val_index[1]
        arr_copy[
            max(0, i - (patch_size[0] - 1)) : i + patch_size[0],
            max(0, j - (patch_size[1] + 1)) : j + patch_size[1],
        ] = -np.inf

    return np.array(indices)


def main():
    input_path = "assets/Lenna.png"
    output_path = "output.png"

    img_grayscale = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    patches = find_patches(img=img_grayscale)
    patches = sort_points_clockwise(patches)
    print(patches)
    area = calculate_area(patches)
    print("Area of the quadrilateral:", area)

    img_rgb = draw_red_polygon(img_grayscale, patches)
    display_side_by_side(img_grayscale, img_rgb, title=f"Area: {area}")
    cv2.imwrite(output_path, img_rgb)


if __name__ == "__main__":
    main()
