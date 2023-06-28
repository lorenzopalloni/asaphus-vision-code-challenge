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
from dataclasses import dataclass


@dataclass
class Patch:
    top_left_corner: tuple[int, int]
    bottom_right_corner: tuple[int, int]


def compute_rectangle_center(
    top_left_corner: tuple[int, int], bottom_right_corner: tuple[int, int]
) -> tuple[int, int]:
    x1, y1 = top_left_corner
    x2, y2 = bottom_right_corner
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def find_patches(img, patch_size=(5, 5)):
    h, w = img.shape
    max_avg_brightness = -np.inf
    patches = []

    # Iterate over the image
    for i in range(h - patch_size[0]):
        for j in range(w - patch_size[1]):
            # Extract patch
            patch = img[i : i + patch_size[0], j : j + patch_size[1]]

            # Calculate average brightness of the patch
            avg_brightness = np.mean(patch)

            # Update max average brightness and store patch center
            if avg_brightness > max_avg_brightness:
                max_avg_brightness = avg_brightness
                patches.append(
                    (
                        (i + patch_size[0] // 2, j + patch_size[1] // 2),
                        max_avg_brightness
                    )
                )

    patches.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in patches[:4]]


# Function to calculate area of quadrilateral given coordinates
def calculate_area(coords):
    # Calculate the vectors of the sides of the quadrilateral
    v1 = np.array(coords[0]) - np.array(coords[1])
    v2 = np.array(coords[0]) - np.array(coords[2])
    v3 = np.array(coords[0]) - np.array(coords[3])

    # Calculate the area of the quadrilateral
    area = 0.5 * np.abs(np.cross(v1, v2) + np.cross(v1, v3))
    return area


def main():
    input_path = "assets/Lenna.png"
    output_path = "output.png"

    img_grayscale = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    patches = find_patches(img=img_grayscale)

    area = calculate_area(patches)
    print("Area of the quadrilateral:", area)

    img_rgb = cv2.cvtColor(img_grayscale, cv2.COLOR_GRAY2BGR)
    # breakpoint()
    cv2.polylines(
        img_rgb,
        [np.array(patches)],
        isClosed=True,
        color=(0, 0, 255),  # red
        thickness=2,  # line width in pixels
    )

    cv2.imwrite(output_path, img_rgb)


if __name__ == "__main__":
    main()
