"""
Write a Python script that reads an image from a file as grayscale, and finds
the four non-overlapping 5x5 patches with highest average brightness. Take
the patch centers as corners of a quadrilateral, calculate its area in
pixels, and draw the quadrilateral in red into the image and save it in PNG
format. Use the opencv-python package for image handling. Write test cases.
"""
from __future__ import annotations

import heapq
from functools import partial
from typing import Any

import cv2
import numpy as np

DEBUG = False

draw_red_polygon = partial(cv2.polylines, isClosed=True, color=(0, 0, 255))

def compute_rectangle_center(
    top_left_corner: tuple[int, int], bottom_right_corner: tuple[int, int]
) -> tuple[int, int]:
    x1, y1 = top_left_corner
    x2, y2 = bottom_right_corner
    return ((x1 + x2) // 2, (y1 + y2) // 2)

class LimitedHeap():
    def __init__(self, maxlen: int = 4):
        self.heap = []
        self.maxlen = maxlen

    def push(self, item: tuple[float, Any]):
        if len(self.heap) < self.maxlen:
            heapq.heappush(self.heap, item)
        else:
            smallest = self.heap[0]
            if item[0] > smallest[0]:
                heapq.heapreplace(self.heap, item)

    def get_elements(self) -> list[tuple[int, Any]]:
        return [heapq.heappop(self.heap) for _ in range(len(self.heap))][::-1]

def sort_points_clockwise(points: np.ndarray) -> np.ndarray:
    """Sort points in clockwise order."""
    center = np.mean(points, axis=0)
    deltas = points - center
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    sorted_indices = np.argsort(angles)[::-1]
    return points[sorted_indices]


def find_patches(
    img: np.ndarray, patch_size: tuple[int, int] = (5, 5)
) -> np.ndarray:
    img_h, img_w = img.shape
    limited_heap = LimitedHeap()
    for i in range(patch_size[0] // 2, img_h - patch_size[0] // 2, patch_size[0]):
        for j in range(patch_size[1] // 2, img_w - patch_size[1] // 2, patch_size[1]):
            ii_start = i - patch_size[0] // 2
            ii_end = i + patch_size[0] // 2
            jj_start = j - patch_size[1] // 2
            jj_end = j + patch_size[1] // 2
            patch = img[ii_start: ii_end, jj_start: jj_end]

            avg_brightness = np.mean(patch)
            patch_center = (i, j)
            limited_heap.push((avg_brightness, patch_center))
    patches = np.array([patch[1] for patch in limited_heap.get_elements()])
    return sort_points_clockwise(patches)


def calculate_area(points: np.ndarray) -> float:
    """Calculate the area of a polygon given its vertices."""
    points = sort_points_clockwise(points)
    points = np.append(points, [points[0]], axis=0)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def main():
    input_path = "assets/Lenna.png"
    output_path = "output.png"

    img_grayscale = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    patches = find_patches(img=img_grayscale)

    area = calculate_area(patches)
    print("Area of the quadrilateral:", area)

    img_rgb = cv2.cvtColor(img_grayscale, cv2.COLOR_GRAY2BGR)

    draw_red_polygon(img_rgb, [patches])
    # cv2.polylines(
    #     img_rgb,
    #     [np.array(patches)],
    #     isClosed=True,
    #     color=(0, 0, 255),  # red
    #     thickness=1,  # line width in pixels
    # )

    cv2.imwrite(output_path, img_rgb)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        DEBUG = True

    # main()

    img = np.zeros((12, 12), dtype=np.uint8)
    img[1, 1] = 255
    img[1, 10] = 255
    img[10, 10] = 255
    img[10, 1] = 255
    patches = find_patches(img, patch_size=(3, 3))


    def draw_solution(
        img_grayscale: np.ndarray,
        patches: np.ndarray,
        figsize: tuple[float, float] = (2, 1),
    ) -> plt.Figure:
        assert len(patches) == 4, f"{len(patches)=}"

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        breakpoint()
        draw_red_polygon(img_rgb, [patches])
        breakpoint()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.imshow(img, cmap="gray")
        ax2.imshow(img_rgb[:, :, ::-1])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        return fig

    import matplotlib.pyplot as plt
    img_solution = draw_solution(img, patches)

    plt.show()
    print(patches)

