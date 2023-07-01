# pylint: disable=missing-function-docstring
from __future__ import annotations

import cv2
import pytest
import numpy as np

from solution import (
    verify_patch_centers,
    verify_pairwise_distances,
    pad_with_edge_and_zero_top_left,
    compute_local_integral,
    find_patch_centers,
    sort_points_clockwise,
    calculate_area,
    draw_red_polygon,
    get_corner_points,
    # display_side_by_side,
    # display_grayscale_img,
)

def test_verify_pairwise_distances_simple():
    points = np.array([
        [0, 0], [3, 3], [9, 9], [12, 12]
    ])
    patch_size = (3, 3)
    assert verify_pairwise_distances(points, patch_size=patch_size)

def test_verify_pairwise_distances_simple_false_case():
    points = np.array([
        [0, 0], [2, 2], [9, 9], [12, 12]
    ])
    patch_size = (3, 3)
    assert not verify_pairwise_distances(points, patch_size=patch_size)

def test_verify_pairwise_distances_rectangular_patch():
    points = np.array([
        [3, 3], [6, 3]
    ])
    patch_size = (5, 3)
    assert verify_pairwise_distances(points, patch_size=patch_size)

def test_verify_pairwise_distances_rectangular_patch_false_case():
    points = np.array([
        [3, 3], [6, 2]
    ])
    patch_size = (5, 3)
    assert not verify_pairwise_distances(points, patch_size=patch_size)


def test_sort_points_clockwise():
    """
    We start with four points forming a square. The order of points is as
    follows:

    2-------3      (0,1)-------(1,1)
    |       |      |            |
    |       |  =>  |            |
    |       |      |            |
    1-------4      (0,0)--------(1,0)

    After applying the function, we expect the points to be sorted in a
    clockwise order, starting from the top left:

    1-------2      (0,1)-------(1,1)
    |       |      |            |
    |       |  =>  |            |
    |       |      |            |
    4-------3      (0,0)--------(1,0)
    """
    points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    expected = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    actual = sort_points_clockwise(points)
    assert np.array_equal(actual, expected)


def test_calculate_area_rectangle():
    points = np.array([[0, 0], [2, 0], [2, 3], [0, 3]])
    sorted_points = sort_points_clockwise(points)
    expected = 6.0
    actual = calculate_area(sorted_points)
    assert actual == pytest.approx(expected)


def test_calculate_area_quadrilateral():
    points = np.array([[2, 3], [0, 0], [5, 0], [5, 10]])
    sorted_points = sort_points_clockwise(points)
    expected = 22.5
    actual = calculate_area(sorted_points)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("patch_size", [(3, 3), (5, 5)])
def test_draw_red_polygon(patch_size):
    img_h, img_w = (4 * patch_size[0], 4 * patch_size[1])
    img_grayscale = np.zeros((img_h, img_w), dtype=np.uint8)
    corner_points = get_corner_points(img_grayscale, patch_size=patch_size)
    sorted_corner_points = sort_points_clockwise(corner_points)

    for i, j in corner_points:
        img_grayscale[i, j] = 255

    img_rgb = cv2.cvtColor(img_grayscale, cv2.COLOR_GRAY2BGR)
    img_rgb_copy = img_rgb.copy()
    draw_red_polygon(img_rgb, sorted_corner_points)

    red_color = (0, 0, 255)
    img_rgb_copy[
        corner_points[0][0], corner_points[0][1] : corner_points[1][1] + 1
    ] = red_color
    img_rgb_copy[
        corner_points[3][0], corner_points[3][1] : corner_points[2][1] + 1
    ] = red_color

    img_rgb_copy[
        corner_points[0][0] : corner_points[3][0] + 1, corner_points[0][1]
    ] = red_color
    img_rgb_copy[
        corner_points[1][0] : corner_points[2][0] + 1, corner_points[1][1]
    ] = red_color

    # display_side_by_side(img_rgb_copy, img_rgb)
    assert np.array_equal(img_rgb_copy, img_rgb)

def test_compute_local_integral_3x3_ones():
    """
    1 1 1      4 6 4
    1 1 1  ->  6 9 6
    1 1 1      4 6 4
    """
    img = np.ones((3, 3), dtype=np.uint8)
    result = compute_local_integral(img, patch_size=(3, 3))

    assert result[1, 1] == pytest.approx(9)

    assert result[0, 0] == pytest.approx(4)
    assert result[0, 2] == pytest.approx(4)
    assert result[2, 0] == pytest.approx(4)
    assert result[2, 2] == pytest.approx(4)

    assert result[0, 1] == pytest.approx(6)
    assert result[1, 0] == pytest.approx(6)
    assert result[2, 1] == pytest.approx(6)
    assert result[1, 2] == pytest.approx(6)


def test_custom_pad():
    arr = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]], dtype=np.float64)
    pad_width = (2, 2)
    actual = pad_with_edge_and_zero_top_left(arr, pad_width=pad_width)
    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 3, 3],
            [0, 0, 2, 4, 6, 6, 6],
            [0, 0, 3, 6, 9, 9, 9],
            [0, 0, 3, 6, 9, 9, 9],
            [0, 0, 3, 6, 9, 9, 9],
        ],
        dtype=np.float64,
    )
    assert np.array_equal(expected, actual)


def test_no_high_brightness_patches():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:10, 5:10] = 50
    patch_centers = find_patch_centers(img, patch_size=(5, 5))
    expected = np.array([[0, 10], [7, 7], [0, 0], [0, 5]])
    actual = sort_points_clockwise(patch_centers)
    assert np.array_equal(actual, expected)


def test_adjacent_high_brightness_patches():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:10, 5:10] = 255
    img[10:15, 10:15] = 255
    patch_centers = find_patch_centers(img, patch_size=(5, 5))
    expected = np.array([[12, 12], [7, 7], [0, 0], [0, 5]])
    actual = sort_points_clockwise(patch_centers)
    assert np.array_equal(actual, expected)


def test_high_brightness_border():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:5, :] = 255
    patch_centers = find_patch_centers(img, patch_size=(5, 5))
    expected = np.array([[2, 17], [2, 12], [2, 7], [2, 2]])
    actual = sort_points_clockwise(patch_centers)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("patch_size", [(3, 3), (5, 5)])
def test_find_patches_at_corners_v1(patch_size):
    img_h, img_w = (4 * patch_size[0], 4 * patch_size[1])
    img = np.zeros((img_h, img_w), dtype=np.uint8)
    corner_points = get_corner_points(img, patch_size=patch_size)

    for i, j in corner_points:
        img[i, j] = 255

    patch_centers = find_patch_centers(img, patch_size=patch_size)
    sorted_patch_centers = sort_points_clockwise(patch_centers)
    assert len(sorted_patch_centers) == 4, f"{len(sorted_patch_centers)=}"
    assert verify_patch_centers(
        patch_centers=sorted_patch_centers, img=img, patch_size=patch_size
    )

    area = calculate_area(sorted_patch_centers)
    assert area == pytest.approx(
        (img_h - patch_size[0]) * (img_w - patch_size[1])
    )


@pytest.mark.parametrize("patch_size", [(3, 3), (5, 5)])
def test_find_patches_at_corners_v2(patch_size):
    img_h, img_w = (4 * patch_size[0], 4 * patch_size[1])
    img = np.zeros((img_h, img_w), dtype=np.uint8)
    selected_points = [
        [0, 0],
        [0, img_w - 1],
        [img_h - 1, 0],
        [img_w - 1, img_h - 1],
    ]

    for i, j in selected_points:
        img[i, j] = 255

    patch_centers = find_patch_centers(img, patch_size=patch_size)
    sorted_patch_centers = sort_points_clockwise(patch_centers)
    assert len(sorted_patch_centers) == 4
    assert verify_patch_centers(
        patch_centers=sorted_patch_centers, img=img, patch_size=patch_size
    )
    expected = np.array(
        [
            [0, img_w - 1 - patch_size[1] // 2],
            [img_h - 1 - patch_size[0] // 2, img_w - 1 - patch_size[1] // 2],
            [img_h - 1 - patch_size[0] // 2, 0],
            [0, 0],
        ]
    )
    assert np.array_equal(sorted_patch_centers, expected)

    area = calculate_area(sorted_patch_centers)
    max_area = img_h * img_w
    assert 0 <= area <= max_area


if __name__ == "__main__":
    # test_find_patches_at_corners_v1(patch_size=(3, 3))
    # test_find_patches_at_corners_v2(patch_size=(3, 3))
    # test_no_high_brightness_patches()
    # test_draw_red_polygon((5, 5))
    test_verify_pairwise_distances_simple()
