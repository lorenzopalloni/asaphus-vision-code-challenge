# pylint: disable=missing-function-docstring
from __future__ import annotations

import cv2
import pytest
import numpy as np

from solution import (
    verify_patch_centers,
    verify_pairwise_distances,
    compute_integral_image,
    find_patch_centers,
    sort_points_clockwise,
    calculate_area,
    draw_red_polygon,
    get_corner_points,
    # display_side_by_side,
    # display_grayscale_img,
)


def test_verify_pairwise_distances_simple():
    points = np.array([[0, 0], [3, 3], [9, 9], [12, 12]])
    patch_size = (3, 3)
    assert verify_pairwise_distances(points, patch_size=patch_size)


def test_verify_pairwise_distances_simple_false_case():
    points = np.array([[0, 0], [2, 2], [9, 9], [12, 12]])
    patch_size = (3, 3)
    assert not verify_pairwise_distances(points, patch_size=patch_size)


def test_verify_pairwise_distances_rectangular_patch():
    points = np.array([[3, 3], [6, 3]])
    patch_size = (5, 3)
    assert verify_pairwise_distances(points, patch_size=patch_size)


def test_verify_pairwise_distances_rectangular_patch_false_case():
    points = np.array([[3, 3], [6, 2]])
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


@pytest.mark.parametrize(
    "img_h,img_w,patch_size",
    [(10, 10, (3, 3)), (29, 29, (3, 3)), (10, 10, (5, 5)), (29, 29, (5, 5))],
)
def test_compute_integral_image_ones(img_h, img_w, patch_size):
    """
    An example with img_h == img_w == 5, and patch_size == 3:
        1 1 1 1 1      0 0 0 0 0
        1 1 1 1 1      0 9 9 9 0
        1 1 1 1 1  ->  0 9 9 9 0
        1 1 1 1 1      0 9 9 9 0
        1 1 1 1 1      0 0 0 0 0
    """
    img = np.ones((img_h, img_w), dtype=np.uint8)
    integral_image = compute_integral_image(img, patch_size=patch_size)

    assert np.array_equal(integral_image[:, 0], np.zeros(img_h))
    assert np.array_equal(integral_image[:, -1], np.zeros(img_h))
    assert np.array_equal(integral_image[0, :], np.zeros(img_w))
    assert np.array_equal(integral_image[-1, :], np.zeros(img_w))

    pi, pj = patch_size[0], patch_size[1]
    ri, rj = pi // 2, pj // 2
    expected = np.ones((img_h - 2 * ri, img_w - 2 * rj)) * (pi * pj)
    actual = integral_image[ri : img_h - ri, rj : img_w - rj]
    assert np.array_equal(actual, expected)


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
    assert len(sorted_patch_centers) == 4
    assert verify_patch_centers(
        patch_centers=sorted_patch_centers, img=img, patch_size=patch_size
    )

    pi, pj = patch_size
    ri, rj = pi // 2, pj // 2
    expected = sort_points_clockwise(
        np.array(
            [
                [ri, rj],
                [img_h - 1 - ri - ri, img_w - 1 - rj - rj],
                [img_h - 1 - ri - ri, rj],
                [ri, img_w - 1 - rj - rj],
            ]
        )
    )
    assert np.array_equal(sorted_patch_centers, expected)


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
    corner_points = get_corner_points(img, patch_size=patch_size)
    sorted_corner_points = sort_points_clockwise(corner_points)
    assert np.array_equal(sorted_patch_centers, sorted_corner_points)


if __name__ == "__main__":
    test_find_patches_at_corners_v1((5, 5))
