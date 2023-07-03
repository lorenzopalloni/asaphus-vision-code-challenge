# pylint: disable=missing-function-docstring
from __future__ import annotations

import cv2
import pytest
import numpy as np

from solution import (
    verify_patch_centers,
    verify_pairwise_distances,
    compute_integral_image,
    compute_integral_image_v2,
    find_patch_centers,
    sort_points_clockwise,
    calculate_area,
    draw_red_polygon,
    get_corner_points,
)


def test_verify_pairwise_distances_ij_simple_case_true():
    """
      0 1 2 3 4 5 6 7 8 9
    0 . . . . . . . . . .
    1 . . . . . . . . . .
    2 x x x . . . . . . .
    3 x o x . . . . . . .
    4 x x x . . . . . . .
    5 . . x x x . . . . .
    6 . . x o x . . . . .
    7 . . x x x . . . . .
    8 . . . . . . . . . .
    9 . . . . . . . . . .
    """
    points = np.array([[3, 1], [6, 3]])
    patch_size = (3, 3)
    assert verify_pairwise_distances(
        points, patch_size=patch_size, coordinates_type="ij"
    )


def test_verify_pairwise_distances_ij_simple_case_false():
    """
      0 1 2 3 4 5 6 7 8 9
    0 . . . . . . . . . .
    1 . . . . . . . . . .
    2 . . . . . . . . . .
    3 x x x . . . . . . .
    4 x o x . . . . . . .
    5 x x @ x x . . . . .
    6 . . x o x . . . . .
    7 . . x x x . . . . .
    8 . . . . . . . . . .
    9 . . . . . . . . . .
    """
    points = np.array([[4, 1], [6, 3]])
    patch_size = (3, 3)
    assert not verify_pairwise_distances(
        points, patch_size=patch_size, coordinates_type="ij"
    )


def test_verify_pairwise_distances_ij_complex_case_true():
    """
      0 1 2 3 4 5 6 7 8 9
    0 . . . . . . . . . .
    1 . . . . . . . . . .
    2 x x x x . . . . . .
    3 x o x x . . . . . .
    4 x x x x . . . . . .
    5 . x x x x x . . . .
    6 . x x o x x . . . .
    7 . x x x x x . . . .
    8 . . . . . . . . . .
    9 . . . . . . . . . .
    """
    points = np.array([[3, 1], [6, 3]])
    patch_size = (3, 5)
    assert verify_pairwise_distances(points, patch_size=patch_size)


def test_verify_pairwise_distances_ij_complex_case_false():
    """
      0 1 2 3 4 5 6 7 8 9
    0 . . . . . . . . . .
    1 x x x . . . . . . .
    2 x x x . . . . . . .
    3 x o x . . . . . . .
    4 x x @ x x . . . . .
    5 x x @ x x . . . . .
    6 . . x o x . . . . .
    7 . . x x x . . . . .
    8 . . x x x . . . . .
    9 . . . . . . . . . .
    """
    points = np.array([[3, 1], [6, 3]])
    patch_size = (5, 3)
    assert not verify_pairwise_distances(points, patch_size=patch_size)


def test_verify_pairwise_distances_xy_simple_case_true():
    """
    9 . . . . . . . . . .
    8 . . . . . . . . . .
    7 . . . . . . . . . .
    6 . . . . . . . . . .
    5 . . . . . . . . . .
    4 . . x x x . . . . .
    3 . . x o x . . . . .
    2 . . x x x . . . . .
    1 x x . . . . . . . .
    0 o x . . . . . . . .
      0 1 2 3 4 5 6 7 8 9
    """
    points = np.array([[0, 0], [3, 3]])
    patch_size = (3, 3)
    assert verify_pairwise_distances(
        points, patch_size=patch_size, coordinates_type="xy"
    )


def test_verify_pairwise_distances_xy_simple_case_false():
    """
    9 . . . . . . . . . .
    8 . . . . . . . . . .
    7 . . . . . . . . . .
    6 . . . . . . . . . .
    5 . . . . . . . . . .
    4 . . x x x . . . . .
    3 . . x o x . . . . .
    2 . . x x x . . . . .
    1 x x . . . . . . . .
    0 o x . . . . . . . .
      0 1 2 3 4 5 6 7 8 9
    """
    points = np.array([[1, 1], [3, 3]])
    patch_size = (3, 3)
    assert not verify_pairwise_distances(
        points, patch_size=patch_size, coordinates_type="xy"
    )


def test_verify_pairwise_distances_xy_complex_case_true():
    """
    9 . . . . . . . . . .
    8 . . . . . . . . . .
    7 . . . . . . . . . .
    6 . . . . . . . . . .
    5 . . x x x x x x . .
    4 . . x x x x x x . .
    3 . . x o x x o x . .
    2 . . x x x x x x . .
    1 . . x x x x x x . .
    0 . . . . . . . . . .
      0 1 2 3 4 5 6 7 8 9
    """
    points = np.array([[3, 3], [6, 3]])
    patch_size = (3, 5)
    assert verify_pairwise_distances(
        points, patch_size=patch_size, coordinates_type="xy"
    )


def test_verify_pairwise_distances_xy_complex_case_false():
    """
    9 . . . . . . . . . .
    8 . . . . . . . . . .
    7 . . . . . . . . . .
    6 . . . . . . . . . .
    5 . . . . . . . . . .
    4 . x x x @ @ x x x .
    3 . x x o @ @ o x x .
    2 . x x x @ @ x x x .
    1 . . . . . . . . . .
    0 . . . . . . . . . .
      0 1 2 3 4 5 6 7 8 9
    """
    points = np.array([[3, 3], [6, 3]])
    patch_size = (5, 3)
    assert not verify_pairwise_distances(
        points, patch_size=patch_size, coordinates_type="xy"
    )


def test_sort_points_clockwise_single_point():
    points = np.array([[1, 1]])
    expected = np.array([[1, 1]])
    actual = sort_points_clockwise(points, "xy")
    assert np.array_equal(actual, expected)


def test_sort_points_clockwise_two_points():
    points = np.array([[1, 1], [2, 2]])
    expected = np.array([[2, 2], [1, 1]])
    actual = sort_points_clockwise(points, "xy")
    assert np.array_equal(actual, expected)


def test_sort_points_clockwise_same_line():
    points = np.array([[1, 1], [2, 2], [3, 3]])
    expected = np.array([[3, 3], [2, 2], [1, 1]])
    actual = sort_points_clockwise(points, "xy")
    assert np.array_equal(actual, expected)


def test_sort_points_clockwise_invalid_coordinates_type():
    points = np.array([[1, 1], [2, 2], [3, 3]])
    with pytest.raises(ValueError):
        sort_points_clockwise(points, "abc")


def test_sort_points_clockwise_on_axes():
    """
    We start with four points located on the axes. The order of points is as
    follows:

    2-------4      (0,5)------(5,0)
    |       |      |          |
    |       |  =>  |          |
    1-------3      (0,0)------(5,0)

    After applying the function, we expect the points to be sorted in a
    clockwise order, starting from the top left:

    1-------2      (0,5)------(5,0)
    |       |      |          |
    |       |  =>  |          |
    4-------3      (0,0)------(5,0)
    """
    points = np.array([[0, 0], [0, 5], [5, 0], [5, 5]])
    expected = np.array([[0, 5], [5, 5], [5, 0], [0, 0]])
    actual = sort_points_clockwise(points, "xy")
    assert np.array_equal(actual, expected)


def test_sort_points_clockwise_reversed():
    """
    1-------2      (0,5)------(5,5)
    |       |      |          |
    |       |  =>  |          |
    4-------3      (0,0)------(5,0)

    After applying the function, we expect the points to be sorted in a
    clockwise order, starting from the top left:

    2-------1      (0,5)------(5,5)
    |       |      |          |
    |       |  =>  |          |
    3-------4      (0,0)------(5,0)
    """
    points = np.array([[5, 5], [0, 5], [5, 0], [0, 0]])
    expected = np.array([[0, 5], [5, 5], [5, 0], [0, 0]])
    actual = sort_points_clockwise(points, "xy")
    assert np.array_equal(actual, expected)


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
    actual = sort_points_clockwise(points, "xy")
    assert np.array_equal(actual, expected)


def test_sort_points_clockwise_yx_axis_order():
    """
    4-------3      (0,0)-------(0,1)
    |       |      |            |
    |       |  =>  |            |
    |       |      |            |
    1-------2      (1,0)--------(1,1)

    After applying the function, we expect the points to be sorted in a
    clockwise order, starting from the top left:

    1-------2      (0,0)-------(0,1)
    |       |      |            |
    |       |  =>  |            |
    |       |      |            |
    4-------3      (1,0)--------(1,1)
    """
    points = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    expected = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    actual = sort_points_clockwise(points, "ij")
    assert np.array_equal(actual, expected)


def test_calculate_area_rectangle():
    points = np.array([[0, 0], [2, 0], [2, 3], [0, 3]])
    sorted_points = sort_points_clockwise(points, coordinates_type="xy")
    expected = 6.0
    actual = calculate_area(sorted_points)
    assert actual == pytest.approx(expected)


def test_calculate_area_quadrilateral():
    points = np.array([[2, 3], [0, 0], [5, 0], [5, 10]])
    sorted_points = sort_points_clockwise(points, coordinates_type="xy")
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

    assert np.array_equal(img_rgb_copy, img_rgb)


@pytest.mark.parametrize(
    "img_h,img_w,patch_size",
    [
        (10, 10, (3, 3)),
        (29, 29, (3, 3)),
        (10, 10, (5, 5)),
        (29, 29, (5, 5)),
        (127, 131, (5, 5)),
    ],
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
    another_opinion = compute_integral_image_v2(img, patch_size=patch_size)
    assert np.array_equal(integral_image, another_opinion)

    assert np.array_equal(integral_image[:, 0], np.zeros(img_h))
    assert np.array_equal(integral_image[:, -1], np.zeros(img_h))
    assert np.array_equal(integral_image[0, :], np.zeros(img_w))
    assert np.array_equal(integral_image[-1, :], np.zeros(img_w))

    patch_h, patch_w = patch_size
    radius_h, radius_w = patch_h // 2, patch_w // 2
    expected = np.ones((img_h - 2 * radius_h, img_w - 2 * radius_w)) * (
        patch_h * patch_w
    )
    actual = integral_image[
        radius_h : img_h - radius_h, radius_w : img_w - radius_w
    ]
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "img_h,img_w,patch_size",
    [
        (10, 10, (3, 3)),
        (29, 29, (3, 3)),
        (10, 10, (5, 5)),
        (29, 29, (5, 5)),
        (127, 131, (5, 5)),
    ],
)
def test_find_patch_centers_one_bright_region(img_h, img_w, patch_size):
    img = np.ones((img_h, img_w), dtype=np.uint8)
    radius_h, radius_w = patch_size[0] // 2, patch_size[1] // 2
    img[
        img_h // 2 - radius_h : img_h // 2 + radius_h + 1,
        img_w // 2 - radius_w : img_w // 2 + radius_w + 1,
    ] = 255
    patch_centers = find_patch_centers(img, patch_size=patch_size)
    assert [img_h // 2, img_w // 2] in patch_centers


@pytest.mark.parametrize(
    "img_h,img_w,patch_size",
    [
        (10, 10, (3, 3)),
        (29, 29, (3, 3)),
        (10, 10, (5, 5)),
        (29, 29, (5, 5)),
        (127, 131, (5, 5)),
    ],
)
def test_find_patch_centers_adjacent_bright_regions(img_h, img_w, patch_size):
    img = np.zeros((img_h, img_w), dtype=np.uint8)
    patch_h, patch_w = patch_size
    radius_h, radius_w = patch_h // 2, patch_w // 2
    img[
        img_h // 2 - radius_h : img_h // 2 + radius_h + 1,
        img_w // 2 - patch_w : img_w // 2,
    ] = 255
    img[
        img_h // 2 - radius_h : img_h // 2 + radius_h + 1,
        img_w // 2 - patch_w : img_w // 2 + patch_w,
    ] = 255
    patch_centers = find_patch_centers(img, patch_size=patch_size)
    assert [img_h // 2, img_w - radius_w] in patch_centers
    assert [img_h // 2, img_w + radius_w] in patch_centers


@pytest.mark.parametrize("patch_size", [(3, 3), (5, 5)])
def test_find_patch_centers_bright_corners_v1(patch_size):
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
def test_find_patch_centers_bright_corners_v2(patch_size):
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
