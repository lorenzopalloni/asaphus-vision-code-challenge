import pytest
import numpy as np

from solution import (
    compute_rectangle_center,
    find_patches,
    sort_points_clockwise,
    calculate_area,
    LimitedHeap,
)

def test_compute_rectangle_center():
    top_left_corner = (0, 0)
    bottom_right_corner = (30, 30)
    actual_center = compute_rectangle_center(top_left_corner, bottom_right_corner)
    expected_center = (15, 15)
    assert actual_center[0] == expected_center[0]
    assert actual_center[1] == expected_center[1]

    top_left_corner = (1, 5)
    bottom_right_corner = (10, 8)
    actual_center = compute_rectangle_center(top_left_corner, bottom_right_corner)
    expected_center = (5, 6)
    assert actual_center[0] == expected_center[0]
    assert actual_center[1] == expected_center[1]

def test_no_high_brightness_patches():
    img = np.zeros((20, 20), dtype=np.uint8)  # A 20x20 black image
    img[5:10, 5:10] = 50  # Add a patch with moderate brightness
    patches = find_patches(img)
    assert max(np.mean(img[i-2:i+3, j-2:j+3]) for i, j in patches) == 50


def test_adjacent_high_brightness_patches():
    img = np.zeros((20, 20), dtype=np.uint8)  # A 20x20 black image
    img[5:10, 5:10] = 255  # Add a bright patch
    img[10:15, 10:15] = 255  # Add an adjacent bright patch
    patches = find_patches(img)
    assert len(patches) >= 2


def test_high_brightness_border():
    img = np.zeros((20, 20), dtype=np.uint8)  # A 20x20 black image
    img[:5, :] = 255  # Add a bright border
    patches = find_patches(img)
    assert max(np.mean(img[i-2:i+3, j-2:j+3]) for i, j in patches) == 255

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
    result = sort_points_clockwise(points)
    assert np.array_equal(result, expected)

def test_calculate_area_square():
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    expected = 1.0
    result = calculate_area(points)
    assert result == pytest.approx(expected)

def test_calculate_area_rectangle():
    points = np.array([[0, 0], [2, 0], [2, 3], [0, 3]])
    expected = 6.0
    result = calculate_area(points)
    assert result == pytest.approx(expected)

def test_calculate_area_quadrilateral():
    points = np.array([[2, 3], [0, 0], [5, 0], [5, 10]])
    expected = 22.5
    result = calculate_area(points)
    assert result == pytest.approx(expected)

def test_limited_heap():
    heap = LimitedHeap(maxlen=4)
    heap.push((1, [0, 0]))
    heap.push((3, [4, 4]))
    heap.push((2, [3, 3]))
    heap.push((2, [4, 4]))
    heap.push((5, [1, 1]))
    heap.push((0, [2, 2]))

    # Check that we only have the 4 largest items, and that we get them in descending order.
    assert heap.get_elements() == [(5, [1, 1]), (3, [4, 4]), (2, [4, 4]), (2, [3, 3])]

