import numpy as np
from solution import compute_rectangle_center, find_patches

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
    img = np.zeros((20,20), dtype=np.uint8)  # A 20x20 black image
    img[5:10, 5:10] = 50  # Add a patch with moderate brightness
    patches = find_patches(img)
    assert max(np.mean(img[i-2:i+3, j-2:j+3]) for i, j in patches) == 50


def test_adjacent_high_brightness_patches():
    img = np.zeros((20,20), dtype=np.uint8)  # A 20x20 black image
    img[5:10, 5:10] = 255  # Add a bright patch
    img[10:15, 10:15] = 255  # Add an adjacent bright patch
    patches = find_patches(img)
    assert len(patches) >= 2


def test_high_brightness_border():
    img = np.zeros((20,20), dtype=np.uint8)  # A 20x20 black image
    img[:5, :] = 255  # Add a bright border
    patches = find_patches(img)
    assert max(np.mean(img[i-2:i+3, j-2:j+3]) for i, j in patches) == 255

