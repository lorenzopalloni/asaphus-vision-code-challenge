"""
Asaphus Vision Code Challenge

Author: Lorenzo Palloni (palloni.lorenzo@gmail.com)

This script reads an image from a file in grayscale and finds the four 
non-overlapping patches with the highest average brightness, with
user-specified patch size. It then draws a quadrilateral using the centers
of these patches as corners and saves the modified image in PNG format.

Required packages: opencv-python, numpy

Usage:
solution.py [-h] \
    --input_image_path INPUT_IMAGE_PATH \
    [--output_image_path OUTPUT_IMAGE_PATH] \
    [--patch_size PATCH_SIZE] \
    [--disable_display]

Arguments:
--input_image_path: The file path of the input image.
--output_image_path (optional): The file path for the output image.
--patch_size (optional): The size of the patches to find bright spots,
    specified as 'height,width'. Default is '5,5'.
--disable_display (optional): Disable display of output image.

If not provided, the script will save the output image in the
current directory with the name 'output.png'.

Output:
The output is a PNG image with a quadrilateral drawn on it, whose corners
are the centers of the patches with the highest average brightness. The area
of the quadrilateral in pixels is printed to the console.
"""


from __future__ import annotations

from pathlib import Path
import argparse

import cv2
import numpy as np

from numpy.typing import NDArray

try:
    from cython_my_utils import compute_local_integral_image
except ImportError:
    print("Could not import optimized version. Using default Python function.")
    from my_utils import compute_local_integral_image


def verify_pairwise_distances(
    points: NDArray[np.int64],
    patch_size: tuple[int, int],
    coordinates_type: str = "ij",
) -> bool:
    """
    Verify if patches centered at the given points don't overlap.

    Function checks if patches of a specified size, when centered at provided
    points on a hypothetical image, would overlap. It respects the order of
    input points based on the specified 'coordinates_type'.

    Args:
        points (NDArray[np.int64]): An array of points in the form of (x, y)
            or (i, j) coordinates.
        patch_size (tuple[int, int]): The size of each patch, as (height,
            width).
        coordinates_type (str, optional): Specifies the type of coordinates
            in the 'points' array. It should be either 'xy' (standard
            cartesian 2D plan) or 'ij' (ith row and jth column of a 2D array).
            Defaults to 'ij'.

    Returns:
        bool: True if all patches centered at the points wouldn't overlap,
            False otherwise.

    Raises:
        ValueError: If 'coordinates_type' is not 'xy' or 'ij'.
    """

    if coordinates_type not in {"xy", "ij"}:
        raise ValueError("coordinates_type should be either 'xy' or 'ij'.")

    if coordinates_type == "xy":
        points = points[:, ::-1]
        patch_size = patch_size[::-1]

    min_point = points.min(axis=0)
    rescaled_points = points - min_point
    max_rescaled_point = rescaled_points.max(axis=0) + 1
    fake_region = np.ones(max_rescaled_point, dtype=np.int64)
    radius_h, radius_w = patch_size[0] // 2, patch_size[1] // 2
    for i, j in rescaled_points:
        local_fake_region = fake_region[
            max(0, i - radius_h) : i + radius_h + 1,
            max(0, j - radius_w) : j + radius_w + 1,
        ]
        local_fake_region -= 1
        if local_fake_region.sum() < 0:
            return False
    return True


def verify_patch_centers(
    patch_centers: NDArray[np.int64],
    img: NDArray[np.uint8],
    patch_size: tuple[int, int] = (5, 5),
) -> bool:
    """
    Determine if patch centers extracted from an image are valid.

    1. Verifies that the given patches do not overlap.
    2. Calculates the integral image of `img`, given a `patch_size`.
    3. For each patch center, the function sets all the pixel values within
       the patch in the image to zero and records the value from the integral
       image.
    4. The minimum value among all patches' integral values is then compared
       to the maximum pixel value in the image (after zeroing all patches).
    5. If the minimum patch value is greater than or equal to the maximum
       pixel value, the function returns True indicating all patches are
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
    if not verify_pairwise_distances(
        points=patch_centers, patch_size=patch_size, coordinates_type="ij"
    ):
        return False
    integral_image = compute_integral_image(img=img, patch_size=patch_size)
    img_h, img_w = img.shape[:2]
    patch_h, patch_w = patch_size
    radius_h, radius_w = patch_h // 2, patch_w // 2
    img_copy = img.copy()
    patch_values = np.zeros(len(patch_centers), dtype=np.float64)
    for patch_id, (i, j) in enumerate(patch_centers):

        if not radius_h <= i <= img_h - 1 - radius_h:
            return False
        if not radius_w <= j <= img_w - 1 - radius_w:
            return False

        patch_values[patch_id] = integral_image[i, j]
        img_copy[
            max(0, i - (patch_h - 1)) : i + (patch_h - 1) + 1,
            max(0, j - (patch_w - 1)) : j + (patch_w - 1) + 1,
        ] = 0.0
    return patch_values.min() >= img_copy.max()


def get_corner_points(
    img: NDArray[np.uint8], patch_size: tuple[int, int] = (5, 5)
) -> NDArray[np.int64]:
    """
    Return corner points within a specified radius of the patch size.

    The function determines the corner points inside an image that are located
    within a certain radius defined by the `patch_size`.

    Args:
        img (NDArray[np.uint8]): The image from which to extract corner points.
        patch_size (tuple[int, int], optional): Size of the patches, defined
            as (height, width). Defaults to (5, 5).

    Returns:
        NDArray[np.int64]: Array of corner points within the patch radius.
    """
    img_h, img_w = img.shape[:2]
    radius_h, radius_w = patch_size[0] // 2, patch_size[1] // 2

    corners = np.array(
        [
            [radius_h, radius_w],
            [radius_h, img_w - 1 - radius_w],
            [img_h - 1 - radius_h, img_w - 1 - radius_w],
            [img_h - 1 - radius_h, radius_w],
        ]
    )

    return corners


def draw_red_polygon(
    img: NDArray[np.uint8],
    vertices: NDArray[np.int64],
    coordinates_type: str = "ij",
) -> NDArray[np.uint8]:
    """
    Draw a red polygon on an image given a set of vertices.

    This function first checks the color space of the image. If it is in
    grayscale, it converts it to BGR. Then, depending on the 'coordinates_type'
    parameter, it might swap the x and y coordinates of the vertices. It then
    uses OpenCV's 'polylines' function to draw a polygon on the image using
    the vertices.

    Args:
        img (NDArray[np.uint8]): Input image on which to draw the polygon.
            Can be grayscale or BGR.
        vertices (NDArray[np.int64]): Numpy array containing the vertices of
            the polygon.
        coordinates_type (str, optional): Specifies the order of coordinates
            in the 'vertices' array. Should be either 'xy' or 'ij'.
            Defaults to 'ij'.

    Returns:
        NDArray[np.uint8]: Image with the drawn polygon.

    Raises:
        ValueError: If 'coordinates_type' is not 'xy' or 'ij'.
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if coordinates_type not in {"xy", "ij"}:
        raise ValueError("coordinates_type should be either 'xy' or 'ij'.")

    if coordinates_type == "ij":
        vertices = vertices[:, ::-1]  # swap x and y coordinates

    cv2.polylines(
        img,
        [vertices],
        isClosed=True,
        color=(0, 0, 255),
        thickness=1,
    )

    return img


def sort_points_clockwise(
    points: NDArray[np.int64], coordinates_type: str = "ij"
) -> NDArray[np.int64]:
    """
    Sort points in a clockwise order around their centroid.

    This function calculates the centroid of the given points and computes the
    angles each point forms with the centroid. It then sorts the points based
    on these angles in clockwise order.

    Note that the coordinates type will impact the orientation of the output.

    Args:
        points (NDArray[np.int64]): Numpy array containing the points to be
            sorted.
        coordinates_type (str, optional): Specifies the type of coordinates
            in the 'points' array. Should be either 'xy' (standard cartesian
            2D plan) or 'ij' (ith row and jth column of a 2D array).
            Defaults to 'ij'.

    Returns:
        NDArray[np.int64]: Numpy array of the sorted points.

    Raises:
        ValueError: If 'coordinates_type' is not 'xy' or 'ij'.
    """
    if coordinates_type not in {"xy", "ij"}:
        raise ValueError("coordinates_type should be either 'xy' or 'ij'.")

    if coordinates_type == "ij":
        points = points[:, ::-1]
        points[:, 1] *= -1

    center = np.mean(points, axis=0)
    deltas = points - center
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    sorted_indices = np.argsort(angles)[::-1]

    if coordinates_type == "ij":
        points[:, 1] *= -1
        points = points[:, ::-1]

    return points[sorted_indices]


def calculate_area(sorted_vertices: NDArray[np.int64]) -> float:
    """
    Calculate the area of a polygon given its vertices.

    The points should be ordered in clockwise or counterclockwise direction.

    Args:
        sorted_vertices (NDArray[np.int64]): Numpy array containing the
            vertices of the polygon in sorted order.

    Returns:
        float: Area of the polygon.

    Raises:
        ValueError: If the points are not 2D.
        ValueError: If there are less than 3 points.

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
) -> None:
    """
    Display two images side-by-side using OpenCV.

    The function combines two images horizontally and displays the resultant
    image in a new window. If the images are grayscale, they are converted to
    RGB before combining.

    Args:
        image1 (NDArray[np.uint8]): First image to display.
        image2 (NDArray[np.uint8]): Second image to display.
        title (str, optional): Title for the display window. Defaults to
            "Combined Image".
    """
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    combined = np.hstack((image1, image2))
    cv2.imshow(title, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def compute_integral_image(
    img: NDArray[np.uint8],
    patch_size: tuple[int, int] = (5, 5),
) -> NDArray[np.float64]:
    """
    Compute the local integral image over a patch for each pixel.

    This function first calculates the global integral image of the input
    image, then computes the local integral image (i.e., the sum of pixel
    values within a patch) at each pixel.

    Args:
        img (NDArray[np.uint8]): The input 2D image array.
        patch_size (tuple[int, int], optional): The size of the patches.
            Defaults to (5, 5).

    Returns:
        NDArray[np.float64]: A 2D array containing the local sum at each pixel.
    """
    global_integral_image = img.astype(np.float64)
    global_integral_image = np.pad(
        global_integral_image, ((1, 0), (1, 0)), mode="constant"
    )
    np.cumsum(global_integral_image, axis=0, out=global_integral_image)
    np.cumsum(global_integral_image, axis=1, out=global_integral_image)

    radius_h, radius_w = patch_size[0] // 2, patch_size[1] // 2
    img_h, img_w = img.shape

    local_integral_image = compute_local_integral_image(
        global_integral_image=global_integral_image,
        img_h=img_h,
        img_w=img_w,
        radius_h=radius_h,
        radius_w=radius_w,
    )

    return local_integral_image


def compute_integral_image_v2(
    img: NDArray[np.uint8],
    patch_size: tuple[int, int] = (5, 5),
) -> NDArray[np.float64]:
    """
    Compute the local integral image over a patch for each pixel.

    This function calculates the local integral image directly from the input
    image (i.e., the sum of pixel values within a patch) at each pixel.

    Args:
        img (NDArray[np.uint8]): Input 2D array.
        patch_size (tuple[int, int], optional): Size of the patches.
            Defaults to (5, 5).

    Returns:
        NDArray[np.float64]: 2D array with the local sum at each pixel.
    """
    img_h, img_w = img.shape
    radius_h, radius_w = patch_size[0] // 2, patch_size[1] // 2

    local_integral_image = np.zeros((img_h, img_w), dtype=np.float64)
    for i in range(radius_h, img_h - radius_h):
        for j in range(radius_w, img_w - radius_w):
            local_integral_image[i, j] = img[
                i - radius_h : i + radius_h + 1,
                j - radius_w : j + radius_w + 1,
            ].sum()

    return local_integral_image


def find_patch_centers(
    img: NDArray[np.uint8],
    patch_size: tuple[int, int] = (5, 5),
    num_patches: int = 4,
) -> NDArray[np.int64]:
    """
    Find non-overlapping patches with highest average brightness in the image.

    This function first computes the local integral image for each pixel, then
    selects the 'num_patches' number of patches with highest integral values
    (which translates to the highest average brightness over the patch).

    Args:
        img (NDArray[np.uint8]): Input 2D array.
        patch_size (tuple[int, int], optional): Size of the patches.
            Defaults to (5, 5).
        num_patches (int, optional): Number of patches to find. Defaults to 4.

    Returns:
        NDArray[np.int64]: Indices of the patch centers.
    """
    local_integral = compute_integral_image(img=img, patch_size=patch_size)
    patch_h, patch_w = patch_size

    integral_copy = local_integral.copy()
    indices = []

    for _ in range(num_patches):
        max_val_index = np.unravel_index(
            np.argmax(integral_copy), integral_copy.shape
        )
        indices.append(max_val_index)

        i, j = max_val_index[0], max_val_index[1]
        integral_copy[
            max(0, i - (patch_h - 1)) : i + (patch_h - 1) + 1,
            max(0, j - (patch_w - 1)) : j + (patch_w - 1) + 1,
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
    parser.add_argument(
        "--patch_size",
        "-p",
        type=str,
        default="5,5",
        help="The size of the patch used to find bright spots. Format: 'h,w'",
    )
    parser.add_argument(
        "--disable_display",
        "-d",
        action="store_true",
        help="Disable display of output image.",
    )
    args = parser.parse_args()

    args.patch_size = tuple(map(int, args.patch_size.split(",")))

    return args


def save_image(img: NDArray[np.uint8], output_image_path: Path) -> None:
    """
    Save the image to a file with PNG format.

    This function verifies the output image extension to ensure it is a PNG
    file, and then writes the image data into the file.

    Args:
        img (NDArray[np.uint8]): The image to be saved, in the form of a numpy
            array.
        output_image_path (Path): The output path for the image file. The file
            extension must be '.png'.

    Raises:
        ValueError: If the file extension of 'output_image_path' is not '.png'.

    Returns:
        None
    """
    if output_image_path.suffix.lower() != ".png":
        raise ValueError("Output file must be a '.png' file.")

    cv2.imwrite(str(output_image_path), img)


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
    patch_size = args.patch_size

    img_grayscale = cv2.imread(
        input_image_path.as_posix(), cv2.IMREAD_GRAYSCALE
    )
    patch_centers = find_patch_centers(
        img=img_grayscale, patch_size=patch_size
    )
    assert verify_patch_centers(
        patch_centers=patch_centers, img=img_grayscale, patch_size=patch_size
    )
    sorted_patch_centers = sort_points_clockwise(patch_centers)
    area = calculate_area(sorted_patch_centers)
    print("Area of the quadrilateral:", area)

    img_rgb = draw_red_polygon(img_grayscale, sorted_patch_centers)

    save_image(img=img_rgb, output_image_path=output_image_path)

    if not args.disable_display:
        display_side_by_side(img_grayscale, img_rgb, title=f"Area: {area}")


if __name__ == "__main__":
    main()
