from cython.parallel import prange
import numpy as np
cimport numpy as cnp

def compute_local_integral_image(
    double[:, :] global_integral_image,
    int img_h,
    int img_w,
    int radius_h,
    int radius_w
):
    cdef double[:, :] local_integral_image = np.zeros((img_h, img_w), dtype=np.float64)
    cdef int idx, i, j
    cdef int total_size = (img_h - 2 * radius_h) * (img_w - 2 * radius_w)

    for idx in prange(total_size, nogil=True):
        i = idx // (img_w - 2 * radius_w) + radius_h
        j = idx % (img_w - 2 * radius_w) + radius_w
        local_integral_image[i, j] = (
            global_integral_image[i + radius_h + 1, j + radius_w + 1]
            - global_integral_image[i - radius_h, j + radius_w + 1]
            - global_integral_image[i + radius_h + 1, j - radius_w]
            + global_integral_image[i - radius_h, j - radius_w]
        )

    return np.asarray(local_integral_image)

# def compute_local_integral_image(
#     double[:, :] global_integral_image,
#     int img_h,
#     int img_w,
#     int radius_h,
#     int radius_w
# ):
#     cdef double[:, :] local_integral_image = np.zeros((img_h, img_w), dtype=np.float64)
#     cdef int i, j
# 
#     for i in prange(radius_h, img_h - radius_h, nogil=True):
#         for j in range(radius_w, img_w - radius_w):
#             local_integral_image[i, j] = (
#                 global_integral_image[i + radius_h + 1, j + radius_w + 1]
#                 - global_integral_image[i - radius_h, j + radius_w + 1]
#                 - global_integral_image[i + radius_h + 1, j - radius_w]
#                 + global_integral_image[i - radius_h, j - radius_w]
#             )
# 
#     return np.asarray(local_integral_image)
