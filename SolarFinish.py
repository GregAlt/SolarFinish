__copyright__ = "Copyright (C) 2023 Greg Alt"
__version__ = "0.14.2"

# TODOS        - clarify silent, interact, verbose modes
#              - possibly add invert option - can just take 1- final grayscale
#              - breakup/cleanup into multiple files (might mean abandoning Colab?)
#              - more attention to removing artifacts/noise beyond limb
#              - better control over min/max contrast adjustment params. Most flexible
#                would be 4 params for min/max input and min/max output
#              - better sub-pixel circle finding, and shifting before processing
#              - how to allow more continuous brightness of filaproms across limb?

try:
    import google.colab.files
    import IPython.display

    IN_COLAB = True
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
except ImportError:
    IN_COLAB = False
    import argparse
    import os
    import re
    import PySimpleGUI as sg

import math
import numpy as np
import cv2 as cv
import scipy as sp
import requests
import datetime
import urllib.request
import astropy.io.fits
import io
from contextlib import redirect_stdout
import sys

#
# Circle finding

# Assumes sun diameter is smaller than image, and sun isn't too small
def is_valid_circle(shape, radius):
    size = min(shape[0], shape[1])
    if 2 * radius > size or 2 * radius < 0.25 * size:
        return False
    return True


# Utility to convert ellipse data to center, radius
def get_circle_data(ellipse):
    if ellipse is None:
        return (0, 0), 0
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    radius = int(0.5 + ellipse[0][2] + (ellipse[0][3] + ellipse[0][4]) * 0.5)
    return center, radius


# Check that ellipse meets valid circle criteria
def is_valid_ellipse(shape, ellipse):
    (center, radius) = get_circle_data(ellipse)
    return is_valid_circle(shape, radius)


# Use Edge Drawing algorithm to find biggest valid circle, assumed to be the solar disk
def find_circle(src):
    # convert to 8bit grayscale for ellipse-detecting
    gray = (src / 256).astype(np.uint8)

    ed_params = cv.ximgproc_EdgeDrawing_Params()
    ed_params.MinPathLength = 300
    ed_params.PFmode = True
    ed_params.MinLineLength = 10
    ed_params.NFAValidation = False

    ed = cv.ximgproc.createEdgeDrawing()
    ed.setParams(ed_params)
    ed.detectEdges(gray)
    ellipses = ed.detectEllipses()

    if ellipses is None:
        return None

    # reject invalid ones *before* finding largest
    ellipses = [e for e in ellipses if is_valid_ellipse(src.shape, e)]
    if len(ellipses) == 0:
        return None

    # find ellipse with the biggest max axis
    return ellipses[np.array([e[0][2] + max(e[0][3], e[0][4]) for e in ellipses]).argmax()]


# Returns True plus center and radius in pixels, if solar disk circle is found
# If it fails at first, maybe that's due to a blurry high-res image, so try on a smaller version
def find_valid_circle(src):
    (center, radius) = get_circle_data(find_circle(src))
    if not is_valid_circle(src.shape, radius):
        # try shrinking image
        thousands = math.ceil(min(src.shape[0], src.shape[1]) / 1000)
        for scale in range(2, thousands + 1):
            smaller = cv.resize(src, (int(src.shape[1] / scale), int(src.shape[0] / scale)))
            (small_center, small_radius) = get_circle_data(find_circle(smaller))
            (center, radius) = ((small_center[0] * scale + scale // 2, small_center[1] * scale + scale // 2),
                                small_radius * scale + scale // 2)
            if is_valid_circle(src.shape, radius):
                break
    return (True, center, radius) if is_valid_circle(src.shape, radius) else (False, None, None)


#
# Pixel format conversions

# Simple linear conversion from RGB to single-channel grayscale
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# Just replicate grayscale channel across RGB channels to make a three-channel grayscale image
def gray2rgb(im):
    return cv.merge([im, im, im])


# Colorize 0-1 float image with given RGB gamma values,
# then return as 0-1 float image with B and R swapped
def colorize_float_bgr(result, r, g, b):
    bgr = (np.power(result, b), np.power(result, g), np.power(result, r))
    return cv.merge(bgr)


# Colorize 0-1 float image with given RGB gamma values,
# then return as 0-65535 16 bit image with B and R swapped
def colorize16_bgr(result, r, g, b):
    return float01_to_16bit(colorize_float_bgr(result, r, g, b))


# Colorize 0-1 float image with given RGB gamma values, then return as 0-255 8 bit image in RGB format
def colorize8_rgb(im, r, g, b):
    rgb = (np.power(im, r), np.power(im, g), np.power(im, b))
    return float01_to_8bit(cv.merge(rgb))


# Given arbitrary image file, whether grayscale or RGB, 8 bit or 16 bit, or even 32 bit float
# return as 0-65535 16 bit single-channel grayscale
def force16_gray(im):
    im = rgb2gray(im) if len(im.shape) > 2 else im
    return cv.normalize(im.astype(np.float32), None, 0, 65535, cv.NORM_MINMAX).astype(np.uint16)


# Given a single-channel grayscale 16 bit image, return a three-channel 0-255 8 bit grayscale image
def gray16_to_rgb8(im):
    return gray2rgb((im / 256).astype(np.uint8))


# Convert image from float 0-1 to 16bit uint, works with grayscale or RGB
def float01_to_16bit(im):
    return (im * 65535).astype(np.uint16)


# Convert image from float 0-1 to 8bit uint, works with grayscale or RGB
def float01_to_8bit(im):
    return (im * 255).astype(np.uint8)


# Convert image from 16bit uint to float 0-1, works with grayscale or RGB
def to_float01_from_16bit(im):
    return im.astype(np.float32) / 65535.0


#
# Image filtering and warp/un-warping

# Return image rotated by the given angle in degrees, around the given center.
# Keeps original image dimensions
def rotate(im, center, angle_deg):
    rows, cols = im.shape[0:2]
    m = cv.getRotationMatrix2D(center, angle_deg, 1)
    return cv.warpAffine(im, m, (cols, rows))


# rotation with expanding the result, and fill the padded triangles from the border
def rotate_with_expand_fill(im, angle_deg):
    # do a test rotate to get new dimensions
    test_shape = sp.ndimage.rotate(im, angle_deg).shape
    ty, tx = test_shape[0:2]

    # expand unrotated image using border fill
    temp_center = (im.shape[1] // 2, im.shape[0] // 2)
    expand_dist = center_and_expand_get_dist((tx // 2, ty // 2), test_shape)
    temp_center, im2 = center_and_expand_to_dist(temp_center, im, expand_dist)

    # rotate the expanded image
    src = sp.ndimage.rotate(im2, angle_deg)

    # and then crop back to the original test rotate image dimensions
    sy, sx = src.shape[0:2]
    start_y, start_x = sy // 2 - ty // 2, sx // 2 - tx // 2
    im2 = src[start_y: start_y + ty, start_x: start_x + tx]
    return im2


# Turns a centered solar disk image from a disk to a rectangle,
# with rows being angle and columns being distance from center
def polar_warp(img):
    return cv.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), img.shape[0], cv.WARP_FILL_OUTLIERS)


# Turns polar warped image from rectangle back to unwarped solar disk
def polar_unwarp(img, shape):
    # INTER_AREA works best to remove artifacts
    # INTER_CUBIC works well except for a horizontal line artifact at angle = 0
    # INTER_LINEAR, the default has a very noticeable vertical banding artifact across the top, and similarly around the limb
    unwarped = cv.linearPolar(img, (shape[0] / 2, shape[1] / 2), shape[0],
                              cv.WARP_FILL_OUTLIERS | cv.WARP_INVERSE_MAP | cv.INTER_AREA)
    return unwarped


# Calc images with kernel mean and stddev per pixel, with n being fraction of
# circle for kernel around each pixel. Example, 6 means kernels are 60 degree
# arcs. Assumes polar image so that curved kernels are treated as rectangles.
def get_mean_and_std_dev_image(polar_image, n):
    h = polar_image.shape[0]  # image is square, so h=w
    k = (h // (n * 2)) * 2 + 1  # find kernel size from fraction of circle, ensure odd
    left_half = polar_image[:, 0:h // 2]  # left half is circle of radius h//2
    (mean, stddev) = mean_and_std_dev_filter_2d_with_wraparound(left_half, (k, 1))

    # don't use mean filter for corners, just copy that data directly to minimize artifacts
    right_half = polar_image[:, h // 2:]  # right half is corners and beyond
    mean_image = cv.hconcat([mean, right_half])

    # don't use stddev filter for corners, just repeat last column to minimize artifacts
    std_dev_image = np.hstack((stddev, np.tile(stddev[:, [-1]], h - h // 2)))
    return mean_image, std_dev_image


# Pad the image on top and bottom to allow filtering with simulated wraparound
def pad_for_wrap_around(inp, pad):
    return cv.vconcat([inp[inp.shape[0] - pad:, :], inp, inp[:pad, :]])


# Remove padding from top and bottom
def remove_wrap_around_pad(input_padded, pad):
    return input_padded[pad:input_padded.shape[0] - pad, :]


# Low-level func to find mean and std dev images for polar warped image,
# ensuring results are as if kernels wrap around at top and bottom
def mean_and_std_dev_filter_2d_with_wraparound(inp, kernel_size):
    # pad input image with half of kernel to simulate wraparound
    image_pad = pad_for_wrap_around(inp, kernel_size[0] // 2)

    # filter the padded image
    mean_pad = sp.ndimage.uniform_filter(image_pad, kernel_size, mode='reflect')
    mean_of_squared_pad = sp.ndimage.uniform_filter(image_pad * image_pad, kernel_size, mode='reflect')

    # sqrt(mean_of_squared - mean*mean) is mathematically equivalent to std dev:
    #   https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows
    std_dev_pad = np.sqrt((mean_of_squared_pad - mean_pad * mean_pad).clip(min=0))

    mean = remove_wrap_around_pad(mean_pad, kernel_size[0] // 2)
    stddev = remove_wrap_around_pad(std_dev_pad, kernel_size[0] // 2)
    return mean, stddev


#
# Misc image processing

# Adjust brightness with a blended function combining true gamma (which
# emphasizes brightening of darker pixels) and a function that brightens
# the same but emphasizes brightening the darker pixels. Weight of 1 means
# just gamma, weight of 0 means the other function. weight of 0.5 puts the
# bulge in the intensity curve in the middle.
def brighten(im, gamma, gamma_weight):
    return gamma_weight * np.power(im, gamma) + (1 - gamma_weight) * (1 - np.power(1 - im, 1 / gamma))


# Swap red and blue due to BGR vs RGB expectations in some OpenCV functions
def swap_rb(im):
    blue = im[:, :, 0].copy()
    im[:, :, 0] = im[:, :, 2].copy()
    im[:, :, 2] = blue
    return im


# Return image shrunk by div. So div=2 means half size
def shrink(im, div):
    return cv.resize(im, np.floor_divide((im.shape[1], im.shape[0]), div))


# scale image by percent
def zoom_image(im, zoom):
    return cv.resize(im, np.floor_divide((int(im.shape[1] * zoom), int(im.shape[0] * zoom)), 100))


# Return an image with circle drawn on it for visualizing circle finding
def add_circle(im, center, radius, color, thickness):
    cv.ellipse(im, center, (radius, radius), 0, 0, 360, color, thickness, cv.LINE_AA)
    return im


# expand given a distance
def center_and_expand_to_dist(center, src, max_dist):
    to_left, to_right = (center[0], src.shape[1] - center[0])
    to_top, to_bottom = (center[1], src.shape[0] - center[1])
    new_center = (max_dist, max_dist)
    out_img = np.pad(src, ((max_dist - to_top, max_dist - to_bottom), (max_dist - to_left, max_dist - to_right)),
                     mode='edge')
    return new_center, out_img


# calculate the distance needed for expanding
def center_and_expand_get_dist(center, shape):
    to_left, to_right = (center[0], shape[1] - center[0])
    to_top, to_bottom = (center[1], shape[0] - center[1])
    to_ul = math.sqrt(to_top * to_top + to_left * to_left)
    to_ur = math.sqrt(to_top * to_top + to_right * to_right)
    to_bl = math.sqrt(to_bottom * to_bottom + to_left * to_left)
    to_br = math.sqrt(to_bottom * to_bottom + to_right * to_right)
    return int(max(to_ul, to_ur, to_bl, to_br)) + 1


# Create an expanded image centered on the sun. Ensure that a bounding circle
# centered on the sun and enclosing the original image's four corners is fully
# enclosed in the resulting image. For added pixels, pad by copying the existing
# edge pixels. This means that processing of the polar-warped image has reasonable
# values out to the maximum distance included in the original source image. This,
# in turn, means that circular banding artifacts will occur farther out and can be
# fully cropped out at the end.
def center_and_expand(center, src):
    return center_and_expand_to_dist(center, src, center_and_expand_get_dist(center, src.shape))


# Crop image to a square with given min distance from center. Return new image and center.
# If specified min_dist is too large clamp it, because this function can only crop
def crop_to_dist(src, center, min_dist):
    min_dist = min(math.ceil(min_dist), src.shape[0] // 2)  # don't allow a crop larger than the image
    new_center = (min_dist, min_dist)
    # note, does NOT force to odd
    out_img = src[center[1] - min_dist:center[1] + min_dist, center[0] - min_dist:center[0] + min_dist]
    return new_center, out_img


# Given a center and image dimensions, find the shortest distance to image boundary
def calc_min_dist_to_edge(center, shape):
    to_left, to_right = (center[0], shape[1] - center[0])
    to_top, to_bottom = (center[1], shape[0] - center[1])
    min_dist = int(min(to_left, to_top, to_right, to_bottom)) - 1
    return min_dist


# Given solar center co-ords and image, return an image with sun centered and
# image cropped to largest square that fits. Also return new center co-ords
def center_and_crop(center, src):
    return crop_to_dist(src, center, calc_min_dist_to_edge(center, src.shape))


# Scale image so that solar radius matches new_rad in pixels
def force_radius(im, center, rad, new_rad):
    scale = new_rad / rad
    im2 = cv.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))
    center2 = (int(center[0] * scale), int(center[1] * scale))
    return center2, im2


# Create a float image mask with 1 for pixels in solar disk, 0 for others
def get_disk_mask(src, center, radius):
    # create 32 bit float disk mask
    disk_mask = np.zeros(src.shape[:2], dtype="float32")
    cv.ellipse(disk_mask, center, (radius, radius), 0, 0, 360, 1.0, -1, cv.FILLED)  # no LINE_AA!
    return disk_mask


#
# Functions need to evaluate alignment similarity, using log-gabor filter

# Log-Gabor filter
# from https://stackoverflow.com/questions/31774071/implementing-log-gabor-filter-bank/31796747
def get_log_gabor_filter(n, f_0, theta_0, number_orientations):
    # filter configuration
    scale_bandwidth = 0.996 * math.sqrt(2 / 3)
    angle_bandwidth = 0.996 * (1 / math.sqrt(2)) * (np.pi / number_orientations)

    # x,y grid
    extent = np.arange(-n / 2, n / 2 + n % 2)
    x, y = np.meshgrid(extent, extent)

    mid = int(n / 2)
    # orientation component
    theta = np.arctan2(y, x)
    center_angle = ((np.pi / number_orientations) * theta_0) if (f_0 % 2) \
        else ((np.pi / number_orientations) * (theta_0 + 0.5))

    # calculate (theta-center_theta), we calculate cos(theta-center_theta)
    # and sin(theta-center_theta) then use arctan2 to get the required value,
    # this way we can eliminate the angular distance wrap around problem
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    ds = sin_theta * math.cos(center_angle) - cos_theta * math.sin(center_angle)
    dc = cos_theta * math.cos(center_angle) + sin_theta * math.sin(center_angle)
    d_theta = np.arctan2(ds, dc)

    orientation_component = np.exp(-0.5 * (d_theta / angle_bandwidth) ** 2)

    # frequency component
    raw = np.sqrt(x ** 2 + y ** 2)  # go to polar space
    raw[mid, mid] = 1  # set origin to 1 as in the log space zero is not defined
    raw = np.log2(raw)  # go to log space

    center_scale = math.log2(n) - f_0
    draw = raw - center_scale
    frequency_component = np.exp(-0.5 * (draw / scale_bandwidth) ** 2)

    # reset origin to zero (not needed as it is already 0?)
    frequency_component[mid, mid] = 0

    kernel = frequency_component * orientation_component
    return kernel


# Utility function to do both fft and shift
def fft(im):
    return np.fft.fftshift(np.fft.fft2(im))


# Utility function to do both inverse fft and shift
def ifft(f):
    return np.real(np.fft.ifft2(np.fft.ifftshift(f)))


# Create the frequency space log-gabor filter image for all orientations
def get_lgs(n, f_0, num_orientations):
    return [get_log_gabor_filter(n, f_0, x, num_orientations) for x in range(0, num_orientations)]


# Apply log-gabor filter to image
def apply_filter(im, lg):
    # apply fft to go to frequency space, apply filter, then inverse fft to go back to spatial
    # take absolute value, so we have only non-negative values, and merge into multichannel image
    f = [np.abs(ifft(fft(im) * lg[x])) for x in range(0, len(lg))]
    im = cv.merge(f)
    return im


#
# Implementation of algorithm in Aligning 'Dissimilar' Images Directly (Yaser Sheikh, 2003)
# Some assumptions and modifications made to improve performance

# Internal function to calculate Rij image for similarity measure
def get_rij(num, den, k):
    # TODO: cleanup this conditional code meant to exclude very small denominators
    den_shape = den.shape
    den = den.flatten()
    num = num.flatten()
    rij = np.zeros(den.shape)
    epsilon = 0.0001 / (k * k)  # divide here because I simplified out k*k* out of den
    rij[den > epsilon] = num[den > epsilon] / den[den > epsilon]
    rij = np.reshape(rij, den_shape)
    return rij


# Internal function to calculate n image given Rij, assume c=2 to simplify and boost performance
def calc_n(rij):
    abs_rij = abs(rij)
    # c = 2#1 # what value for constant?
    # n = 1/(1+np.power((1-abs_rij)/(1+abs_rij), c/2))
    n = 1 / (1 + ((1 - abs_rij) / (1 + abs_rij)))  # optimized for c=2, power can be removed
    return n


# Calculate similarity sum, H, given n
def get_similarity_sum(n):
    n_flat = np.zeros(n.shape[:2])
    for i in range(n.shape[2]):
        n_flat = n_flat + n[:, :, i]
    h = np.sum(n)
    return h, n_flat


# Calculate high-level H similarity between two images, the core of alignment algorithm
def similarity(im1, im2):
    # find the correlation coefficient at each pixel rij
    k = 5
    bf1 = cv.boxFilter(im1, -1, (k, k))
    bf2 = cv.boxFilter(im2, -1, (k, k))
    bf12 = cv.boxFilter(im1 * im2, -1, (k, k))
    bf11 = cv.boxFilter(im1 * im1, -1, (k, k))
    bf22 = cv.boxFilter(im2 * im2, -1, (k, k))

    # Optimized by heavily simplifying from sum over phi1*phi2:
    # equivalent to quadruple-nested loop with:
    #   r[i,j] += (im1[i+ki,j+kj]-u1[i,j]) * im2[i+ki,j+kj]-u2[i,j])
    # which is
    #   r[i,j] += im1[i+ki,j+kj]*im2[i+ki,j+kj] -u1[i,j]*im2[i+ki,j+kj] -u2[i,j]*im1[i+ki,j+kj] +u1[i,j]*u2[i,j]
    # r = 25*(boxFilter(im1*im2,5) + (-u1)*boxFilter(im2,5) + (-u2)*boxFilter(im1,5) + (u1*u2))
    # also removed the 25* since it all cancels out, except I had to adjust the epsilon
    phi11 = (bf11 - bf1 * bf1)
    phi12 = (bf12 - bf1 * bf2)
    phi22 = (bf22 - bf2 * bf2)
    num = phi12
    den = cv.sqrt(phi22 * phi11)
    rij = get_rij(num, den, k)
    n = calc_n(rij)
    return get_similarity_sum(n)


#
# local and global search of alignment based on similarity evaluation

# Visualization to show progress of alignment evaluations
def show_eval(_gong_filtered, input_rot, _input_filtered, _h, n, _angle):
    if not IN_COLAB:
        cv.imshow("input_rot", input_rot)
        cv.imshow("n", n * 0.5 - 1.0)
        cv.waitKey(1)


# Given prefiltered target image and input image plus mask and log-gabor,
# evaluate similarity H for a given angle
def local_search_evaluate(inp, lg, mask, gong_filtered, angle, show_eval_func):
    input_rot = rotate(inp, (inp.shape[1] // 2, inp.shape[0] // 2), angle)
    input_filtered = apply_filter(input_rot, lg) * mask
    (H, n) = similarity(gong_filtered, input_filtered)
    if show_eval_func is not None:
        show_eval_func(gong_filtered, input_rot, input_filtered, H, n, angle)
    return H, n, input_rot, input_filtered


# Start with a rough peak expected to be near the best angle, with 3 data
# points, iterate until narrow enough and return final peak which is the
# best matching angle
def local_search(inp, lg, mask, gong_filtered, triple, stopping, report_callback):
    angles, h_list = triple
    while abs(angles[1] - angles[2]) > stopping:
        # evaluate midpoints
        ang = [angles[0], 0.5 * (angles[0] + angles[1]), angles[1], 0.5 * (angles[1] + angles[2]), angles[2]]
        results1 = local_search_evaluate(inp, lg, mask, gong_filtered, ang[1], report_callback)
        results2 = local_search_evaluate(inp, lg, mask, gong_filtered, ang[3], report_callback)
        sim = [h_list[0], results1[0], h_list[1], results2[0], h_list[2]]

        # find resulting peak and new triples
        peak = np.argmax(sim[1:5]) + 1  # first one can't be peak, but might be equal to peak
        # print(f"Local {ang[peak]}, {sim[peak]}, delta={ang[1]-ang[0]}")
        angles, h_list = (ang[peak - 1:peak + 2], sim[peak - 1:peak + 2])
    return angles[1], h_list[1]


# Roughly find neighborhood of best matching angle for given input and
# pre-filtered target. Do a full step-by-step scan from start to end angle,
# with count steps. Return the best angle and neighbors on either side, to
# be used as input to local search
def global_search(inp, lg, mask, gong_filtered, start, end, count, _best_angle, _best_sim, show_eval_func):
    angles = []
    h_list = []
    for angle in np.linspace(start, end, num=count, endpoint=False):
        input_rot = rotate(inp, (inp.shape[1] // 2, inp.shape[0] // 2), angle)
        input_filtered = apply_filter(input_rot, lg) * mask
        (h, n) = similarity(gong_filtered, input_filtered)
        angles.append(angle)
        h_list.append(h)
        if show_eval_func is not None:
            show_eval_func(gong_filtered, input_rot, input_filtered, h, n, angle)

    peak = np.argmax(h_list)
    a = (peak - 1 + len(h_list)) % len(h_list)
    b = (peak + 1 + len(h_list)) % len(h_list)
    return [angles[a], angles[peak], angles[b]], [h_list[a], h_list[peak], h_list[b]]


# Expand, then center and crop to get a square image cropped to a size in terms of solar radii
def center_and_crop_to_fixed_radius(center, radius, im, pixel_radius, solar_radii):
    # ensure we have enough buffer, scale to get fixed radius, then crop
    (center, im) = center_and_expand(center, im)
    (center, im) = force_radius(im, center, radius, pixel_radius)
    (center, im) = crop_to_dist(im, center, pixel_radius * solar_radii)
    return center, im


# Assumes global alignment search has already been done, supplying is_flipped and initial 3 results in triple
def local_search_helper(inp, is_flipped, lg, mask, gong, gong_filtered, triple, silent):
    input_search = cv.flip(inp, 1) if is_flipped else inp
    show_eval_func = None if silent else show_eval
    (angle, sim) = local_search(input_search, lg, mask, gong_filtered, triple, 0.1, show_eval_func)
    return angle, sim, is_flipped, True, triple, gong, input_search


# Does a full alignment search including global and local
def full_search_helper(inp, _is_flipped, lg, mask, gong, gong_filtered, _triple, silent):
    show_eval_func = None if silent else show_eval
    triple = global_search(inp, lg, mask, gong_filtered, -180, 180, 20, 0, -1, show_eval_func)
    (unflipped_angle, unflipped_sim) = local_search(inp, lg, mask, gong_filtered, triple, 0.1, show_eval_func)
    flipped = cv.flip(inp, 1)
    flipped_triple = global_search(flipped, lg, mask, gong_filtered, -180, 180, 20, 0, -1, show_eval_func)
    (flippedAngle, flippedSim) = local_search(flipped, lg, mask, gong_filtered, flipped_triple, 0.1, show_eval_func)

    return (unflipped_angle, unflipped_sim, False, True, triple, gong, inp) if unflipped_sim > flippedSim else (
        flippedAngle, flippedSim, True, True, flipped_triple, gong, flipped)


#
# high-level alignment implementation

# Prepare source and target images for alignment by centering and cropping
# and forcing to equal size solar disk
def center_images_for_alignment(inp, gong, fixed_radius, solar_radii):
    (is_valid_gong, gong_center, gong_radius) = find_valid_circle(gong)
    if not is_valid_gong:
        print("Error: Couldn't find valid circle for GONG solar disk!", flush=True)

    (is_valid_input, input_center, input_radius) = find_valid_circle(inp)
    if not is_valid_input:
        print("Error: Couldn't find valid circle for input solar disk!", flush=True)

    if is_valid_gong and is_valid_input:
        (gong_center, gong) = center_and_crop_to_fixed_radius(gong_center, gong_radius, to_float01_from_16bit(gong),
                                                              fixed_radius,
                                                              solar_radii)
        (input_center, inp) = center_and_crop_to_fixed_radius(input_center, input_radius, to_float01_from_16bit(inp),
                                                              fixed_radius,
                                                              solar_radii)

    return is_valid_gong, gong, gong_center, is_valid_input, inp, input_center


# Do a single alignment with one source image and target image
def align_images(gong, inp, fixed_radius, triple, is_flipped, silent, search_func):
    (is_valid_gong, gong, gong_center, is_valid_input, inp, input_center) = center_images_for_alignment(inp, gong,
                                                                                                        fixed_radius,
                                                                                                        1.1)
    if not is_valid_gong or not is_valid_input:
        return 0, 0, False, False, ([], []), gong, inp

    if not silent and not IN_COLAB:
        cv.imshow("gong", gong)
        cv.waitKey(1)

    lg = get_lgs(gong.shape[0], 4, 4)
    mask = cv.merge([get_disk_mask(gong, gong_center, int(fixed_radius * 0.8)) for _ in range(len(lg))])
    gong_filtered = apply_filter(gong, lg) * mask

    return search_func(inp, is_flipped, lg, mask, gong, gong_filtered, triple, silent)


# Do a single alignment with one source image and one date and percentage time of day
def align(inp, date, percent, triple, flipped, silent, search_func):
    gong = get_gong_image_for_date(datetime.datetime.strptime(date, '%Y/%m/%d'), percent)
    (angle, sim, flipped, matchFound, triple, gong_out, input_out) = align_images(gong, inp, 128, triple, flipped,
                                                                                  silent, search_func)
    return angle, sim, triple, flipped, gong, gong_out, inp, input_out


# Given an image and a date, find the best alignment angle using GONG images for that day
def find_best_alignment(input_i, date, silent):
    (angle, sim, triple, flipped, gong_big, gong, input_big, inp) = align(input_i, date, 0.5, None, None, silent,
                                                                          full_search_helper)
    best = (0.5, angle, flipped, sim, gong_big, gong, input_big, inp)
    percents = np.linspace(0.1, 0.9, 9).tolist()  # from 0.1 to 0.9, inclusive, by 0.1
    percents.remove(0.5)  # and remove 0.5
    for percent in percents:
        (angle, sim, triple, flipped, gong_big, gong, input_big, inp) = align(input_i, date, percent, triple,
                                                                              flipped, silent,
                                                                              local_search_helper)
        if sim > best[3]:
            best = (percent, angle, flipped, sim, gong_big, gong, input_big, inp)
    return best


#
# File uploading, downloading, and rendering

# Fetch a .fz image file by url, assumes conventions used by GONG
def get_image_fz(url):
    trap = io.StringIO()
    with redirect_stdout(trap):  # hide the obnoxious progress bar
        image_data = astropy.io.fits.getdata(url)
    img_float = image_data.astype(np.float32).clip(min=0) / np.max(image_data)
    img = float01_to_16bit(img_float)
    return cv.flip(img, 0)  # image co-ords are upside down


# Fetch image file by url, return as 16-bit grayscale
def get_image(url):
    fn = "testalign.png"
    open(fn, 'wb').write(requests.get(url, allow_redirects=True).content)
    return force16_gray(read_image(fn))  # force to single-channel 16-bit grayscale


# Internal function to get GONG image url by date and percentage time of day
def get_gong_image_url(date, percent, gong_root, file_end):
    yyyy = date.strftime("%Y")
    mm = date.strftime("%m")
    dd = date.strftime("%d")
    gong_dir = gong_root + yyyy + mm + "/" + yyyy + mm + dd + "/"
    data = urllib.request.urlopen(gong_dir).read()
    data2 = data.split(b'\"')
    w = [str(a)[2:-1] for a in data2 if str(a)[2:4] == '20' and str(a)[-5:-1].lower() == file_end]
    fn = w[int(len(w) * percent)]
    gong_full_path = gong_dir + fn
    return gong_full_path


# Get GONG .fz raw image url by date and percentage time of day
def get_gong_image_for_date(date, percent):
    # return getImage(GetGongImageURL(date, percent, "https://gong2.nso.edu/HA/hag/", '.jpg'))
    return get_image_fz(get_gong_image_url(date, percent, "https://nispdata.nso.edu/ftp/HA/haf/", 's.fz'))


# Utility function to read image by filename
def read_image(fn):
    return cv.imread(cv.samples.findFile(fn), cv.IMREAD_UNCHANGED | cv.IMREAD_ANYDEPTH)


# Ask Colab user to select file to upload and return resulting filename on server
def upload_file():
    keys = list(google.colab.files.upload().keys())
    return keys[0] if keys else ""


# Construct output filename for output by replacing extension and adding suffix
def make_filename_for_write(fn, suffix):
    # strip full path after the last .
    without_extension = fn[::-1].split('.', 1)[1][::-1]  # reverse, split first ., take 2nd part, reverse again
    out_fn = without_extension + '-' + suffix + '.png'
    return out_fn


# Write png image to disk with given suffix
def write_image(im, fn, suffix):
    out_fn = make_filename_for_write(fn, suffix)
    print(f"writing: {out_fn}", flush=True)
    try:
        cv.imwrite(out_fn, im)
    except Exception as error:
        print(f"Error: Failed to save {out_fn}, likely bad file extension. Try .PNG\n{error}", flush=True)
    return out_fn


# Utility function to download file from Colab
def download_image(im, fn, suffix):
    google.colab.files.download(write_image(im, fn, suffix))


# Create a download button for a file in Colab
def download_button(im, fn, suffix):
    if IN_COLAB:
        button = widgets.Button(description='Download Image')
        button.on_click(lambda x: download_image(im, fn, suffix))
        IPython.display.display(button)


# Visualization of RGB image, either in Colab or running locally
def show_rgb(im):
    if IN_COLAB:
        plt.imshow(im)
        plt.show()
    else:
        cv.imshow(str(id(im)), swap_rb(shrink(im, 3)))
        cv.waitKey(1)


# Visualization of 0-1 float grayscale image, either in Colab or running locally
def show_float01(im):
    show_rgb(colorize8_rgb(im, 1, 1, 1))


#
# Enhancement

# Returns a function that normalizes to within a given range
def get_std_dev_scaler(min_recip, max_recip):
    return lambda sd: cv.normalize(sd, None, 1 / max_recip, 1 / min_recip, cv.NORM_MINMAX)


# Helper function for displaying and downloading image
def display_and_download(im, text, should_download, fn, suffix):
    print(text, flush=True)
    show_float01(im)
    if should_download:
        download_button(float01_to_16bit(im), fn, suffix)


# Visualization and download of intermediate images for enhancement process, for part 1
def display_cnrgf_intermediate_1(polar_image, mean_image, unwarped_mean, fn):
    # display_and_download(polar_image, "Displaying intermediate image: polar warp the image as an initial step to make a pseudo-flat", False, fn, "")
    # display_and_download(mean_image, "Displaying intermediate image: mean filter on polar warp image", False, fn, "")
    display_and_download(unwarped_mean, "Displaying intermediate image: unwarped mean pseudo-flat", True, fn,
                         "unwarpedmean")


# Visualization and download of intermediate images for enhancement process, for part 2
def display_cnrgf_intermediate_2(diff, norm_std_dev, enhance_factor, fn):
    display_and_download(diff + 0.5, "Displaying intermediate image: subtract pseudo-flat from input image", True, fn,
                         "diff")
    display_and_download(norm_std_dev,
                         "Displaying intermediate image: standard deviation filter, to drive contrast enhancement",
                         False, fn, "")
    display_and_download((diff * enhance_factor + 0.5).clip(min=0, max=1),
                         "Displaying intermediate image: enhanced contrast in diff image", True, fn, "diff-enhanced")


# CNRGF split into two parts, first part does expensive convolutions
def cnrgf_enhance_part1(img, n, show_intermediate_1, fn):
    # find mean and standard deviation image from polar-warped image, then un-warp
    polar_image = polar_warp(img)
    mean_image, std_devs = get_mean_and_std_dev_image(polar_image, n)
    unwarped_mean = polar_unwarp(mean_image, img.shape)
    unwarped_std_dev = polar_unwarp(std_devs, img.shape)

    if show_intermediate_1 is not None:
        show_intermediate_1(polar_image, mean_image, unwarped_mean, fn)
    return unwarped_mean, unwarped_std_dev


# convert img with cropping/padding to be of size shape, and midpoint at center
def extend_to_match(img, shape, center):
    crop_y_start = img.shape[0] // 2 - center[1]
    crop_y_end = shape[0] - center[1] + img.shape[0] // 2
    crop_x_start = img.shape[1] // 2 - center[0]
    crop_x_end = shape[1] - center[0] + img.shape[1] // 2
    cropped = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    before_y = min(0, img.shape[0] // 2 - center[1])
    after_y = shape[0] - cropped.shape[0] - before_y
    before_x = min(0, img.shape[1] // 2 - center[0])
    after_x = shape[1] - cropped.shape[1] - before_x

    result = np.pad(cropped, ((before_y, after_y), (before_x, after_x)), mode='constant')
    return result


# CNRGF split into two parts, second part is cheaper and has tunable parameters
# using scaleStdDev as a function that has tunable parameters baked into it.
def cnrgf_enhance_part2(img, center, mean_and_stddev, scale_std_dev, show_intermediate_2, fn):
    # adjust range of standard deviation image to get preferred range of contrast enhancement
    unwarped_mean, unwarped_std_dev = mean_and_stddev
    norm_std_dev = scale_std_dev(unwarped_std_dev)

    # subtract mean, divide by standard deviation, and add back mean
    enhance_factor = np.reciprocal(norm_std_dev)

    # img might not be centered and square. So unwarped_mean and enhance_factor need to be
    # first extended to be the same size and solar center as img
    um2 = extend_to_match(unwarped_mean, img.shape, center)
    ef2 = extend_to_match(enhance_factor, img.shape, center)
    diff = img - um2
    enhanced = diff * ef2 + um2

    if show_intermediate_2 is not None:
        show_intermediate_2(diff, norm_std_dev, enhance_factor, fn)
    return enhanced


# Do full CNRGF enhancement from start to finish, displaying intermediate results. Takes a
# float 0-1 image with a centered solar disk.
#
# This uses a process I call Convolutional Normalizing Radial Graded Filter (CNRGF).
# CNRGF was developed largely independently of but was influenced by Druckmullerova's
# FNRGF technique. Instead of using a fourier series to approximate mean and stddev
# around each ring, CNRGF does a simple mean and stddev convolutional filter on a
# polar warped image, and then un-warps those results. This allows for a fairly simple
# and fast python implementation with similar effect of adaptively applying enhancement
# and addressing the radial gradient. CNRGF was developed for processing full-disk
# hydrogen alpha images, while FNRGF was developed for coronal images beyond 1 solar
# radius, but the problems have many similarities, and it should be possible to use the
# algorithms interchangeably for solar images more generally, including full disk
# white light images.
def cnrgf_enhance(src, src_center, n, min_recip, max_recip, min_clip, show_intermediate_1, show_intermediate_2, fn):
    (center, centered) = center_and_expand(src_center, src)
    mean_and_stddev = cnrgf_enhance_part1(centered, n, show_intermediate_1, fn)
    enhanced = cnrgf_enhance_part2(src, src_center, mean_and_stddev, get_std_dev_scaler(min_recip, max_recip),
                                   show_intermediate_2, fn)
    clipped = enhanced.clip(min=min_clip)
    normalized = cv.normalize(clipped, None, 0, 1, cv.NORM_MINMAX).clip(min=0).clip(max=1)
    return normalized


#
# Interactive

# Drive interactive adjustment and visualization of parameters with sliders, return final params selected
def interactive_adjust(filename_or_url, directory, output_directory, suffix, silent, should_enhance, min_adj, max_adj,
                       gamma, gamma_weight, min_clip, should_crop, crop_radius, h_flip, v_flip, rotation,
                       show_colorized, rgb_weights, align_date, should_align):
    def on_change_min(val):
        nonlocal min_adj, max_adj
        min_adj = val
        max_adj = max(max_adj, val)
        window['-MAXADJUST-'].update(max_adj)
        update()

    def on_change_max(val):
        nonlocal min_adj, max_adj
        max_adj = val
        min_adj = min(min_adj, val)
        window['-MINADJUST-'].update(min_adj)
        update()

    def on_change_gamma(val):
        nonlocal gamma
        gamma = val
        update_post_rotate()

    def on_change_gamma_weight(val):
        nonlocal gamma_weight
        gamma_weight = val
        update_post_rotate()

    def on_change_radius(val):
        nonlocal crop_radius
        crop_radius = val
        update()

    def on_change_rotation(val):
        nonlocal rotation
        rotation = val
        update_post_enhance()

    def on_change_clip(val):
        nonlocal min_clip
        min_clip = val
        update()

    def on_change_colorize(val):
        nonlocal show_colorized
        show_colorized = val
        update_post_rotate()

    def on_enhance_change(val):
        nonlocal should_enhance
        should_enhance = val
        update()

    def on_crop_change(val):
        nonlocal should_crop
        should_crop = val
        update()

    def on_change_zoom(val):
        nonlocal zoom
        zoom = val
        update_post_rotate()

    def on_change_red(val):
        rgb_weights[0] = val
        update_post_rotate()

    def on_change_green(val):
        rgb_weights[1] = val
        update_post_rotate()

    def on_change_blue(val):
        rgb_weights[2] = val
        update_post_rotate()

    def on_change_horiz_flip(val):
        nonlocal h_flip, src, src_center
        src = flip_image(src, h_flip != val, False)
        src_center = flip_center(src_center, src, h_flip != val, False)
        h_flip = val
        update()

    def on_change_vert_flip(val):
        nonlocal v_flip, src, src_center
        src = flip_image(src, False, v_flip != val)
        src_center = flip_center(src_center, src, False, v_flip != val)
        v_flip = val
        update()

    def do_align(date):
        nonlocal v_flip, h_flip, rotation, src_center, src
        old_v, old_h = v_flip, h_flip
        v_flip = False
        h_flip, rotation = align_image(src16_unflipped, date, False)
        src = flip_image(src, h_flip != old_h, v_flip != old_v)
        src_center = flip_center(src_center, src, h_flip != old_h, v_flip != old_v)
        window['-VFLIP-'].update(v_flip)
        window['-HFLIP-'].update(h_flip)
        window['-ROTATION-'].update(rotation)
        cv.destroyAllWindows()
        update()

    # this is the expensive part
    def update_enhance():
        nonlocal enhanced
        c1, im1 = center_and_expand(src_center, src)
        c2, im2 = crop_to_dist(im1, c1, radius * crop_radius) if should_crop else (src_center, src)
        enhanced = cnrgf_enhance(im2, c2, 6, min_adj, max_adj, min_clip, None, None, "") if should_enhance else im2

    def make_image_window(win_size, controls):
        image_col = sg.Column([[sg.Image(key='-IMAGE-')]], size=win_size, expand_x=True, expand_y=True, scrollable=True,
                              key='-SCROLLABLE-')
        return sg.Window('SolarFinish ', [[image_col, sg.Column(controls)]], resizable=True, finalize=True)

    def update_image(win, im):
        win['-IMAGE-'].update(size=(im.shape[1], im.shape[0]), data=cv.imencode('.ppm', im)[1].tobytes())
        win.refresh()
        win['-SCROLLABLE-'].contents_changed()

    def update_post_enhance():
        nonlocal enhanced_rot
        enhanced_rot = rotate(enhanced, (enhanced.shape[1] // 2, enhanced.shape[0] // 2), rotation)
        update_post_rotate()

    # this is the cheap part to update
    def update_post_rotate():
        nonlocal enhanced_rot
        if show_colorized:
            # shrunk = shrink(enhanced_rot, 5)  # shrink to make this go faster
            # adjusted_gamma = find_gamma_for_colorized(shrunk, gamma, gamma_weight, 0.5, 1.25, 3.75)
            # brighten_for_color = brighten(enhanced_rot, adjusted_gamma, gamma_weight)
            brighten_for_color = brighten(enhanced_rot, gamma, gamma_weight)
            enhance8 = colorize8_rgb(brighten_for_color, rgb_weights[0], rgb_weights[1], rgb_weights[2])
        else:
            brightened_for_gray = brighten(enhanced_rot, gamma, gamma_weight)
            enhance8 = colorize8_rgb(brightened_for_gray, 1, 1, 1)

        update_image(window, swap_rb(zoom_image(enhance8, zoom)))

    def make_command_line_string(gamma, gamma_weight, min_adj, max_adj, should_enhance, crop_radius, should_crop,
                                 h_flip, v_flip, rotation, min_clip, show_colorized, rgb_weights):
        hv = ("h" if h_flip else "") + ("v" if v_flip else "")
        flip = " --flip " + hv if h_flip or v_flip else ""
        enhance_val = str(min_adj) + ',' + str(max_adj) if should_enhance else 'no'
        crop_val = str(crop_radius) if should_crop else 'no'
        colorize_val = f"{rgb_weights[0]},{rgb_weights[1]},{rgb_weights[2]}" if show_colorized else 'no'
        return f"--brighten {gamma} --brightenweight {gamma_weight} --enhance {enhance_val} --crop {crop_val}{flip} --rotate {rotation} --darkclip {min_clip} --colorize {colorize_val}"

    def write_result(gray16_result, color16_result, output_directory, filename, suffix):
        out_fn = output_directory + '/' + os.path.basename(filename)  # replace input dir without output dir
        write_image(color16_result, out_fn, "enhancedcolor" + suffix)
        write_image(gray16_result, out_fn, "enhancedgray" + suffix)

    # split updating the image into cheap and expensive parts, to allow faster refresh
    def update():
        update_enhance()
        update_post_enhance()

    def post_load_image(src16_unflipped, filename, filename_or_url):
        # find the solar disk circle
        (is_valid, src_center, radius) = find_valid_circle(src16_unflipped)
        if not is_valid:
            print("Error: Couldn't find valid circle for input solar disk!", flush=True)
            return None

        window.set_title('SolarFinish ' + filename_or_url)
        src = to_float01_from_16bit(src16_unflipped)
        src = flip_image(src, h_flip, v_flip)
        src_center = flip_center(src_center, src, h_flip, v_flip)
        return is_valid, filename, src16_unflipped, src, src_center, radius

    def load_image(filename_or_url):
        src16_unflipped, filename = fetch_image(filename_or_url)
        return post_load_image(src16_unflipped, filename, filename_or_url)

    def save_as_image(im, fn, suffix):
        out_fn = make_filename_for_write(fn, suffix)
        out_fn = popup_get_file(True, output_directory, out_fn)
        if out_fn != '':
            print(f"writing: {out_fn}", flush=True)
            try:
                cv.imwrite(out_fn, im)
            except Exception as error:
                print(f"Error: Failed to save {out_fn}, likely bad file extension. Try .PNG\n{error}", flush=True)

    def load():
        nonlocal is_valid, filename, src16_unflipped, src, src_center, radius, output_directory
        fn = popup_get_file(False, directory, "")
        print(f"loading: {fn}", flush=True)
        result = load_image(fn)
        if result is not None:
            output_directory = os.path.dirname(fn)
            is_valid, filename, src16_unflipped, src, src_center, radius = result
            update()

    def save():
        command_line = make_command_line_string(gamma, gamma_weight, min_adj, max_adj, should_enhance, crop_radius,
                                                should_crop, h_flip, v_flip, rotation, min_clip, show_colorized,
                                                rgb_weights)
        print("\nCommand line equivalent to adjusted parameters:", flush=True)
        print(f"    SolarFinish {command_line}\n", flush=True)

        result_ims = process_image(src16_unflipped, should_enhance, min_adj, max_adj, gamma, gamma_weight, should_crop,
                                   crop_radius, min_clip, h_flip, v_flip, rotation, False, filename, True, rgb_weights)

        if result is not None:
            gray16_result, color16_result = result_ims
            out_fn = output_directory + '/' + os.path.basename(filename)  # replace input dir without output dir
            if show_colorized:
                save_as_image(color16_result, out_fn, "enhancedcolor" + suffix)
            else:
                save_as_image(gray16_result, out_fn, "enhancedgray" + suffix)
            cv.destroyAllWindows()

    rotation %= 360.0
    enhanced_rot = enhanced = np.zeros(0)
    zoom = 33
    if not should_align:
        align_date = datetime.datetime.today().strftime('%Y-%m-%d')

    layout = [[sg.Button('Quit', enable_events=True, key='Exit'),
               sg.Button('Load', enable_events=True, key='-LOAD-'),
               sg.Button('Save', enable_events=True, key='-SAVE-')],
              [sg.Checkbox('Contrast Enhance (CNRGF)', should_enhance, enable_events=True, key='-ENHANCE-')],
              [sg.Text('MinAdjust', size=(12, 1)),
               sg.Slider(range=(0.5, 5.0), resolution=0.05, default_value=min_adj, expand_x=True, enable_events=True,
                         orientation='h', key='-MINADJUST-')],
              [sg.Text('MaxAdjust', size=(12, 1)),
               sg.Slider(range=(0.5, 5.0), resolution=0.05, default_value=max_adj, expand_x=True, enable_events=True,
                         orientation='h', key='-MAXADJUST-')],
              [sg.Text('Gamma', size=(12, 1)),
               sg.Slider(range=(0.1, 3.0), resolution=0.025, default_value=gamma, expand_x=True, enable_events=True,
                         orientation='h', key='-GAMMA-')],
              [sg.Text('GammaWeight', size=(12, 1)),
               sg.Slider(range=(0.0, 1.0), resolution=0.05, default_value=gamma_weight, expand_x=True,
                         enable_events=True, orientation='h', key='-GAMMAWEIGHT-')],
              [sg.Text('DarkClip', size=(12, 1)),
               sg.Slider(range=(0.0, 0.5), resolution=0.001, default_value=min_clip, expand_x=True, enable_events=True,
                         orientation='h', key='-DARKCLIP-')],
              [sg.Checkbox('Center and Crop', should_crop, enable_events=True, key='-CROP-')],
              [sg.Text('CropRadius', size=(12, 1)),
               sg.Slider(range=(1.0, 2.5), resolution=0.05, default_value=crop_radius, expand_x=True,
                         enable_events=True, orientation='h', key='-CROPRADIUS-')],
              [sg.Text('Rotation', size=(12, 1)),
               sg.Slider(range=(0.0, 360.0), resolution=0.1, default_value=rotation, expand_x=True, enable_events=True,
                         orientation='h', key='-ROTATION-')],
              [sg.Checkbox('Horizontal Flip', default=h_flip, enable_events=True, key='-HFLIP-'),
               sg.Checkbox('Vertical Flip', default=v_flip, enable_events=True, key='-VFLIP-')],
              [sg.Checkbox('Colorize', default=show_colorized, enable_events=True, key='-COLORIZE-')],
              [sg.Text('Colorize Red', size=(12, 1)),
               sg.Slider(range=(0.0, 6.0), resolution=0.05, default_value=rgb_weights[0], expand_x=True,
                         enable_events=True, orientation='h', key='-RED-')],
              [sg.Text('Colorize Green', size=(12, 1)),
               sg.Slider(range=(0.0, 6.0), resolution=0.05, default_value=rgb_weights[1], expand_x=True,
                         enable_events=True, orientation='h', key='-GREEN-')],
              [sg.Text('Colorize Blue', size=(12, 1)),
               sg.Slider(range=(0.0, 6.0), resolution=0.05, default_value=rgb_weights[2], expand_x=True,
                         enable_events=True, orientation='h', key='-BLUE-')],
              [sg.HorizontalSeparator()],
              [sg.CalendarButton('Align Date:', close_when_date_chosen=True, target='-DATE-', format='%Y-%m-%d', ),
               sg.Input(key='-DATE-', size=(15, 1), default_text=align_date),
               sg.Button('Align', enable_events=True, key='-ALIGN-'), sg.Text("Takes a minute!")],
              [sg.HorizontalSeparator()],
              [sg.Text('Zoom', size=(12, 1)),
               sg.Slider(range=(10, 300), resolution=1, default_value=zoom, expand_x=True, enable_events=True,
                         orientation='h', key='-ZOOM-')],
              ]
    window = make_image_window((500, 500), layout)

    result = load_image(filename_or_url)
    if result is None:
        print(f"Error: Failed to load {filename_or_url}", flush=True)
        sys.exit(0)
    is_valid, filename, src16_unflipped, src, src_center, radius = result
    update()

    callbacks = {'-RED-': on_change_red, '-GREEN-': on_change_green, '-BLUE-': on_change_blue,
                 '-ENHANCE-': on_enhance_change, '-CROP-': on_crop_change,
                 '-MINADJUST-': on_change_min, '-MAXADJUST-': on_change_max, '-GAMMA-': on_change_gamma,
                 '-GAMMAWEIGHT-': on_change_gamma_weight, '-DARKCLIP-': on_change_clip,
                 '-CROPRADIUS-': on_change_radius,
                 '-ROTATION-': on_change_rotation, '-COLORIZE-': on_change_colorize, '-ZOOM-': on_change_zoom,
                 '-HFLIP-': on_change_horiz_flip, '-VFLIP-': on_change_vert_flip}
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == '-ALIGN-':
            do_align(values['-DATE-'])
        elif event == '-SAVE-':
            save()
        elif event == '-LOAD-':
            load()
        elif event in callbacks:
            callbacks[event](values[event])
    window.close()
    sys.exit(0)


#
# main - drive the high-level flow

# Utility function to differentiate between local file and remote URL
def is_url(filename_or_url):
    prefix = filename_or_url[:6]
    return prefix == "https:" or prefix == "http:/"


# Fetch an image as 16 bit grayscale, given a local filename or URL
# also returns the filename on disk, in case image came from URL
def fetch_image(filename_or_url):
    fn = filename_or_url

    # if it's a URL, download it
    if is_url(filename_or_url):
        fn = "tempsolarimage.tif"
        open(fn, 'wb').write(requests.get(filename_or_url, allow_redirects=True).content)

    # force to single-channel 16-bit grayscale
    src = force16_gray(read_image(fn))
    return src, fn


# if you flip an image, the center needs flipping, too
def flip_center(center, im, horiz, vert):
    cx = im.shape[1] - center[0] - 1 if horiz else center[0]
    cy = im.shape[0] - center[1] - 1 if vert else center[1]
    return cx, cy


# Flip an image, horizontally, vertically, or both
def flip_image(im, horiz, vert):
    if horiz:
        im = cv.flip(im, 1)
    if vert:
        im = cv.flip(im, 0)
    return im


# High-level flow to align a single image, given a date to compare against
def align_image(im, date, silent):
    if not silent:
        print(
            f"Aligning with GONG image from {date}. This might take a minute. Displaying input image before alignment:",
            flush=True)
        show_float01(to_float01_from_16bit(im))

    date = date.replace('-', '/')
    best = find_best_alignment(im, date, silent)
    percent, angle, flipped, sim, gong_big, gong, input_big, inp = best
    angle %= 360.0
    if not silent:
        print(f"Displaying GONG image used for alignment:", flush=True)
        show_float01(gong)
        print(
            f"\nAlignment result: angle is {angle}{' and horizontally flipped' if flipped else ''}. Equivalent to command line args:",
            flush=True)
        print(f"     --rotate {angle} {'--flip h' if flipped else ''}", flush=True)

    return flipped, angle


# TODO decide if I want to try to adjust gamma for colorization, probably not
# def find_gamma_for_colorized(im, gamma, weight, r_weight, g_weight, b_weight):
#     goal = np.mean(brighten(im, gamma, weight))
#     lo, hi = 0, 1
#     adjusted_gamma = gamma
#     while hi-lo > 0.01:
#         adjusted_gamma = (lo+hi)*0.5
#         b = brighten(im, adjusted_gamma, weight)
#         c = colorize_float_bgr(b, r_weight, g_weight, b_weight)
#         g = cv.cvtColor(c, cv.COLOR_BGR2GRAY)
#         a = np.mean(g)
#         if a > goal:
#             lo = adjusted_gamma
#         else:
#             hi = adjusted_gamma
#     return adjusted_gamma


# Process a single image, with optional verbose output.
def process_image(src, should_enhance, min_recip, max_recip, brighten_gamma, gamma_weight, should_crop, crop_radius,
                  min_clip, h_flip, v_flip, rotation, interact, fn, silent, rgb_weights):
    if h_flip or v_flip:
        src = flip_image(src, h_flip, v_flip)

    if rotation != 0.0:
        src = rotate_with_expand_fill(src, rotation)

    # find the solar disk circle
    (is_valid, src_center, radius) = find_valid_circle(src)
    if not is_valid:
        print("Error: Couldn't find valid circle for input solar disk! " + fn, flush=True)
        return None

    if not silent:
        # show original image as uploaded
        print(f"\nDisplaying input image, size is {src.shape[1]},{src.shape[0]}:", flush=True)
        show_float01(to_float01_from_16bit(src))

        # show image with circle drawn
        image_with_circle = add_circle(gray2rgb(src), src_center, radius, (1, 0, 0), 3)
        solar_radius_in_km = 695700
        print(
            f"Circle found with radius {radius} and center {src_center[0]},{src_center[1]}. Pixel size is about {solar_radius_in_km / radius:.1f}km. Displaying sun with circle found--should be very close to the edge of the photosphere.",
            flush=True)
        show_rgb(image_with_circle)

    if should_crop:
        # use an expanded/centered grayscale 0-1 float image for all calculations
        (center, centered) = center_and_expand(src_center, src)
        img = to_float01_from_16bit(centered)

        enhanced = cnrgf_enhance(img, center, 6, min_recip, max_recip, min_clip, None, None,
                                 fn) if should_enhance else img
        dist = min(crop_radius * radius, calc_min_dist_to_edge(src_center, src.shape))
        (center, enhanced) = crop_to_dist(enhanced, center, dist)
    else:
        src = to_float01_from_16bit(src)
        enhanced = cnrgf_enhance(src, src_center, 6, min_recip, max_recip, min_clip, None, None,
                                 fn) if should_enhance else src

    # brighten and colorize
    orig_enhanced = enhanced
    grayscale_result = brighten(enhanced, brighten_gamma, gamma_weight) if brighten_gamma != 1.0 else enhanced

    # shrunk = shrink(orig_enhanced, 5)  # shrink to make this go faster
    # adjusted_gamma = find_gamma_for_colorized(shrunk, brighten_gamma, gamma_weight, 0.5, 1.25, 3.75)
    # brighten_for_color = brighten(orig_enhanced, adjusted_gamma, gamma_weight)
    brighten_for_color = brighten(orig_enhanced, brighten_gamma, gamma_weight)
    color16_result = colorize16_bgr(brighten_for_color, rgb_weights[0], rgb_weights[1], rgb_weights[2])

    if not silent and not interact:
        print("Displaying grayscale enhanced result:", flush=True)
        show_float01(grayscale_result)
        download_button(float01_to_16bit(grayscale_result), fn, "enhancedgraybright")

        print("Displaying colorized enhanced result:", flush=True)
        enhance8 = colorize8_rgb(enhanced, rgb_weights[0], rgb_weights[1], rgb_weights[2])
        show_rgb(enhance8)
        download_button(color16_result, fn, "enhancedcolor")
    return float01_to_16bit(grayscale_result), color16_result


# Popup a filepicker - can be for loading or saving. Return filename or ""
def popup_get_file(save_as, folder, default):
    if save_as:
        types = (("PNG File", "*.png"), ("Tiff File", "*.tif"), ("Tiff File", "*.tiff"))
        filename = sg.popup_get_file('', save_as=save_as, no_window=True, initial_folder=folder, default_path=default,
                                 default_extension=".png", file_types=types)
    else:
        filename = sg.popup_get_file('', save_as=save_as, no_window=True, initial_folder=folder, default_path=default)
    return filename


# Process command line arguments to get parameters, and get list of files to process
def process_args():
    fn_list = []
    parser = argparse.ArgumentParser(description='Process solar images')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
    parser.add_argument('-t', '--type', type=str, default='tif', help='filetype to go along with -d, defaults to tif')
    parser.add_argument('-p', '--pattern', type=str, default='', help='String pattern to match for -d')
    parser.add_argument('-o', '--output', type=str, nargs='?', help='Output directory')
    parser.add_argument('-s', '--silent', action='store_true', help='run silently')
    parser.add_argument('-a', '--append', action='store_true',
                        help='append the settings used for gamma, min contrast, and max contrast as part of the output filename')
    parser.add_argument('-f', '--flip', type=str, default='', choices=['h', 'v', 'hv'],
                        help='rotate final images horizontally, vertically, or both')
    parser.add_argument('-g', '--gongalign', type=str, default='',
                        help='Date of GONG image to compare for auto-align, YYYY-MM-DD')
    parser.add_argument('-b', '--brighten', type=float, default=0.7,
                        help='gamma value to brighten by, 1 = none, 0.1 = extreme bright, 2.0 darken')
    parser.add_argument('-w', '--brightenweight', type=float, default=0.5,
                        help='weight to shift gamma brightening, 1 = use gamma curve, 0 = less brightening of darker pixels')
    parser.add_argument('-e', '--enhance', type=str, default='1.5,3.0',
                        help='contrast enhance min,max or no. 1 = no enhance, 5 = probably too much.')
    parser.add_argument('-c', '--crop', type=str, default="1.4", help='final crop radius in solar radii. Or no')
    parser.add_argument('-r', '--rotate', type=float, default=0.0, help='rotation in degrees')
    parser.add_argument('-d', '--darkclip', type=float, default=0.015,
                        help='clip minimum after contrast enhancement and before normalization')
    parser.add_argument('-i', '--interact', default=False, action='store_true', help='interactively adjust parameters')
    default_colorize = '0.5,1.25,3.75'
    parser.add_argument('-k', '--colorize', type=str, default=default_colorize,
                        help='R,G,B weights for colorization or no')
    # parser.add_argument('-x', '--imagealign', type=str, nargs='?', help='file or URL for image to use for alignment')
    parser.add_argument('filename', nargs='?', type=str, help='Image file to process')

    args = parser.parse_args()
    directory = '.'
    silent = args.silent
    if args.filename:
        if os.path.isdir(args.filename):
            directory = args.filename
            silent = True
            fn_list = [fn for fn in os.listdir(directory) if fn.endswith(args.type) and re.search(args.pattern, fn)]
        elif os.path.isfile(args.filename):
            if os.path.isabs(args.filename):
                directory = os.path.dirname(args.filename)
                fn_list = [os.path.basename(args.filename)]
            else:
                fn_list = [args.filename]
        elif is_url(args.filename):
            fn_list = [args.filename]
    elif not IN_COLAB:
        args.interact = True  # no file specified, force interactive mode
        fn = popup_get_file(False, directory, "")
        if os.path.isfile(fn):
            directory = os.path.dirname(fn)
            fn_list = [os.path.basename(fn)]
            print(f"Selected image: {fn}", flush=True)

    if len(fn_list) == 0 or fn_list[0] == "":
        fn_list = [""]
        print(f"No input image files found, using sample image", flush=True)

    if not args.output:
        output = directory
    else:
        output = args.output

    should_colorize = args.colorize.lower()[0:2] != 'no'
    if should_colorize:
        rgb_weights = [float(f) for f in args.colorize.split(",")]
    if not should_colorize or len(rgb_weights) != 3:
        rgb_weights = [float(f) for f in default_colorize.split(",")]
    should_enhance = args.enhance.lower()[0:2] != 'no'
    min_contrast_adjust, max_contrast_adjust = [float(f) for f in args.enhance.split(",")] if should_enhance else [1, 1]
    h_flip = 'h' in args.flip
    v_flip = 'v' in args.flip
    should_crop = args.crop.lower()[0:2] != 'no'
    crop = float(args.crop) if should_crop else 1.4
    return fn_list, silent, directory, h_flip, v_flip, output, args.append, args.gongalign, args.brighten, args.brightenweight, should_enhance, min_contrast_adjust, max_contrast_adjust, should_crop, crop, args.rotate, args.darkclip, args.interact, should_colorize, rgb_weights  # , args.imagealign


def main():
    print(
        '''
SolarFinish is an early work in progress. Originally built for my personal 
use, I've incorporated contributions from others and tried to make it more 
generally useful so others can benefit from easier post-processing with 
better results. Feel free to use, but I make no promises. It may fail on 
images from cameras and telescopes different from my own. Expect it to 
continue to evolve, and don't expect much tech support.
''', flush=True)

    min_contrast_adjust = 1.7  # @param {type:"number"}   # 1.6
    max_contrast_adjust = 3.0  # @param {type:"number"}   # 4.0
    brighten_gamma = 0.5  # @param {type:"number"}        # 0.7
    gamma_weight = 0.5  # @param {type:"number")          # 0.5
    crop_radius = 1.4  # @param {type:"number")           # 1.4
    dark_clip = 0.015  # @param {type:"number")           # 0.015
    rotation = 0.0  # @param {type:"number")              # 0.0
    should_align_first = False  # @param {type:"boolean"}
    date_if_aligning = "2023-12-17"  # @param {type:"date"}
    should_use_url = False  # @param {type:"boolean"}
    url_to_use = "https://www.cloudynights.com/uploads/monthly_01_2023/post-412916-0-66576300-1674591059.jpg"  # @param{type:"string"}
    fallback_url = 'https://www.cloudynights.com/uploads/gallery/album_24182/gallery_79290_24182_1973021.png'
    should_enhance = True
    should_crop = True
    rgb_weights = [0.5, 1.25, 3.75]

    # get the parameters
    if IN_COLAB:
        fn_list, silent, directory, h_flip, v_flip, output_directory, append, gong_align_date, interact = (
            [""], False, ".", False, False, ".", False, "", False)
        print("Upload full disk solar image now, or click cancel to use default test image")
        fn_list[0] = url_to_use if should_use_url else upload_file()
    else:
        fn_list, silent, directory, h_flip, v_flip, output_directory, append, gong_align_date, brighten_gamma, gamma_weight, should_enhance, min_contrast_adjust, max_contrast_adjust, should_crop, crop_radius, rotation, dark_clip, interact, should_colorize, rgb_weights = process_args()

    suffix = f"minc_{str(min_contrast_adjust)}_maxc_{str(max_contrast_adjust)}_g{str(brighten_gamma)}" if append else ""
    if gong_align_date != "":
        should_align_first = True

    if fn_list[0] == "":
        fn_list[0] = fallback_url

    if len(fn_list) > 1 or silent:
        interact = False
    elif interact and not IN_COLAB:
        full_name = fn_list[0] if is_url(fn_list[0]) else directory + '/' + fn_list[0]
        interactive_adjust(full_name, directory, output_directory, suffix, silent, should_enhance, min_contrast_adjust,
                           max_contrast_adjust, brighten_gamma, gamma_weight, dark_clip, should_crop,
                           crop_radius, h_flip, v_flip, rotation, should_colorize, rgb_weights, gong_align_date,
                           should_align_first)
        sys.exit(0)

    for fn in fn_list:
        full_name = fn if is_url(fn) else directory + '/' + fn
        src, filename = fetch_image(full_name)

        if should_align_first:
            v_flip = False
            h_flip, rotation = align_image(src, gong_align_date, silent)

        result = process_image(src, should_enhance, min_contrast_adjust, max_contrast_adjust, brighten_gamma,
                               gamma_weight, should_crop, crop_radius, dark_clip, h_flip, v_flip, rotation,
                               interact, filename, silent, rgb_weights)

        if result is not None and not IN_COLAB:
            gray16_result, color16_result = result
            out_fn = output_directory + '/' + os.path.basename(filename)  # replace input dir without output dir
            if should_colorize:
                write_image(color16_result, out_fn, "enhancedcolor" + suffix)
            write_image(gray16_result, out_fn, "enhancedgray" + suffix)
            cv.destroyAllWindows()


main()
