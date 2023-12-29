__copyright__ = "Copyright (C) 2023 Greg Alt"

# Version 0.11 - Added more command line args, for all parameters. Also expanded
#                interactive mode with more sliders.
# Version 0.10 - Fixed some bugs in command line args
# Version 0.09 - Added batch mode and command line args from kraegar
# Version 0.08 - Refactored main() to better incorporate batch mode ability
# Version 0.07 - Added ability to auto-align with GONG image, given a date. Also
#                added checkbox to load from URL.
# Version 0.06 - Better circle finding for large images, and for GONG images with
#                extra halo clipped by image boundary
# Version 0.05 - Added interactive adjustment when running locally.
# Version 0.04 - Same python can be run in both colab and local command line. Also
#                adjusted colorization gamma values to better match grayscale
# Version 0.03 - Expand instead of crop before processing to minimize banding, then
#                crop at then end
# Version 0.02 - More code cleanup and commenting, plus fixed 8-bit inputs
# Version 0.01 - Switched from median to mean, simplifies things and speeds up
#                processing without noticeable artifacts. Also, generally cleaned
#                up the script.

# TODOS        - cleanup of functions responsible for main flow, moving towards a chain
#                of optional filter tools.
#              - breakup/cleanup into multiple files (might mean abandoning Colab?)
#              - more attention to removing artifacts/noise beyond limb
#              - better control over min/max contrast adjustment params. Most flexible
#                would be 4 params for min/max input and min/max output
#              - better sub-pixel circle finding, and shifting before processing
#              - how to allow more continuous brightness of filaproms across limb?

try:
    from google.colab import files
    IN_COLAB = True
except:
    IN_COLAB = False
    import argparse
    import os
    import re

import math
import numpy as np
import cv2 as cv
import scipy as sp
import matplotlib.pyplot as plt
import ipywidgets as widgets
import requests
import sys
import datetime
import urllib.request
import astropy.io.fits
import io
from contextlib import redirect_stdout


#
# Circle finding

def is_valid_circle(shape, radius):
    size = min(shape[0], shape[1])
    if 2 * radius > size or 2 * radius < 0.25 * size:
        return False
    return True


def get_circle_data(ellipse):
    if ellipse is None:
        return (0, 0), 0
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    radius = int(0.5 + ellipse[0][2] + (ellipse[0][3] + ellipse[0][4]) * 0.5)
    return center, radius


def is_valid_ellipse(shape, ellipse):
    (center, radius) = get_circle_data(ellipse)
    return is_valid_circle(shape, radius)


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

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def gray2rgb(im):
    return cv.merge([im, im, im])


def colorize16_bgr(result, r, g, b):
    bgr = (np.power(result, b), np.power(result, g), np.power(result, r))
    return float01_to_16bit(cv.merge(bgr))


def colorize8_rgb(im, r, g, b):
    rgb = (np.power(im, r), np.power(im, g), np.power(im, b))
    return float01_to_8bit(cv.merge(rgb))


def force16_gray(im):
    im = rgb2gray(im) if len(im.shape) > 2 else im
    return cv.normalize(im.astype(np.uint16), None, 0, 65535, cv.NORM_MINMAX)


def gray16_to_rgb8(im):
    return gray2rgb((im / 256).astype(np.uint8))


# convert image from float 0-1 to 16bit uint, works with grayscale or RGB
def float01_to_16bit(im):
    return (im * 65535).astype(np.uint16)


# convert image from float 0-1 to 8bit uint, works with grayscale or RGB
def float01_to_8bit(im):
    return (im * 255).astype(np.uint8)


def to_float01_from_16bit(im):
    return im.astype(np.float32) / 65535.0


#
# Image filtering and warp/un-warping

def rotate(im, center, angle_deg):
    rows, cols = im.shape[0:2]
    m = cv.getRotationMatrix2D(center, angle_deg, 1)
    return cv.warpAffine(im, m, (cols, rows))


def polar_warp(img):
    return cv.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), img.shape[0], cv.WARP_FILL_OUTLIERS)


def polar_unwarp(img, shape):
    # INTER_AREA works best to remove artifacts
    # INTER_CUBIC works well except for a horizontal line artifact at angle = 0
    # INTER_LINEAR, the default has a very noticeable vertical banding artifact across the top, and similarly around the limb
    unwarped = cv.linearPolar(img, (shape[0] / 2, shape[1] / 2), shape[0],
                              cv.WARP_FILL_OUTLIERS | cv.WARP_INVERSE_MAP | cv.INTER_AREA)
    return unwarped


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


# pad the image on top and bottom to allow filtering with simulated wraparound
def pad_for_wrap_around(inp, pad):
    return cv.vconcat([inp[inp.shape[0] - pad:, :], inp, inp[:pad, :]])


# remove padding from top and bottom
def remove_wrap_around_pad(input_padded, pad):
    return input_padded[pad:input_padded.shape[0] - pad, :]


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

def brighten(im, gamma, gamma_weight):
    return gamma_weight * np.power(im, gamma) + (1 - gamma_weight) * (1 - np.power(1 - im, 1 / gamma))


def swap_rb(im):
    blue = im[:, :, 0].copy()
    im[:, :, 0] = im[:, :, 2].copy()
    im[:, :, 2] = blue
    return im


def shrink(im, div):
    return cv.resize(im, np.floor_divide((im.shape[1], im.shape[0]), div))


def add_circle(im, center, radius, color, thickness):
    cv.ellipse(im, center, (radius, radius), 0, 0, 360, color, thickness, cv.LINE_AA)
    return im


# Create an expanded image centered on the sun. Ensure that a bounding circle
# centered on the sun and enclosing the original image's four corners is fully
# enclosed in the resulting image. For added pixels, pad by copying the existing
# edge pixels. This means that processing of the polar-warped image has reasonable
# values out to the maximum distance included in the original source image. This,
# in turn, means that circular banding artifacts will occur farther out and can be
# fully cropped out at the end.
def center_and_expand(center, src):
    to_left, to_right = (center[0], src.shape[1] - center[0])
    to_top, to_bottom = (center[1], src.shape[0] - center[1])
    to_ul = math.sqrt(to_top * to_top + to_left * to_left)
    to_ur = math.sqrt(to_top * to_top + to_right * to_right)
    to_bl = math.sqrt(to_bottom * to_bottom + to_left * to_left)
    to_br = math.sqrt(to_bottom * to_bottom + to_right * to_right)
    max_dist = int(max(to_ul, to_ur, to_bl, to_br)) + 1
    new_center = (max_dist, max_dist)
    out_img = np.pad(src, ((max_dist - to_top, max_dist - to_bottom), (max_dist - to_left, max_dist - to_right)),
                     mode='edge')
    return new_center, out_img


def crop_to_dist(src, center, min_dist):
    min_dist = min(math.ceil(min_dist), src.shape[0] // 2)  # don't allow a crop larger than the image
    new_center = (min_dist, min_dist)
    # note, does NOT force to odd
    out_img = src[center[1] - min_dist:center[1] + min_dist, center[0] - min_dist:center[0] + min_dist]
    return new_center, out_img


def calc_min_dist_to_edge(center, shape):
    to_left, to_right = (center[0], shape[1] - center[0])
    to_top, to_bottom = (center[1], shape[0] - center[1])
    min_dist = int(min(to_left, to_top, to_right, to_bottom)) - 1
    return min_dist


def center_and_crop(center, src):
    return crop_to_dist(src, center, calc_min_dist_to_edge(center, src.shape))


def force_radius(im, center, rad, new_rad):
    scale = new_rad / rad
    im2 = cv.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))
    center2 = (int(center[0] * scale), int(center[1] * scale))
    return center2, im2


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
    # orientation component #
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

    # frequency component #
    # go to polar space
    raw = np.sqrt(x ** 2 + y ** 2)
    # set origin to 1 as in the log space zero is not defined
    raw[mid, mid] = 1
    # go to log space
    raw = np.log2(raw)

    center_scale = math.log2(n) - f_0
    draw = raw - center_scale
    frequency_component = np.exp(-0.5 * (draw / scale_bandwidth) ** 2)

    # reset origin to zero (not needed as it is already 0?)
    frequency_component[mid, mid] = 0

    kernel = frequency_component * orientation_component
    return kernel


# simpler function to do both fft and shift
def fft(im):
    return np.fft.fftshift(np.fft.fft2(im))


# simpler function to do both inverse fft and shift
def ifft(f):
    return np.real(np.fft.ifft2(np.fft.ifftshift(f)))


# create the frequency space filter image for all orientations
def get_lgs(n, f_0, num_orientations):
    return [get_log_gabor_filter(n, f_0, x, num_orientations) for x in range(0, num_orientations)]


def apply_filter(im, lg):
    # apply fft to go to frequency space, apply filter, then inverse fft to go back to spatial
    # take absolute value, so we have only non-negative values, and merge into multichannel image
    f = [np.abs(ifft(fft(im) * lg[x])) for x in range(0, len(lg))]
    im = cv.merge(f)
    return im


#
# Implementation of algorithm in Aligning 'Dissimilar' Images Directly

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


def calc_n(rij):
    abs_rij = abs(rij)
    # c = 2#1 # what value for constant?
    # n = 1/(1+np.power((1-abs_rij)/(1+abs_rij), c/2))
    n = 1 / (1 + ((1 - abs_rij) / (1 + abs_rij)))  # optimized for c=2, power can be removed
    return n


def get_similarity_sum(n):
    n_flat = np.zeros(n.shape[:2])
    for i in range(n.shape[2]):
        n_flat = n_flat + n[:, :, i]
    h = np.sum(n)
    return h, n_flat


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

def show_eval(_gong_filtered, input_rot, _input_filtered, _h, n, _angle):
    if not IN_COLAB:
        cv.imshow("input_rot", input_rot)
        cv.imshow("n", n * 0.5 - 1.0)
        cv.waitKey(1)


def local_search_evaluate(inp, lg, mask, gong_filtered, angle, show_eval_func):
    input_rot = rotate(inp, (inp.shape[1] // 2, inp.shape[0] // 2), angle)
    input_filtered = apply_filter(input_rot, lg) * mask
    (H, n) = similarity(gong_filtered, input_filtered)
    if show_eval_func is not None:
        show_eval_func(gong_filtered, input_rot, input_filtered, H, n, angle)
    return H, n, input_rot, input_filtered


# start with a rough peak, with 3 data points, iterate until narrow enough and return final peak
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


def center_and_crop_to_fixed_radius(center, radius, im, pixel_radius, solar_radii):
    # ensure we have enough buffer, scale to get fixed radius, then crop
    (center, im) = center_and_expand(center, im)
    (center, im) = force_radius(im, center, radius, pixel_radius)
    (center, im) = crop_to_dist(im, center, pixel_radius * solar_radii)
    return center, im


# assumes global search has already been done, supplying is_flipped and initial 3 results in triple
def local_search_helper(inp, is_flipped, lg, mask, gong, gong_filtered, triple, silent):
    input_search = cv.flip(inp, 1) if is_flipped else inp
    show_eval_func = None if silent else show_eval
    (angle, sim) = local_search(input_search, lg, mask, gong_filtered, triple, 0.1, show_eval_func)
    return angle, sim, is_flipped, True, triple, gong, input_search


# does a full search including global and local
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

def center_images_for_alignment(inp, gong, fixed_radius, solar_radii):
    (is_valid_gong, gong_center, gong_radius) = find_valid_circle(gong)
    if not is_valid_gong:
        print("Error: Couldn't find valid circle for GONG solar disk!")

    (is_valid_input, input_center, input_radius) = find_valid_circle(inp)
    if not is_valid_input:
        print("Error: Couldn't find valid circle for input solar disk!")

    if is_valid_gong and is_valid_input:
        (gong_center, gong) = center_and_crop_to_fixed_radius(gong_center, gong_radius, to_float01_from_16bit(gong),
                                                              fixed_radius,
                                                              solar_radii)
        (input_center, inp) = center_and_crop_to_fixed_radius(input_center, input_radius, to_float01_from_16bit(inp),
                                                              fixed_radius,
                                                              solar_radii)

    return is_valid_gong, gong, gong_center, is_valid_input, inp, input_center


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


# do a single experiment with one image and one percent
def align(inp, date, percent, triple, flipped, silent, search_func):
    gong = get_gong_image_for_date(datetime.datetime.strptime(date, '%Y/%m/%d'), percent)
    (angle, sim, flipped, matchFound, triple, gong_out, input_out) = align_images(gong, inp, 128, triple, flipped,
                                                                                  silent, search_func)
    return angle, sim, triple, flipped, gong, gong_out, inp, input_out


# given an image and a date, find the best angle
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

def get_image_fz(url):
    trap = io.StringIO()
    with redirect_stdout(trap):  # hide the obnoxious progress bar
        image_data = astropy.io.fits.getdata(url)
    img_float = image_data.astype(np.float32).clip(min=0) / np.max(image_data)
    img = float01_to_16bit(img_float)
    return cv.flip(img, 0)  # image co-ords are upside down


def get_image(url):
    fn = "testalign.png"
    open(fn, 'wb').write(requests.get(url, allow_redirects=True).content)
    return force16_gray(read_image(fn))  # force to single-channel 16-bit grayscale


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


def get_gong_image_for_date(date, percent):
    # return getImage(GetGongImageURL(date, percent, "https://gong2.nso.edu/HA/hag/", '.jpg'))
    return get_image_fz(get_gong_image_url(date, percent, "https://nispdata.nso.edu/ftp/HA/haf/", 's.fz'))


def read_image(fn):
    return cv.imread(cv.samples.findFile(fn), cv.IMREAD_UNCHANGED | cv.IMREAD_ANYDEPTH)


def upload_file():
    keys = list(files.upload().keys())
    return keys[0] if keys else ""


def write_image(im, fn, suffix):
    # strip full path after the last .
    without_extension = fn[::-1].split('.', 1)[1][::-1]  # reverse, split first ., take 2nd part, reverse again
    out_fn = without_extension + '-' + suffix + '.png'
    cv.imwrite(out_fn, im)
    return out_fn


def download_image(im, fn, suffix):
    files.download(write_image(im, fn, suffix))


def download_button(im, fn, suffix):
    if IN_COLAB:
        button = widgets.Button(description='Download Image')
        button.on_click(lambda x: download_image(im, fn, suffix))
        display(button)


def show_rgb(im):
    if IN_COLAB:
        plt.imshow(im)
        plt.show()
    else:
        cv.imshow(str(id(im)), swap_rb(shrink(im, 3)))
        cv.waitKey(1)


def show_float01(im):
    show_rgb(colorize8_rgb(im, 1, 1, 1))


#
# Enhancement

def display_intermediate_results(polar_image, mean_image, unwarped_mean, diff, norm_std_dev, enhance_factor, enhanced,
                                 fn,
                                 silent):
    print("Polar warp the image as an initial step to make a pseudo-flat")
    if not silent:
        show_float01(polar_image)

    print("Mean filter on polar warp image")
    if not silent:
        show_float01(mean_image)

    print("Finally un-warp the mean image to get the pseudo-flat:")
    if not silent:
        show_float01(unwarped_mean)
    download_button(float01_to_16bit(unwarped_mean), fn, "unwarpedmean")

    print("Subtract pseudo-flat from image:")
    if not silent:
        show_float01(diff + 0.5)
    download_button(float01_to_16bit(diff + 0.5), fn, "diff")

    print("Result of standard deviation filter, to drive contrast enhancement")
    if not silent:
        show_float01(norm_std_dev)

    print("Enhanced contrast in diff image:")
    if not silent:
        show_float01((diff * enhance_factor + 0.5).clip(min=0, max=1))
    download_button(float01_to_16bit((diff * enhance_factor + 0.5).clip(min=0, max=1)), fn, "diff-enhanced")

    print("Enhance contrast and add back to pseudo-flat:")
    if not silent:
        show_float01(enhanced)
    download_button(float01_to_16bit(enhanced), fn, "enhancedgray")


# Do full enhancement from start to finish in one function, displaying intermediate
# results. Takes a float 0-1 image with a centered solar disk.
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
def cnrgf_enhance(img, min_recip, max_recip, fn, min_clip, silent):
    # find mean and standard deviation image from polar-warped image, then un-warp
    polar_image = polar_warp(img)
    (mean_image, std_devs) = get_mean_and_std_dev_image(polar_image, 6)
    unwarped_mean = polar_unwarp(mean_image, img.shape)
    unwarped_std_dev = polar_unwarp(std_devs, img.shape)

    # adjust range of standard deviation image to get preferred range of contrast enhancement
    norm_std_dev = cv.normalize(unwarped_std_dev, None, 1 / max_recip, 1 / min_recip, cv.NORM_MINMAX)

    # subtract mean, divide by standard deviation, and add back mean
    enhance_factor = np.reciprocal(norm_std_dev)
    diff = img - unwarped_mean
    enhanced = diff * enhance_factor + unwarped_mean

    # final normalize and clip
    enhanced = enhanced.clip(min=min_clip)  # don't want sunspot pixels blowing up the normalization
    enhanced = cv.normalize(enhanced, None, 0, 1, cv.NORM_MINMAX).clip(min=0).clip(max=1)

    display_intermediate_results(polar_image, mean_image, unwarped_mean, diff, norm_std_dev, enhance_factor, enhanced,
                                 fn,
                                 silent)
    return enhanced


# returns a function that normalizes to within a given range
def get_std_dev_scaler(min_recip, max_recip):
    return lambda sd: cv.normalize(sd, None, 1 / max_recip, 1 / min_recip, cv.NORM_MINMAX)


# CNRGF split into two parts, first part does expensive compute
def cnrgf_enhance_part1(img, n):
    (mean_image, std_devs) = get_mean_and_std_dev_image(polar_warp(img), n)
    return polar_unwarp(mean_image, img.shape), polar_unwarp(std_devs, img.shape)


# CNRGF split into two parts, second part is cheaper and has tunable parameters
# using scaleStdDev as a function that has tunable parameters baked into it.
def cnrgf_enhance_part2(img, mean_and_stddev, scale_std_dev):
    (unwarped_mean, unwarped_std_dev) = mean_and_stddev
    norm_std_dev = scale_std_dev(unwarped_std_dev)
    return (img - unwarped_mean) * np.reciprocal(norm_std_dev) + unwarped_mean


# CNRGF combining the two parts in one go
def enhance(img, n, min_recip, max_recip, min_clip):
    mean_and_stddev = cnrgf_enhance_part1(img, n)
    e = cnrgf_enhance_part2(img, mean_and_stddev, get_std_dev_scaler(min_recip, max_recip))
    return cv.normalize(e.clip(min=min_clip), None, 0, 1, cv.NORM_MINMAX).clip(min=0).clip(max=1)


#
# Interactive

def interactive_adjust(img, center, radius, _dist_to_edge, min_adj, max_adj, gamma, gamma_weight, min_clip, crop_radius,
                       rotation):
    def on_change_min(val):
        nonlocal min_adj
        min_adj = 1.0 + val / 10.0
        update()

    def on_change_max(val):
        nonlocal max_adj
        max_adj = 1.0 + val / 10.0
        update()

    def on_change_gamma(val):
        nonlocal gamma
        gamma = val / 100.0
        update_post_enhance()

    def on_change_gamma_weight(val):
        nonlocal gamma_weight
        gamma_weight = val / 100.0
        update_post_enhance()

    def on_change_quadrant(val):
        nonlocal quadrant
        quadrant = val
        update()

    def on_change_radius(val):
        nonlocal crop_radius
        crop_radius = 1.0 + val / 50.0
        update()

    def on_change_rotation(val):
        nonlocal rotation
        rotation = val / 10.0
        update_post_enhance()

    def update_enhance():
        (new_center, new_img) = crop_to_dist(img, center, radius * crop_radius)
        im = shrink(new_img, 3) if quadrant == 0 else new_img
        nonlocal enhanced
        enhanced = enhance(im, 6, min_adj, max_adj, 0.01)

    def update_post_enhance():
        nonlocal enhanced
        if quadrant == 0:
            im = enhanced
        else:
            h = enhanced.shape[0]
            r = (quadrant - 1) // 2
            c = (quadrant - 1) % 2
            im = enhanced[r * (h // 2):r * (h // 2) + h // 2, c * (h // 2):c * (h // 2) + h // 2]
        brightened = brighten(im, gamma, gamma_weight)
        enhance8 = swap_rb(colorize8_rgb(brightened, 0.5, 1.25, 3.75))
        enhance8 = rotate(enhance8, (enhance8.shape[1] // 2, enhance8.shape[0] // 2), rotation - init_rotation)
        cv.imshow('adjust', enhance8)

    def update():
        update_enhance()
        update_post_enhance()

    print("starting interactive")
    rotation %= 360.0
    init_rotation = rotation
    quadrant = 0
    enhanced = None
    update()
    cv.createTrackbar('min adjust', 'adjust', 7, 100, on_change_min)
    cv.createTrackbar('max adjust', 'adjust', 30, 100, on_change_max)
    cv.createTrackbar('gamma', 'adjust', int(100 * gamma), 100, on_change_gamma)
    cv.createTrackbar('gammaweight', 'adjust', int(100 * gamma_weight), 100, on_change_gamma_weight)
    cv.createTrackbar('cropradius', 'adjust', int(50 * 0.2), 100, on_change_radius)
    cv.createTrackbar('quadrant', 'adjust', 0, 4, on_change_quadrant)
    cv.createTrackbar('rotation', 'adjust', int(10 * rotation), 3600, on_change_rotation)
    cv.waitKey(0)
    return min_adj, max_adj, gamma, gamma_weight, min_clip, crop_radius, rotation


#
# main - drive the high-level flow

def is_url(filename_or_url):
    prefix = filename_or_url[:6]
    return prefix == "https:" or prefix == "http:/"


# fetch an image as 16 bit grayscale, given a local filename or URL
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


def flip_image(im, horiz, vert):
    if horiz:
        im = cv.flip(im, 1)
    if vert:
        im = cv.flip(im, 0)
    return im


# align a single image, given a date to compare against
def align_image(im, date, silent):
    if not silent:
        print(f"Original image before alignment:")
        show_float01(to_float01_from_16bit(im))
        print(f"Aligning with GONG image from {date}. This might take a minute.")

    date = date.replace('-', '/')
    best = find_best_alignment(im, date, silent)
    percent, angle, flipped, sim, gong_big, gong, input_big, inp = best

    if not silent:
        flipped_text = 'and horizontally flipped' if flipped else ''
        print(f"Best angle is {angle} {flipped_text}")
        print(f"GONG image used for alignment:")
        show_float01(gong)

    if flipped:
        im = cv.flip(im, 1)
    im = sp.ndimage.rotate(im, angle)
    return im


# process a single image, silently
def silent_process_image(src, min_recip, max_recip, brighten_gamma, gamma_weight, crop_radius, min_clip):
    (is_valid, src_center, radius) = find_valid_circle(src)
    if not is_valid:
        return None

    # use an expanded/centered grayscale 0-1 float image for all calculations
    (center, centered) = center_and_expand(src_center, src)
    img = to_float01_from_16bit(centered)

    enhanced = cnrgf_enhance(img, min_recip, max_recip, "", min_clip, True)
    dist = min(crop_radius * radius, calc_min_dist_to_edge(src_center, src.shape))
    (center, enhanced) = crop_to_dist(enhanced, center, dist)

    # brighten and colorize
    enhanced = brighten(enhanced, brighten_gamma, gamma_weight)
    enhance16 = colorize16_bgr(enhanced, 0.5, 1.25, 3.75)
    return enhance16


# process a single image, with verbose output
def process_image(src, min_recip, max_recip, brighten_gamma, gamma_weight, crop_radius, min_clip, rotation, fn):
    # find the solar disk circle
    (is_valid, src_center, radius) = find_valid_circle(src)
    if not is_valid:
        print("Couldn't find valid circle for solar disk!")
        return None

    # show original image as uploaded
    print(
        f"\nOriginal image size: {src.shape[1]},{src.shape[0]}  Circle found with radius {radius} and center {src_center[0]},{src_center[1]}")
    show_float01(to_float01_from_16bit(src))

    # use an expanded/centered grayscale 0-1 float image for all calculations
    (center, centered) = center_and_expand(src_center, src)
    img = to_float01_from_16bit(centered)
    print(f"centered image size: {centered.shape[1]},{centered.shape[0]}  New center: {center[0]},{center[1]}")
    show_float01(img)
    download_button(centered, fn, "centered")

    # show image with circle drawn
    image_with_circle = add_circle(gray2rgb(img), center, radius, (1, 0, 0), 3)
    solar_radius_in_km = 695700
    print(
        f"centered image with solar limb circle highlighted. Circle should be very close to the edge of the photosphere. Pixel size is about {solar_radius_in_km / radius:.1f}km")
    show_rgb(image_with_circle)
    image_with_circle = add_circle(gray2rgb(img), center, radius, (0, 0, 1), 1)
    download_button(float01_to_16bit(image_with_circle), fn, "withcircle")

    init_rotation = rotation
    if not IN_COLAB:
        dist_to_edge = calc_min_dist_to_edge(src_center, src.shape)
        params = interactive_adjust(img, center, radius, dist_to_edge, min_recip, max_recip, brighten_gamma,
                                    gamma_weight,
                                    min_clip, crop_radius, rotation)
        (min_recip, max_recip, brighten_gamma, gamma_weight, min_clip, crop_radius, rotation) = params
        print(
            f"Command line:\nSolarFinish --brighten {brighten_gamma} --brightenweight {gamma_weight} --enhance {min_recip},{max_recip} --crop {crop_radius} --rotate {rotation} --darkclip {min_clip}\n")

    enhanced = cnrgf_enhance(img, min_recip, max_recip, fn, min_clip, False)
    if init_rotation != rotation:
        enhanced = rotate(enhanced, (enhanced.shape[1] // 2, enhanced.shape[0] // 2), rotation - init_rotation)
    dist = min(crop_radius * radius, calc_min_dist_to_edge(src_center, src.shape))
    (center, enhanced) = crop_to_dist(enhanced, center, dist)

    # brighten and colorize
    enhanced = brighten(enhanced, brighten_gamma, gamma_weight)
    print("Brighten image:")
    show_float01(enhanced)
    download_button(float01_to_16bit(enhanced), fn, "enhancedgraybright")

    enhance8 = colorize8_rgb(enhanced, 0.5, 1.25, 3.75)
    enhance16 = colorize16_bgr(enhanced, 0.5, 1.25, 3.75)
    print("And finally colorize image:")
    show_rgb(enhance8)
    download_button(enhance16, fn, "enhancedcolor")
    return enhance16


# process a single image - from filename or URL
def image_main(filename_or_url, silent, h_flip, v_flip, should_align, date, min_contrast_adjust, max_contrast_adjust,
               brighten_gamma, gamma_weight, crop_radius, dark_clip, rotation):
    src, filename = fetch_image(filename_or_url)
    src = flip_image(src, h_flip, v_flip)

    if should_align:
        src = align_image(src, date, silent)
    elif rotation != 0.0:
        src = sp.ndimage.rotate(src, rotation)

    if silent:
        enhance16 = silent_process_image(src, min_contrast_adjust, max_contrast_adjust, brighten_gamma, gamma_weight,
                                         crop_radius, dark_clip)
    else:
        enhance16 = process_image(src, min_contrast_adjust, max_contrast_adjust, brighten_gamma, gamma_weight,
                                  crop_radius,
                                  dark_clip, rotation, filename)

    return enhance16, filename


# process command line arguments to get parameters, and get list of files to process
def process_args():
    fn_list = []
    parser = argparse.ArgumentParser(description='Process solar images')
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
                        help='contrast enhance min,max. 1 = no enhance, 5 = probably too much')
    parser.add_argument('-c', '--crop', type=float, default=1.4, help='final crop radius in solar radii')
    parser.add_argument('-r', '--rotate', type=float, default=0.0, help='rotation in degrees')
    parser.add_argument('-d', '--darkclip', type=float, default=0.015,
                        help='clip minimum after contrast enhancement and before normalization')
    # parser.add_argument('-i', '--imagealign', type=str, nargs='?', help='file or URL for image to use for alignment')
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

    if len(fn_list) == 0:
        print(f"No files found, using sample image")
        fn_list.append("")

    if not args.output:
        output = directory
    else:
        output = args.output

    min_contrast_adjust, max_contrast_adjust = [float(f) for f in args.enhance.split(",")]
    h_flip = 'h' in args.flip
    v_flip = 'v' in args.flip
    return fn_list, silent, directory, h_flip, v_flip, output, args.append, args.gongalign, args.brighten, args.brightenweight, min_contrast_adjust, max_contrast_adjust, args.crop, args.rotate, args.darkclip  # , args.imagealign


def main():
    print(
        '''
This solar image processing python script is an experimental work in progress.
It is intended for my own use, but I'm sharing so others can look at the code
and give feedback. Feel free to use, but I make no promises. It is likely that
it will fail on images from cameras and telescopes different from my own. Expect
it to change without warning as I continue to tinker with it, and do not expect
tech support.

I've tried to make it as parameter-free as possible, but some important
parameters are still hardcoded in the script.

All that said, it has generated some compelling results from a range of
different inputs, so give it a shot.
''')

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

    # get the solar disk image
    if IN_COLAB:
        fn_list, silent, directory, h_flip, v_flip, output_directory, append, gong_align_date = [
                                                                                                    ""], False, ".", False, False, ".", False, ""
        print("Upload full disk solar image now, or click cancel to use default test image")
        fn_list[0] = url_to_use if should_use_url else upload_file()
    else:
        fn_list, silent, directory, h_flip, v_flip, output_directory, append, gong_align_date, brighten_gamma, gamma_weight, min_contrast_adjust, max_contrast_adjust, crop_radius, rotation, dark_clip = process_args()

    suffix = f"minc_{str(min_contrast_adjust)}_maxc_{str(max_contrast_adjust)}_g{str(brighten_gamma)}" if append else ""
    if gong_align_date != "":
        should_align_first = True
        date_if_aligning = gong_align_date

    if fn_list[0] == "":
        fn_list[0] = fallback_url

    for fn in fn_list:
        full_name = fn if is_url(fn) else directory + '/' + fn
        enhance16, out_fn = image_main(full_name, silent, h_flip, v_flip, should_align_first, date_if_aligning,
                                       min_contrast_adjust, max_contrast_adjust, brighten_gamma, gamma_weight,
                                       crop_radius, dark_clip, rotation)
        if not IN_COLAB:
            out_fn = output_directory + '/' + os.path.basename(out_fn)  # replace input dir without output dir
            write_image(enhance16, out_fn, "enhancedcolor" + suffix)
            write_image(cv.cvtColor(enhance16, cv.COLOR_BGR2GRAY), out_fn, "enhancedgray" + suffix)
            cv.waitKey(0)
            cv.destroyAllWindows()


main()
