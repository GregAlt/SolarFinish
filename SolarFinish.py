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
#                processing without noticeable artifcacts. Also generally cleaned
#                up the script.

# TODOS        - cleanup of variable/function names to be consistent and match coding
#                standards
#              - cleanup of functions responsible for main flow, moving towards a chain
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

def is_valid_circle(shape, center, radius):
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
    return is_valid_circle(shape, center, radius)


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

    # find ellipse with biggest max axis
    return ellipses[np.array([e[0][2] + max(e[0][3], e[0][4]) for e in ellipses]).argmax()]


def find_valid_circle(src):
    (center, radius) = get_circle_data(find_circle(src))
    if not is_valid_circle(src.shape, center, radius):
        # try shrinking image
        thousands = math.ceil(min(src.shape[0], src.shape[1]) / 1000)
        for scale in range(2, thousands + 1):
            smaller = cv.resize(src, (int(src.shape[1] / scale), int(src.shape[0] / scale)))
            (smallcenter, smallradius) = get_circle_data(find_circle(smaller))
            (center, radius) = ((smallcenter[0] * scale + scale // 2, smallcenter[1] * scale + scale // 2),
                                smallradius * scale + scale // 2)
            if is_valid_circle(src.shape, center, radius):
                break
    return (True, center, radius) if is_valid_circle(src.shape, center, radius) else (False, None, None)


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
# Image filtering and warp/unwarping

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
    lefthalf = polar_image[:, 0:h // 2]  # left half is circle of radius h//2
    (mean, stddev) = mean_and_std_dev_filt2d_with_wraparound(lefthalf, (k, 1))

    # don't use mean filter for corners, just copy that data directly to minimize artifacts
    righthalf = polar_image[:, h // 2:]  # right half is corners and beyond
    meanimage = cv.hconcat([mean, righthalf])

    # don't use stddev filter for corners, just repeat last column to minimize artifacts
    stddevimage = np.hstack((stddev, np.tile(stddev[:, [-1]], h - h // 2)))
    return meanimage, stddevimage


# pad the image on top and bottom to allow filtering with simulated wraparound
def pad_for_wrap_around(inp, pad):
    return cv.vconcat([inp[inp.shape[0] - pad:, :], inp, inp[:pad, :]])


# remove padding from top and bottom
def remove_wrap_around_pad(input_padded, pad):
    return input_padded[pad:input_padded.shape[0] - pad, :]


def mean_and_std_dev_filt2d_with_wraparound(inp, kernel_size):
    # pad input image with half of kernel to simulate wraparound
    imagepad = pad_for_wrap_around(inp, kernel_size[0] // 2)

    # filter the padded image
    meanpad = sp.ndimage.uniform_filter(imagepad, kernel_size, mode='reflect')
    meanofsquaredpad = sp.ndimage.uniform_filter(imagepad * imagepad, kernel_size, mode='reflect')

    # sqrt(meanofsquared - mean*mean) is mathematically equivalent to std dev:
    #   https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows
    stddevpad = np.sqrt((meanofsquaredpad - meanpad * meanpad).clip(min=0))

    mean = remove_wrap_around_pad(meanpad, kernel_size[0] // 2)
    stddev = remove_wrap_around_pad(stddevpad, kernel_size[0] // 2)
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
    toleft, toright = (center[0], src.shape[1] - center[0])
    totop, tobottom = (center[1], src.shape[0] - center[1])
    toUL = math.sqrt(totop * totop + toleft * toleft)
    toUR = math.sqrt(totop * totop + toright * toright)
    toBL = math.sqrt(tobottom * tobottom + toleft * toleft)
    toBR = math.sqrt(tobottom * tobottom + toright * toright)
    maxdist = int(max(toUL, toUR, toBL, toBR)) + 1
    newcenter = (maxdist, maxdist)
    outimg = np.pad(src, ((maxdist - totop, maxdist - tobottom), (maxdist - toleft, maxdist - toright)), mode='edge')
    return newcenter, outimg


def crop_to_dist(src, center, mindist):
    mindist = min(math.ceil(mindist), src.shape[0] // 2)  # don't allow a crop larger than the image
    newcenter = (mindist, mindist)
    # note, does NOT force to odd
    outimg = src[center[1] - mindist:center[1] + mindist, center[0] - mindist:center[0] + mindist]
    return newcenter, outimg


def calc_min_dist_to_edge(center, shape):
    toleft, toright = (center[0], shape[1] - center[0])
    totop, tobottom = (center[1], shape[0] - center[1])
    mindist = int(min(toleft, totop, toright, tobottom)) - 1
    return mindist


def center_and_crop(center, src):
    return crop_to_dist(src, center, calc_min_dist_to_edge(center, src.shape))


def force_radius(im, center, rad, newrad):
    scale = newrad / rad
    im2 = cv.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))
    center2 = (int(center[0] * scale), int(center[1] * scale))
    return center2, im2


def get_disk_mask(src, center, radius):
    # create 32 bit float disk mask
    diskmask = np.zeros(src.shape[:2], dtype="float32")
    cv.ellipse(diskmask, center, (radius, radius), 0, 0, 360, 1.0, -1, cv.FILLED)  # no LINE_AA!
    return diskmask


#
# Functions need to evaluate alignment similarity, using log-gabor filter

# Log-Gabor filter
# from https://stackoverflow.com/questions/31774071/implementing-log-gabor-filter-bank/31796747
def get_log_gabor_filter(N, f_0, theta_0, number_orientations):
    # filter configuration
    scale_bandwidth = 0.996 * math.sqrt(2 / 3)
    angle_bandwidth = 0.996 * (1 / math.sqrt(2)) * (np.pi / number_orientations)

    # x,y grid
    extent = np.arange(-N / 2, N / 2 + N % 2)
    x, y = np.meshgrid(extent, extent)

    mid = int(N / 2)
    # orientation component #
    theta = np.arctan2(y, x)
    center_angle = ((np.pi / number_orientations) * theta_0) if (f_0 % 2) \
        else ((np.pi / number_orientations) * (theta_0 + 0.5))

    # calculate (theta-center_theta), we calculate cos(theta-center_theta)
    # and sin(theta-center_theta) then use atan to get the required value,
    # this way we can eliminate the angular distance wrap around problem
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    ds = sintheta * math.cos(center_angle) - costheta * math.sin(center_angle)
    dc = costheta * math.cos(center_angle) + sintheta * math.sin(center_angle)
    dtheta = np.arctan2(ds, dc)

    orientation_component = np.exp(-0.5 * (dtheta / angle_bandwidth) ** 2)

    # frequency component #
    # go to polar space
    raw = np.sqrt(x ** 2 + y ** 2)
    # set origin to 1 as in the log space zero is not defined
    raw[mid, mid] = 1
    # go to log space
    raw = np.log2(raw)

    center_scale = math.log2(N) - f_0
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
def get_lgs(N, f_0, num_orientations):
    return [get_log_gabor_filter(N, f_0, x, num_orientations) for x in range(0, num_orientations)]


def apply_filter(im, lg):
    # apply fft to go to freqency space, apply filter, then inverse fft to go back to spatial
    # take absolute value so we have only non-negative values, and merge into multi-channel image
    f = [np.abs(ifft(fft(im) * lg[x])) for x in range(0, len(lg))]
    im = cv.merge(f)
    return im


#
# Implementation of algorithm in Aligning 'Dissimilar' Images Directly

def get_rij(num, den, k):
    # TODO: cleanup this conditional code meant to exclude very small denominators
    denshape = den.shape
    den = den.flatten()
    num = num.flatten()
    rij = np.zeros(den.shape)
    epsilon = 0.0001 / (k * k)  # divide here because I simplified out k*k* out of den
    rij[den > epsilon] = num[den > epsilon] / den[den > epsilon]
    rij = np.reshape(rij, denshape)
    return rij


def calc_n(rij):
    abs_rij = abs(rij)
    # c = 2#1 # what value for constant?
    # n = 1/(1+np.power((1-abs_rij)/(1+abs_rij), c/2))
    n = 1 / (1 + ((1 - abs_rij) / (1 + abs_rij)))  # optimized for c=2, power can be removed
    return n


def get_similarity_sum(n):
    nflat = np.zeros(n.shape[:2])
    for i in range(n.shape[2]):
        nflat = nflat + n[:, :, i]
    H = np.sum(n)
    return H, nflat


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
    # r = 25*(boxfilter(im1*im2,5) + (-u1)*boxfilter(im2,5) + (-u2)*boxfilter(im1,5) + (u1*u2))
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
# local and global search of alignment based on simularity evaluation

def show_eval(gong_filtered, inpu_rot, inpu_filtered, H, n, angle):
    if not IN_COLAB:
        cv.imshow("inpu_rot", inpu_rot)
        cv.imshow("n", n * 0.5 - 1.0)
        cv.waitKey(1)


def local_search_evaluate(inpu, lg, mask, gong_filtered, angle, showEvalFunc):
    inpu_rot = rotate(inpu, (inpu.shape[1] // 2, inpu.shape[0] // 2), angle)
    inpu_filtered = apply_filter(inpu_rot, lg) * mask
    (H, n) = similarity(gong_filtered, inpu_filtered)
    if showEvalFunc is not None:
        showEvalFunc(gong_filtered, inpu_rot, inpu_filtered, H, n, angle)
    return H, n, inpu_rot, inpu_filtered


# start with a rough peak, with 3 data points, iterate until narrow enough and return final peak
def local_search(inpu, lg, mask, gong_filtered, triple, stopping, reportcallback):
    angles, Hs = triple
    while abs(angles[1] - angles[2]) > stopping:
        # evaluate midpoints
        ang = [angles[0], 0.5 * (angles[0] + angles[1]), angles[1], 0.5 * (angles[1] + angles[2]), angles[2]]
        results1 = local_search_evaluate(inpu, lg, mask, gong_filtered, ang[1], reportcallback)
        results2 = local_search_evaluate(inpu, lg, mask, gong_filtered, ang[3], reportcallback)
        sim = [Hs[0], results1[0], Hs[1], results2[0], Hs[2]]

        # find resulting peak and new triples
        peak = np.argmax(sim[1:5]) + 1  # first one can't be peak, but might be equal to peak
        # print(f"Local {ang[peak]}, {sim[peak]}, delta={ang[1]-ang[0]}")
        angles, Hs = (ang[peak - 1:peak + 2], sim[peak - 1:peak + 2])
    return angles[1], Hs[1]


def global_search(inpu, lg, mask, gong_filtered, start, end, count, bestangle, bestsim, showEvalFunc):
    angles = []
    Hs = []
    index = -1
    for angle in np.linspace(start, end, num=count, endpoint=False):
        inpu_rot = rotate(inpu, (inpu.shape[1] // 2, inpu.shape[0] // 2), angle)
        inpu_filtered = apply_filter(inpu_rot, lg) * mask
        (H, n) = similarity(gong_filtered, inpu_filtered)
        angles.append(angle)
        Hs.append(H)
        if showEvalFunc is not None:
            showEvalFunc(gong_filtered, inpu_rot, inpu_filtered, H, n, angle)

    peak = np.argmax(Hs)
    a = (peak - 1 + len(Hs)) % len(Hs)
    b = (peak + 1 + len(Hs)) % len(Hs)
    return [angles[a], angles[peak], angles[b]], [Hs[a], Hs[peak], Hs[b]]


def center_and_crop_to_fixed_radius(center, radius, im, pixelRadius, solarRadii):
    # ensure we have enough buffer, scale to get fixed radius, then crop
    (center, im) = center_and_expand(center, im)
    (center, im) = force_radius(im, center, radius, pixelRadius)
    (center, im) = crop_to_dist(im, center, pixelRadius * solarRadii)
    return center, im


# assumes global search has already been done, supplying isflipped and initial 3 results in triple
def local_search_helper(inpu, isflipped, lg, mask, gong, gong_filtered, triple, silent):
    inpu_search = cv.flip(inpu, 1) if isflipped else inpu
    showEvalFunc = None if silent else show_eval
    (angle, sim) = local_search(inpu_search, lg, mask, gong_filtered, triple, 0.1, showEvalFunc)
    return angle, sim, isflipped, True, triple, gong, inpu_search


# does a full search including global and local
def full_search_helper(inpu, isflipped, lg, mask, gong, gong_filtered, triple, silent):
    showEvalFunc = None if silent else show_eval
    triple = global_search(inpu, lg, mask, gong_filtered, -180, 180, 20, 0, -1, showEvalFunc)
    (unflippedAngle, unflippedSim) = local_search(inpu, lg, mask, gong_filtered, triple, 0.1, showEvalFunc)
    flipped = cv.flip(inpu, 1)
    flippedtriple = global_search(flipped, lg, mask, gong_filtered, -180, 180, 20, 0, -1, showEvalFunc)
    (flippedAngle, flippedSim) = local_search(flipped, lg, mask, gong_filtered, flippedtriple, 0.1, showEvalFunc)

    return (unflippedAngle, unflippedSim, False, True, triple, gong, inpu) if unflippedSim > flippedSim else (
        flippedAngle, flippedSim, True, True, flippedtriple, gong, flipped)


#
# high-level alignment implementation

def center_images_for_alignment(inpu, gong, fixedRadius, solarradii):
    (isValidGong, gongcenter, gongradius) = find_valid_circle(gong)
    if not isValidGong:
        print("Error: Couldn't find valid circle for GONG solar disk!")

    (isValidInpu, inpucenter, inpuradius) = find_valid_circle(inpu)
    if not isValidInpu:
        print("Error: Couldn't find valid circle for input solar disk!")

    if isValidGong and isValidInpu:
        (gongcenter, gong) = center_and_crop_to_fixed_radius(gongcenter, gongradius, to_float01_from_16bit(gong),
                                                             fixedRadius,
                                                             solarradii)
        (inpucenter, inpu) = center_and_crop_to_fixed_radius(inpucenter, inpuradius, to_float01_from_16bit(inpu),
                                                             fixedRadius,
                                                             solarradii)

    return isValidGong, gong, gongcenter, isValidInpu, inpu, inpucenter


def align_images(gong, inpu, fixedRadius, triple, isflipped, silent, searchfunc):
    (isValidGong, gong, gongcenter, isValidInpu, inpu, inpucenter) = center_images_for_alignment(inpu, gong,
                                                                                                 fixedRadius,
                                                                                                 1.1)
    if not isValidGong or not isValidInpu:
        return 0, 0, False, False, ([], []), gong, inpu

    if not silent and not IN_COLAB:
        cv.imshow("gong", gong)
        cv.waitKey(1)

    lg = get_lgs(gong.shape[0], 4, 4)
    mask = cv.merge([get_disk_mask(gong, gongcenter, int(fixedRadius * 0.8)) for _ in range(len(lg))])
    gong_filtered = apply_filter(gong, lg) * mask

    return searchfunc(inpu, isflipped, lg, mask, gong, gong_filtered, triple, silent)


# do a single experiment with one image and one percent
def align(inpu, date, percent, triple, flipped, silent, searchfunc):
    gong = get_gong_image_for_date(datetime.datetime.strptime(date, '%Y/%m/%d'), percent)
    (angle, sim, flipped, matchFound, triple, gongout, inpuout) = align_images(gong, inpu, 128, triple, flipped,
                                                                               silent, searchfunc)
    return angle, sim, triple, flipped, gong, gongout, inpu, inpuout


# given an image and a date, find the best angle
def find_best_alignment(input_i, date, silent):
    (angle, sim, triple, flipped, gong_big, gong, inpu_big, inpu) = align(input_i, date, 0.5, None, None, silent,
                                                                          full_search_helper)
    best = (0.5, angle, flipped, sim, gong_big, gong, inpu_big, inpu)
    percents = np.linspace(0.1, 0.9, 9).tolist()  # from 0.1 to 0.9, inclusive, by 0.1
    percents.remove(0.5)  # and remove 0.5
    for percent in percents:
        (angle, sim, triple, flipped, gong_big, gong, inpu_big, inpu) = align(input_i, date, percent, triple,
                                                                              flipped, silent,
                                                                              local_search_helper)
        if sim > best[3]:
            best = (percent, angle, flipped, sim, gong_big, gong, inpu_big, inpu)
    return best


#
# File uploading, downloading, and rendering

def get_image_fz(url):
    trap = io.StringIO()
    with redirect_stdout(trap):  # hide the obnoxious progress bar
        image_data = astropy.io.fits.getdata(url)
    img_float = image_data.astype(np.float32).clip(min=0) / np.max(image_data)
    img = float01_to_16bit(img_float)
    return cv.flip(img, 0)  # image coords are upside down


def get_image(url):
    fn = "testalign.png"
    open(fn, 'wb').write(requests.get(url, allow_redirects=True).content)
    return force16_gray(read_image(fn))  # force to single-channel 16-bit grayscale


def get_gong_image_url(date, percent, gongroot, fileend):
    yyyy = date.strftime("%Y")
    mm = date.strftime("%m")
    dd = date.strftime("%d")
    gongdir = gongroot + yyyy + mm + "/" + yyyy + mm + dd + "/"
    data = urllib.request.urlopen(gongdir).read()
    data2 = data.split(b'\"')
    w = [str(a)[2:-1] for a in data2 if str(a)[2:4] == '20' and str(a)[-5:-1].lower() == fileend]
    fn = w[int(len(w) * percent)]
    gongfullpath = gongdir + fn
    return gongfullpath


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
    withoutextension = fn[::-1].split('.', 1)[1][::-1]  # reverse, split first ., take 2nd part, reverse again
    outfn = withoutextension + '-' + suffix + '.png'
    cv.imwrite(outfn, im)
    return outfn


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

def display_intermediate_results(polar_image, meanimage, unwarpedmean, diff, normstddev, enhancefactor, enhanced, fn,
                                 silent):
    print("Polar warp the image as an initial step to make a pseudoflat")
    if not silent:
        show_float01(polar_image)

    print("Mean filter on polar warp image")
    if not silent:
        show_float01(meanimage)

    print("Finally unwarp the mean image to get the pseudoflat:")
    if not silent:
        show_float01(unwarpedmean)
    download_button(float01_to_16bit(unwarpedmean), fn, "unwarpedmean")

    print("Subtract pseudoflat from image:")
    if not silent:
        show_float01(diff + 0.5)
    download_button(float01_to_16bit(diff + 0.5), fn, "diff")

    print("Result of standard deviation filter, to drive contrast enhancement")
    if not silent:
        show_float01(normstddev)

    print("Enhanced contrast in diff image:")
    if not silent:
        show_float01((diff * enhancefactor + 0.5).clip(min=0, max=1))
    download_button(float01_to_16bit((diff * enhancefactor + 0.5).clip(min=0, max=1)), fn, "diff-enhanced")

    print("Enhance contrast and add back to pseudoflat:")
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
# polar warped image, and then unwarps those results. This allows for a fairly simple
# and fast python implementation with similar effect of adaptively applying enhancement
# and addressing the radial gradient. CNRGF was developed for processing full-disk
# hydrogen alpha images, while FNRGF was developed for coronal images beyond 1 solar
# radius, but the problems have many similarities and it should be possible to use the
# algorithms interchangeably for solar images more generally, including full disk
# white light images.
def cnrgf_enhance(img, minrecip, maxrecip, fn, minclip, silent):
    # find mean and standard deviation image from polar-warped image, then unwarp
    polar_image = polar_warp(img)
    (meanimage, stddevs) = get_mean_and_std_dev_image(polar_image, 6)
    unwarpedmean = polar_unwarp(meanimage, img.shape)
    unwarpedstddev = polar_unwarp(stddevs, img.shape)

    # adjust range of standard deviation image to get preferred range of contrast enhancement
    normstddev = cv.normalize(unwarpedstddev, None, 1 / maxrecip, 1 / minrecip, cv.NORM_MINMAX)

    # subtract mean, divide by standard deviation, and add back mean
    enhanceFactor = np.reciprocal(normstddev)
    diff = img - unwarpedmean
    enhanced = diff * enhanceFactor + unwarpedmean

    # final normalize and clip
    enhanced = enhanced.clip(min=minclip)  # don't want sunspot pixels blowing up the normalize
    enhanced = cv.normalize(enhanced, None, 0, 1, cv.NORM_MINMAX).clip(min=0).clip(max=1)

    display_intermediate_results(polar_image, meanimage, unwarpedmean, diff, normstddev, enhanceFactor, enhanced, fn,
                                 silent)
    return enhanced


# returns a function that normalizes to within a given range
def get_std_dev_scaler(minrecip, maxrecip):
    return lambda sd: cv.normalize(sd, None, 1 / maxrecip, 1 / minrecip, cv.NORM_MINMAX)


# CNRGF split into two parts, first part does expensive compute
def cnrgf_enhance_part1(img, n):
    (meanimage, stddevs) = get_mean_and_std_dev_image(polar_warp(img), n)
    return polar_unwarp(meanimage, img.shape), polar_unwarp(stddevs, img.shape)


# CNRGF split into two parts, second part is cheaper and has tunable parameters
# using scaleStdDev as a function that has tunable parameters baked into it.
def cnrgf_enhance_part2(img, mean_and_stddev, scaleStdDev):
    (unwarpedmean, unwarpedstddev) = mean_and_stddev
    normstddev = scaleStdDev(unwarpedstddev)
    return (img - unwarpedmean) * np.reciprocal(normstddev) + unwarpedmean


# CNRGF combining the two parts in one go
def enhance(img, n, minrecip, maxrecip, minclip):
    mean_and_stddev = cnrgf_enhance_part1(img, n)
    e = cnrgf_enhance_part2(img, mean_and_stddev, get_std_dev_scaler(minrecip, maxrecip))
    return cv.normalize(e.clip(min=minclip), None, 0, 1, cv.NORM_MINMAX).clip(min=0).clip(max=1)


#
# Interactive

def interactive_adjust(img, center, radius, disttoedge, minadj, maxadj, gamma, gammaweight, minclip, cropradius,
                       rotation):
    def on_change_min(val):
        nonlocal minadj
        minadj = 1.0 + val / 10.0
        update()

    def on_change_max(val):
        nonlocal maxadj
        maxadj = 1.0 + val / 10.0
        update()

    def on_change_gamma(val):
        nonlocal gamma
        gamma = val / 100.0
        update_post_enhance()

    def on_change_gamma_weight(val):
        nonlocal gammaweight
        gammaweight = val / 100.0
        update_post_enhance()

    def on_change_quadrant(val):
        nonlocal quadrant
        quadrant = val
        update()

    def on_change_radius(val):
        nonlocal cropradius
        cropradius = 1.0 + val / 50.0
        update()

    def on_change_rotation(val):
        nonlocal rotation
        rotation = val / 10.0
        update_post_enhance()

    def update_enhance():
        (newcenter, newimg) = crop_to_dist(img, center, radius * cropradius)
        im = shrink(newimg, 3) if quadrant == 0 else newimg
        nonlocal enhanced
        enhanced = enhance(im, 6, minadj, maxadj, 0.01)

    def update_post_enhance():
        nonlocal enhanced
        if quadrant == 0:
            im = enhanced
        else:
            h = enhanced.shape[0]
            r = (quadrant - 1) // 2
            c = (quadrant - 1) % 2
            im = enhanced[r * (h // 2):r * (h // 2) + h // 2, c * (h // 2):c * (h // 2) + h // 2]
        brightened = brighten(im, gamma, gammaweight)
        enhance8 = swap_rb(colorize8_rgb(brightened, 0.5, 1.25, 3.75))
        enhance8 = rotate(enhance8, (enhance8.shape[1] // 2, enhance8.shape[0] // 2), rotation - initrotation)
        cv.imshow('adjust', enhance8)

    def update():
        update_enhance()
        update_post_enhance()

    print("starting interactive")
    rotation %= 360.0
    initrotation = rotation
    quadrant = 0
    enhanced = None
    update()
    cv.createTrackbar('min adjust', 'adjust', 7, 100, on_change_min)
    cv.createTrackbar('max adjust', 'adjust', 30, 100, on_change_max)
    cv.createTrackbar('gamma', 'adjust', int(100 * gamma), 100, on_change_gamma)
    cv.createTrackbar('gammaweight', 'adjust', int(100 * gammaweight), 100, on_change_gamma_weight)
    cv.createTrackbar('cropradius', 'adjust', int(50 * 0.2), 100, on_change_radius)
    cv.createTrackbar('quadrant', 'adjust', 0, 4, on_change_quadrant)
    cv.createTrackbar('rotation', 'adjust', int(10 * rotation), 3600, on_change_rotation)
    cv.waitKey(0)
    return minadj, maxadj, gamma, gammaweight, minclip, cropradius, rotation


#
# main - drive the high-level flow

def is_url(filenameOrUrl):
    prefix = filenameOrUrl[:6]
    return prefix == "https:" or prefix == "http:/"


# fetch an image as 16 bit grayscale, given a local filename or URL
# also returns the filename on disk, in case image came from URL
def fetch_image(filenameOrUrl):
    fn = filenameOrUrl

    # if it's a URL, download it
    if is_url(filenameOrUrl):
        fn = "tempsolarimage.tif"
        open(fn, 'wb').write(requests.get(filenameOrUrl, allow_redirects=True).content)

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
    percent, angle, flipped, sim, gong_big, gong, inpu_big, inpu = best

    if not silent:
        flippedtext = 'and horizontally flipped' if flipped else ''
        print(f"Best angle is {angle} {flippedtext}")
        print(f"GONG image used for alignment:")
        show_float01(gong)

    if flipped:
        im = cv.flip(im, 1)
    im = sp.ndimage.rotate(im, angle)
    return im


# process a single image, silently
def silent_process_image(src, minrecip, maxrecip, brightengamma, gammaweight, cropradius, minclip):
    (isValid, srccenter, radius) = find_valid_circle(src)
    if not isValid:
        return None

    # use a expanded/centered grayscale 0-1 float image for all calculations
    (center, centered) = center_and_expand(srccenter, src)
    img = to_float01_from_16bit(centered)

    enhanced = cnrgf_enhance(img, minrecip, maxrecip, "", minclip, True)
    dist = min(cropradius * radius, calc_min_dist_to_edge(srccenter, src.shape))
    (center, enhanced) = crop_to_dist(enhanced, center, dist)

    # brighten and colorize
    enhanced = brighten(enhanced, brightengamma, gammaweight)
    enhance16 = colorize16_bgr(enhanced, 0.5, 1.25, 3.75)
    return enhance16


# process a single image, with verbose output
def process_image(src, minrecip, maxrecip, brightengamma, gammaweight, cropradius, minclip, rotation, fn):
    # find the solar disk circle
    (isValid, srccenter, radius) = find_valid_circle(src)
    if not isValid:
        print("Couldn't find valid circle for solar disk!")
        return None

    # show original image as uploaded
    print(
        f"\nOriginal image size: {src.shape[1]},{src.shape[0]}  Circle found with radius {radius} and center {srccenter[0]},{srccenter[1]}")
    show_float01(to_float01_from_16bit(src))

    # use a expanded/centered grayscale 0-1 float image for all calculations
    (center, centered) = center_and_expand(srccenter, src)
    img = to_float01_from_16bit(centered)
    print(f"centered image size: {centered.shape[1]},{centered.shape[0]}  New center: {center[0]},{center[1]}")
    show_float01(img)
    download_button(centered, fn, "centered")

    # show image with circle drawn
    imageWithCircle = add_circle(gray2rgb(img), center, radius, (1, 0, 0), 3)
    solarRadiusInKm = 695700
    print(
        f"centered image with solar limb circle highlighted. Circle should be very close to the edge of the photosphere. Pixel size is about {solarRadiusInKm / radius:.1f}km")
    show_rgb(imageWithCircle)
    imageWithCircle = add_circle(gray2rgb(img), center, radius, (0, 0, 1), 1)
    download_button(float01_to_16bit(imageWithCircle), fn, "withcircle")

    initrotation = rotation
    if not IN_COLAB:
        disttoedge = calc_min_dist_to_edge(srccenter, src.shape)
        params = interactive_adjust(img, center, radius, disttoedge, minrecip, maxrecip, brightengamma, gammaweight,
                                    minclip, cropradius, rotation)
        (minrecip, maxrecip, brightengamma, gammaweight, minclip, cropradius, rotation) = params
        print(
            f"Command line:\nSolarFinish --brighten {brightengamma} --brightenweight {gammaweight} --enhance {minrecip},{maxrecip} --crop {cropradius} --rotate {rotation} --darkclip {minclip}\n")

    enhanced = cnrgf_enhance(img, minrecip, maxrecip, fn, minclip, False)
    if initrotation != rotation:
        enhanced = rotate(enhanced, (enhanced.shape[1] // 2, enhanced.shape[0] // 2), rotation - initrotation)
    dist = min(cropradius * radius, calc_min_dist_to_edge(srccenter, src.shape))
    (center, enhanced) = crop_to_dist(enhanced, center, dist)

    # brighten and colorize
    enhanced = brighten(enhanced, brightengamma, gammaweight)
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
def image_main(filenameOrUrl, silent, hflip, vflip, should_align, date, mincontrastadjust, maxcontrastadjust, brightengamma,
               gammaweight, cropradius, darkclip, rotation):
    src, filename = fetch_image(filenameOrUrl)
    src = flip_image(src, hflip, vflip)

    if should_align:
        src = align_image(src, date, silent)
    elif rotation != 0.0:
        src = sp.ndimage.rotate(src, rotation)

    if silent:
        enhance16 = silent_process_image(src, mincontrastadjust, maxcontrastadjust, brightengamma, gammaweight,
                                         cropradius, darkclip)
    else:
        enhance16 = process_image(src, mincontrastadjust, maxcontrastadjust, brightengamma, gammaweight, cropradius,
                                  darkclip, rotation, filename)

    return enhance16, filename


# process command line arguments to get parameters, and get list of files to process
def process_args():
    fnlist = []
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
            fnlist = [fn for fn in os.listdir(directory) if fn.endswith(args.type) and re.search(args.pattern, fn)]
        elif os.path.isfile(args.filename):
            if os.path.isabs(args.filename):
                directory = os.path.dirname(args.filename)
                fnlist = [os.path.basename(args.filename)]
            else:
                fnlist = [args.filename]
        elif is_url(args.filename):
            fnlist = [args.filename]

    if len(fnlist) == 0:
        print(f"No files found, using sample image")
        fnlist.append("")

    if not args.output:
        output = directory
    else:
        output = args.output

    mincontrastadjust, maxcontrastadjust = [float(f) for f in args.enhance.split(",")]
    hflip = 'h' in args.flip
    vflip = 'v' in args.flip
    return fnlist, silent, directory, hflip, vflip, output, args.append, args.gongalign, args.brighten, args.brightenweight, mincontrastadjust, maxcontrastadjust, args.crop, args.rotate, args.darkclip  # , args.imagealign


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

    mincontrastadjust = 1.7  # @param {type:"number"}   # 1.6
    maxcontrastadjust = 3.0  # @param {type:"number"}   # 4.0
    brightengamma = 0.5  # @param {type:"number"}       # 0.7
    gammaweight = 0.5  # @param {type:"number")         # 0.5
    cropradius = 1.4  # @param {type:"number")          # 1.4
    darkclip = 0.015  # @param {type:"number")          # 0.015
    rotation = 0.0  # @param {type:"number")            # 0.0
    shouldAlignFirst = False  # @param {type:"boolean"}
    dateIfAligning = "2023-12-17"  # @param {type:"date"}
    shouldUseUrl = False  # @param {type:"boolean"}
    urlToUse = "https://www.cloudynights.com/uploads/monthly_01_2023/post-412916-0-66576300-1674591059.jpg"  # @param{type:"string"}
    fallbackUrl = 'https://www.cloudynights.com/uploads/gallery/album_24182/gallery_79290_24182_1973021.png'

    # get the solar disk image
    if IN_COLAB:
        fnlist, silent, directory, hflip, vflip, outputDirectory, append, gongAlignDate = [
                                                                                              ""], False, ".", False, False, ".", False, ""
        print("Upload full disk solar image now, or click cancel to use default test image")
        fnlist[0] = urlToUse if shouldUseUrl else upload_file()
    else:
        fnlist, silent, directory, hflip, vflip, outputDirectory, append, gongAlignDate, brightengamma, gammaweight, mincontrastadjust, maxcontrastadjust, cropradius, rotation, darkclip = process_args()

    suffix = f"minc_{str(mincontrastadjust)}_maxc_{str(maxcontrastadjust)}_g{str(brightengamma)}" if append else ""
    if gongAlignDate != "":
        shouldAlignFirst = True
        dateIfAligning = gongAlignDate

    if fnlist[0] == "":
        fnlist[0] = fallbackUrl

    for fn in fnlist:
        fullName = fn if is_url(fn) else directory + '/' + fn
        enhance16, outfn = image_main(fullName, silent, hflip, vflip, shouldAlignFirst, dateIfAligning,
                                      mincontrastadjust, maxcontrastadjust, brightengamma, gammaweight, cropradius,
                                      darkclip, rotation)
        if not IN_COLAB:
            outfn = outputDirectory + '/' + os.path.basename(outfn)  # replace input dir without output dir
            write_image(enhance16, outfn, "enhancedcolor" + suffix)
            write_image(cv.cvtColor(enhance16, cv.COLOR_BGR2GRAY), outfn, "enhancedgray" + suffix)
            cv.waitKey(0)
            cv.destroyAllWindows()


main()
