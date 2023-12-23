__copyright__ = "Copyright (C) 2023 Greg Alt"

## Version 0.10 - Fixed some bugs in command line args
## Version 0.09 - Added batch mode and command line args from kraegar
## Version 0.08 - Refactored main() to better incorporate batch mode ability
## Version 0.07 - Added ability to auto-align with GONG image, given a date. Also
##                added checkbox to load from URL.
## Version 0.06 - Better circle finding for large images, and for GONG images with
##                extra halo clipped by image boundary
## Version 0.05 - Added interactive adjustment when running locally.
## Version 0.04 - Same python can be run in both colab and local command line. Also
##                adjusted colorization gamma values to better match grayscale
## Version 0.03 - Expand instead of crop before processing to minimize banding, then
##                crop at then end
## Version 0.02 - More code cleanup and commenting, plus fixed 8-bit inputs
## Version 0.01 - Switched from median to mean, simplifies things and speeds up
##                processing without noticeable artifcacts. Also generally cleaned
##                up the script.

## TODOS        - better control over min/max contrast adjustment params. Most flexible
##                would be 4 params for min/max input and min/max output
##              - tunable final minclip, to hide noise beyond limb
##              - parameter for final crop, in solar radii, with reasonable default
##              - try implementing sliders for contrast and brightness params
##              - better sub-pixel circle finding, and shifting before processing
##              - how to allow more continuous brightness of filaproms across limb?

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

###
### Circle finding

def IsValidCircle(shape, center, radius):
    size = min(shape[0],shape[1])
    if 2*radius > size or 2*radius < 0.25*size:
        return False
    return True

def GetCircleData(ellipse):
    if ellipse is None:
      return ((0,0),0)
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    radius = int(0.5+ellipse[0][2]+(ellipse[0][3]+ellipse[0][4])*0.5)
    return(center, radius)

def IsValidEllipse(shape, ellipse):
    (center, radius) = GetCircleData(ellipse)
    return IsValidCircle(shape,center, radius)

def findCircle(src):
    # convert to 8bit grayscale for ellipse-detecting
    gray = (src / 256).astype(np.uint8)

    EDParams = cv.ximgproc_EdgeDrawing_Params()
    EDParams.MinPathLength = 300
    EDParams.PFmode = True
    EDParams.MinLineLength = 10
    EDParams.NFAValidation = False

    ed = cv.ximgproc.createEdgeDrawing()
    ed.setParams(EDParams)
    ed.detectEdges(gray)
    ellipses = ed.detectEllipses()

    if ellipses is None:
      return None

    # reject invalid ones *before* finding largest
    ellipses = [e for e in ellipses if IsValidEllipse(src.shape, e)]
    if len(ellipses) == 0:
      return None

    # find ellipse with biggest max axis
    return ellipses[np.array([e[0][2]+max(e[0][3],e[0][4]) for e in ellipses]).argmax()]

def findValidCircle(src):
  (center, radius) = GetCircleData(findCircle(src))
  if not IsValidCircle(src.shape, center, radius):
      # try shrinking image
      thousands = math.ceil(min(src.shape[0], src.shape[1]) / 1000)
      for scale in range(2,thousands+1):
        smaller = cv.resize(src, ((int)(src.shape[1]/scale), (int)(src.shape[0]/scale)))
        (smallcenter, smallradius) = GetCircleData(findCircle(smaller))
        (center, radius) = ((smallcenter[0]*scale+scale//2, smallcenter[1]*scale+scale//2), smallradius*scale+scale//2)
        if IsValidCircle(src.shape, center, radius):
          break
  return (True, center, radius) if IsValidCircle(src.shape, center, radius) else (False, None, None)

###
### Pixel format conversions

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def gray2rgb(im):
    return cv.merge([im,im,im])

def colorize16BGR(result, r,g,b):
    bgr = (np.power(result,b), np.power(result,g), np.power(result,r))
    return float01to16bit(cv.merge(bgr))

def colorize8RGB(im, r,g,b):
    rgb = (np.power(im,r), np.power(im,g), np.power(im,b))
    return float01to8bit(cv.merge(rgb))

def force16Gray(im):
  im = rgb2gray(im) if len(im.shape) > 2 else im
  return cv.normalize(im.astype(np.uint16), None, 0,65535, cv.NORM_MINMAX)

def gray16toRgb8(im):
  return gray2rgb((im/256).astype(np.uint8))

# convert image from float 0-1 to 16bit uint, works with grayscale or RGB
def float01to16bit(im):
  return (im*65535).astype(np.uint16)

# convert image from float 0-1 to 8bit uint, works with grayscale or RGB
def float01to8bit(im):
  return (im*255).astype(np.uint8)

def toFloat01from16bit(im):
  return im.astype(np.float32)/65535.0

###
### Image filtering and warp/unwarping

def Rotate(im, center, angleDeg):
    rows,cols = im.shape
    M = cv.getRotationMatrix2D(center,angleDeg,1)
    return cv.warpAffine(im,M,(cols,rows))

def PolarWarp(img):
    return cv.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), img.shape[0], cv.WARP_FILL_OUTLIERS)

def PolarUnwarp(img, shape):
    # INTER_AREA works best to remove artifeacts
    # INTER_CUBIC works well except for a horizontal line artifact at angle = 0
    # INTER_LINEAR, the default has a very noticeable vertical banding artifact across the top, and similarly around the limb
    unwarped = cv.linearPolar(img, (shape[0]/2, shape[1]/2), shape[0], cv.WARP_FILL_OUTLIERS| cv.WARP_INVERSE_MAP | cv.INTER_AREA)
    return unwarped

def GetMeanAndStddevImage(polar_image, n):
    h = polar_image.shape[0] # image is square, so h=w
    k = (h//(n*2))*2+1 # find kernel size from fraction of circle, ensure odd
    lefthalf = polar_image[:,0:h//2] # left half is circle of radius h//2
    (mean, stddev) = meanAndStdevfilt2dWithWraparound(lefthalf, (k,1))

    # don't use mean filter for corners, just copy that data directly to minimize artifacts
    righthalf = polar_image[:,h//2:] # right half is corners and beyond
    meanimage = cv.hconcat([mean, righthalf])

    # don't use stddev filter for corners, just repeat last column to minimize artifacts
    stddevimage = np.hstack((stddev, np.tile(stddev[:, [-1]], h - h//2)))
    return (meanimage, stddevimage)

# pad the image on top and bottom to allow filtering with simulated wraparound
def PadForWrapAround(input, pad):
    return cv.vconcat([input[input.shape[0]-pad:,:], input, input[:pad,:]])

# remove padding from top and bottom
def RemoveWrapAroundPad(inputpadded, pad):
    return inputpadded[pad:inputpadded.shape[0]-pad,:]

def meanAndStdevfilt2dWithWraparound(input, kernel_size):
    # pad input image with half of kernel to simulate wraparound
    imagepad = PadForWrapAround(input, kernel_size[0]//2)

    # filter the padded image
    meanpad = sp.ndimage.uniform_filter(imagepad, kernel_size, mode='reflect')
    meanofsquaredpad = sp.ndimage.uniform_filter(imagepad*imagepad, kernel_size, mode='reflect')

    # sqrt(meanofsquared - mean*mean) is mathematically equivalent to std dev:
    #   https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows
    stddevpad = np.sqrt((meanofsquaredpad - meanpad*meanpad).clip(min=0))

    mean = RemoveWrapAroundPad(meanpad, kernel_size[0]//2)
    stddev = RemoveWrapAroundPad(stddevpad, kernel_size[0]//2)
    return (mean, stddev)

###
### Misc image processing

def brighten(im, gamma, gammaweight):
  return gammaweight*np.power(im, gamma)+(1-gammaweight)*(1-np.power(1-im,1/gamma))

def swapRB(im):
  blue = im[:,:,0].copy()
  im[:,:,0] = im[:,:,2].copy()
  im[:,:,2] = blue
  return im

def shrink(im,div):
    return cv.resize(im,np.floor_divide((im.shape[1],im.shape[0]),div))

def addCircle(im, center, radius, color, thickness):
    cv.ellipse(im, center, (radius, radius), 0, 0, 360, color, thickness, cv.LINE_AA)
    return im

# Create an expanded image centered on the sun. Ensure that a bounding circle
# centered on the sun and enclosing the original image's four corners is fully
# enclosed in the resulting image. For added pixels, pad by copying the existing
# edge pixels. This means that processing of the polar-warped image has reasonable
# values out to the maximum distance included in the original source image. This,
# in turn, means that circular banding artifacts will occur farther out and can be
# fully cropped out at the end.
def CenterAndExpand(center,src):
    (toleft,toright) = (center[0], src.shape[1] - center[0])
    (totop, tobottom) = (center[1], src.shape[0] - center[1])
    toUL = math.sqrt(totop*totop+toleft*toleft)
    toUR = math.sqrt(totop*totop+toright*toright)
    toBL = math.sqrt(tobottom*tobottom+toleft*toleft)
    toBR = math.sqrt(tobottom*tobottom+toright*toright)
    maxdist = int(max(toUL,toUR,toBL,toBR))+1
    newcenter = (maxdist, maxdist)
    outimg = np.pad(src, ((maxdist-totop, maxdist-tobottom), (maxdist-toleft, maxdist-toright)), mode='edge')
    return (newcenter, outimg)

def CropToDist(src, center, mindist):
    mindist = math.ceil(mindist)
    newcenter = (mindist, mindist)
    # note, does NOT force to odd
    outimg = src[center[1]-mindist:center[1]+mindist, center[0]-mindist:center[0]+mindist]
    return (newcenter, outimg)

def CalcMinDistToEdge(center, shape):
    (toleft,toright) = (center[0], shape[1] - center[0])
    (totop, tobottom) = (center[1], shape[0] - center[1])
    mindist = int(min(toleft,totop,toright,tobottom))-1
    return mindist

def CenterAndCrop(center,src):
    return CropToDist(src, center, CalcMinDistToEdge(center, src.shape))

def ForceRadius(im, center, rad, newrad):
  scale = newrad/rad
  im2 = cv.resize(im, ((int)(im.shape[1]*scale), (int)(im.shape[0]*scale)))
  center2 = ((int)(center[0]*scale), (int)(center[1]*scale))
  return (center2, im2)

def getDiskMask(src, center, radius):
    # create 32 bit float disk mask
    diskmask = np.zeros(src.shape[:2], dtype="float32")
    cv.ellipse(diskmask, center, (radius,radius), 0,0, 360, 1.0, -1, cv.FILLED) # no LINE_AA!
    return diskmask

###
### Functions need to evaluate alignment similarity, using log-gabor filter

# Log-Gabor filter
# from https://stackoverflow.com/questions/31774071/implementing-log-gabor-filter-bank/31796747
def getLogGaborFilter(N, f_0, theta_0, number_orientations):
    # filter configuration
    scale_bandwidth =  0.996 * math.sqrt(2/3)
    angle_bandwidth =  0.996 * (1/math.sqrt(2)) * (np.pi/number_orientations)

    # x,y grid
    extent = np.arange(-N/2, N/2 + N%2)
    x, y = np.meshgrid(extent,extent)

    mid = int(N/2)
    ## orientation component ##
    theta = np.arctan2(y,x)
    center_angle = ((np.pi/number_orientations) * theta_0) if (f_0 % 2) \
                else ((np.pi/number_orientations) * (theta_0+0.5))

    # calculate (theta-center_theta), we calculate cos(theta-center_theta)
    # and sin(theta-center_theta) then use atan to get the required value,
    # this way we can eliminate the angular distance wrap around problem
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    ds = sintheta * math.cos(center_angle) - costheta * math.sin(center_angle)
    dc = costheta * math.cos(center_angle) + sintheta * math.sin(center_angle)
    dtheta = np.arctan2(ds,dc)

    orientation_component =  np.exp(-0.5 * (dtheta/angle_bandwidth)**2)

    ## frequency component ##
    # go to polar space
    raw = np.sqrt(x**2+y**2)
    # set origin to 1 as in the log space zero is not defined
    raw[mid,mid] = 1
    # go to log space
    raw = np.log2(raw)

    center_scale = math.log2(N) - f_0
    draw = raw-center_scale
    frequency_component = np.exp(-0.5 * (draw/ scale_bandwidth)**2)

    # reset origin to zero (not needed as it is already 0?)
    frequency_component[mid,mid] = 0

    kernel =  frequency_component * orientation_component
    return kernel

# simpler function to do both fft and shift
def fft(im):
    return np.fft.fftshift(np.fft.fft2(im))

# simpler function to do both inverse fft and shift
def ifft(f):
    return np.real(np.fft.ifft2(np.fft.ifftshift(f)))

# create the frequency space filter image for all orientations
def getLGs(N, f_0, num_orientations):
    return [getLogGaborFilter(N,f_0,x,num_orientations) for x in range(0,num_orientations)]

def applyFilter(im, lg):
    # apply fft to go to freqency space, apply filter, then inverse fft to go back to spatial
    # take absolute value so we have only non-negative values, and merge into multi-channel image
    f = [np.abs(ifft(fft(im) * lg[x])) for x in range(0,len(lg))]
    im = cv.merge(f)
    return im

###
### Implementation of algorithm in Aligning 'Dissimilar' Images Directly

def GetRij(num,den,k):
    # TODO: cleanup this conditional code meant to exclude very small denominators
    denshape = den.shape
    den = den.flatten()
    num = num.flatten()
    rij = np.zeros(den.shape)
    epsilon = 0.0001/(k*k) # divide here because I simplified out k*k* out of den
    rij[den > epsilon] = num[den > epsilon]/den[den > epsilon]
    rij = np.reshape(rij, denshape)
    return rij

def CalcN(rij):
    abs_rij = abs(rij)
    #c = 2#1 # what value for constant?
    #n = 1/(1+np.power((1-abs_rij)/(1+abs_rij), c/2))
    n = 1/(1+((1-abs_rij)/(1+abs_rij))) # optimized for c=2, power can be removed
    return n

def GetSimilaritySum(n):
    nflat = np.zeros(n.shape[:2])
    for i in range(n.shape[2]):
        nflat = nflat + n[:,:,i]
    H = np.sum(n)
    return (H, nflat)

def Similarity(im1,im2):
    # find the correlation coefficient at each pixel rij
    k=5
    bf1 = cv.boxFilter(im1, -1, (k,k))
    bf2 = cv.boxFilter(im2, -1, (k,k))
    bf12 = cv.boxFilter(im1*im2, -1, (k,k))
    bf11 = cv.boxFilter(im1*im1, -1, (k,k))
    bf22 = cv.boxFilter(im2*im2, -1, (k,k))

    # Optimized by heavily simplifying from sum over phi1*phi2:
    # equivalent to quadruple-nested loop with:
    #   r[i,j] += (im1[i+ki,j+kj]-u1[i,j]) * im2[i+ki,j+kj]-u2[i,j])
    # which is
    #   r[i,j] += im1[i+ki,j+kj]*im2[i+ki,j+kj] -u1[i,j]*im2[i+ki,j+kj] -u2[i,j]*im1[i+ki,j+kj] +u1[i,j]*u2[i,j]
    #r = 25*(boxfilter(im1*im2,5) + (-u1)*boxfilter(im2,5) + (-u2)*boxfilter(im1,5) + (u1*u2))
    # also removed the 25* since it all cancels out, except I had to adjust the epsilon
    phi11 = (bf11 - bf1*bf1)
    phi12 = (bf12 - bf1*bf2)
    phi22 = (bf22 - bf2*bf2)
    num = phi12
    den = cv.sqrt(phi22 * phi11)
    rij = GetRij(num,den,k)
    n = CalcN(rij)
    return GetSimilaritySum(n)

###
### local and global search of alignment based on simularity evaluation

def ShowLocalEval(gong_filtered, inpu_rot, inpu_filtered, H, n, angle):
    pass
    #cv.imshow("inpu_rot",inpu_rot)
    #cv.imshow("n",n*0.5-1.0)
    #cv.waitKey(1)

def LocalSearchEvaluate(inpu, lg, mask, gong_filtered, angle, reportcallback):
    inpu_rot = Rotate(inpu, (inpu.shape[1]//2, inpu.shape[0]//2), angle)
    inpu_filtered = applyFilter(inpu_rot,lg)*mask
    (H,n) = Similarity(gong_filtered, inpu_filtered)
    reportcallback(gong_filtered, inpu_rot, inpu_filtered, H, n, angle)
    return (H, n, inpu_rot, inpu_filtered)

## start with a rough peak, with 3 data points, iterate until narrow enough and return final peak
def LocalSearch(inpu, lg, mask, gong_filtered, triple, stopping, reportcallback):
    angles, Hs = triple
    while(abs(angles[1]-angles[2]) > stopping):
        # evaluate midpoints
        ang = [angles[0], 0.5*(angles[0]+angles[1]), angles[1], 0.5*(angles[1]+angles[2]), angles[2]]
        results1 = LocalSearchEvaluate(inpu, lg, mask, gong_filtered, ang[1], reportcallback)
        results2 = LocalSearchEvaluate(inpu, lg, mask, gong_filtered, ang[3], reportcallback)
        sim = [Hs[0], results1[0], Hs[1], results2[0], Hs[2]]

        #find resulting peak and new triples
        peak = np.argmax(sim[1:5])+1 # first one can't be peak, but might be equal to peak
        #print(f"Local {ang[peak]}, {sim[peak]}, delta={ang[1]-ang[0]}")
        angles, Hs = (ang[peak-1:peak+2], sim[peak-1:peak+2])
    return (angles[1], Hs[1])

def GlobalSearch(inpu, lg, mask, gong_filtered, start, end, count, bestangle, bestsim):
    angles = []
    Hs = []
    index = -1
    for angle in np.linspace(start, end, num=count, endpoint=False):
        inpu_rot = Rotate(inpu, (inpu.shape[1]//2, inpu.shape[0]//2), angle)
        inpu_filtered = applyFilter(inpu_rot,lg)*mask
        (H,n) = Similarity(gong_filtered, inpu_filtered)
        angles.append(angle)
        Hs.append(H)
        #cv.imshow("inpu_rot",inpu_rot)
        #cv.imshow("n",n*0.5-1.0)
        #cv.waitKey(1)

    peak = np.argmax(Hs)
    a = (peak-1 + len(Hs)) % len(Hs)
    b = (peak+1 + len(Hs)) % len(Hs)
    return ([angles[a], angles[peak], angles[b]], [Hs[a], Hs[peak], Hs[b]])

def CenterAndCropToFixedRadius(center, radius, im, pixelRadius, solarRadii):
    # ensure we have enough buffer, scale to get fixed radius, then crop
    (center, im) = CenterAndExpand(center, im)
    (center, im) = ForceRadius(im, center, radius, pixelRadius)
    (center, im) = CropToDist(im, center, pixelRadius*solarRadii)
    return (center, im)

## assumes global search has already been done, supplying isflipped and initial 3 results in triple
def LocalSearchHelper(inpu, isflipped, lg, mask, gong, gong_filtered, triple):
    inpu_search = cv.flip(inpu,1) if isflipped else inpu
    (angle, sim) = LocalSearch(inpu_search, lg, mask, gong_filtered, triple, 0.1, ShowLocalEval)
    return (angle, sim, isflipped, True, triple, gong, inpu_search)

## does a full search including global and local
def FullSearchHelper(inpu, isflipped, lg, mask, gong, gong_filtered, triple):
    triple = GlobalSearch(inpu, lg, mask, gong_filtered, -180, 180, 20, 0, -1)
    (unflippedAngle, unflippedSim) = LocalSearch(inpu, lg, mask, gong_filtered, triple, 0.1, ShowLocalEval)
    flipped = cv.flip(inpu,1)
    flippedtriple = GlobalSearch(flipped, lg, mask, gong_filtered, -180, 180, 20, 0, -1)
    (flippedAngle, flippedSim) = LocalSearch(flipped, lg, mask, gong_filtered, flippedtriple, 0.1, ShowLocalEval)

    return (unflippedAngle, unflippedSim, False, True, triple, gong, inpu) if unflippedSim > flippedSim else (flippedAngle, flippedSim, True, True, flippedtriple, gong, flipped)

###
### high-level alignment implementation

def ShowResult(gong,inpu,angle,isflipped):
    inpu = cv.flip(inpu,1) if isflipped else inpu
    (isValidGong, gong, gongcenter, isValidInpu, inpu, inpucenter) = CenterImagesForAlignment(inpu, gong, 250, 1.3)
    if not isValidGong or not isValidInpu:
        return
    #composite = cv.merge([gong, np.zeros(gong.shape).astype(np.float32), Translate(Rotate(inpu,inpucenter,angle), tx,ty)])
    showImage(gong)
    showImage(inpu)
    showImage(sp.ndimage.rotate(inpu,angle))

def CenterImagesForAlignment(inpu, gong, fixedRadius, solarradii):
    (isValidGong, gongcenter, gongradius) = findValidCircle(gong)
    if(not isValidGong):
        print("Error: Couldn't find valid circle for GONG solar disk!")

    (isValidInpu, inpucenter, inpuradius) = findValidCircle(inpu)
    if(not isValidInpu):
        print("Error: Couldn't find valid circle for input solar disk!")

    if(isValidGong and isValidInpu):
        (gongcenter, gong) = CenterAndCropToFixedRadius(gongcenter, gongradius, toFloat01from16bit(gong), fixedRadius, solarradii)
        (inpucenter, inpu) = CenterAndCropToFixedRadius(inpucenter, inpuradius, toFloat01from16bit(inpu), fixedRadius, solarradii)

    return (isValidGong, gong, gongcenter, isValidInpu, inpu, inpucenter)

def AlignImages(gong, inpu, fixedRadius, triple, isflipped, searchfunc):
    (isValidGong, gong, gongcenter, isValidInpu, inpu, inpucenter) = CenterImagesForAlignment(inpu, gong, fixedRadius, 1.1)
    if not isValidGong or not isValidInpu:
        return (0, 0, False, False, ([],[]), gong, inpu)

    lg = getLGs(gong.shape[0],4,4)
    mask = cv.merge([getDiskMask(gong, gongcenter, (int)(fixedRadius * 0.8)) for _ in range(len(lg))])
    gong_filtered = applyFilter(gong,lg)*mask

    return searchfunc(inpu, isflipped, lg, mask, gong, gong_filtered, triple)

# do a single experiment with one image and one percent
def Align(inpu, date, i, percent, triple, flipped, searchfunc):
    gong = GetGongImageForDate(datetime.datetime.strptime(date, '%Y/%m/%d'), percent)
    (angle, similarity, flipped, matchFound, triple, gongout, inpuout) = AlignImages(gong, inpu, 128, triple, flipped, searchfunc)
    #if matchFound:
    #    print(f"  {i}, {percent}, {angle}, {flipped}, {similarity}")
    #    ShowResult(gong,inpu,angle, flipped)
    #else:
    #    print(f"  {i} Failed")
    return (angle, similarity, triple, flipped, gong, gongout, inpu, inpuout)

# given an image and a date, find the best angle
def FindBestAlignment(input_i, date, i):
    (angle, similarity, triple, flipped, gong_big, gong, inpu_big, inpu) = Align(input_i, date, i, 0.5, None, None, FullSearchHelper)
    best = (0.5, angle, flipped, similarity, gong_big, gong, inpu_big, inpu)
    percents = np.linspace(0.1,0.9,9).tolist() # from 0.1 to 0.9, inclusive, by 0.1
    percents.remove(0.5) # and remove 0.5
    for percent in percents:
        (angle, similarity, triple, flipped, gong_big, gong, inpu_big, inpu) = Align(input_i, date, i, percent, triple, flipped, LocalSearchHelper)
        if similarity > best[3]:
            best = (percent, angle, flipped, similarity, gong_big, gong, inpu_big, inpu)
    return best

###
### File uploading, downloading, and rendering

def getImageFz(url):
    trap = io.StringIO()
    with redirect_stdout(trap): # hide the obnoxious progress bar
        image_data = astropy.io.fits.getdata(url)
    img_float = image_data.astype(np.float32).clip(min=0)/np.max(image_data)
    img = float01to16bit(img_float)
    return cv.flip(img,0) # image coords are upside down

def getImage(url):
    fn = "testalign.png"
    open(fn, 'wb').write(requests.get(url, allow_redirects=True).content)
    return force16Gray(readImage(fn)) # force to single-channel 16-bit grayscale

def GetGongImageURL(date, percent, gongroot, fileend):
    yyyy = date.strftime("%Y")
    mm = date.strftime("%m")
    dd = date.strftime("%d")
    gongdir = gongroot+yyyy+mm+"/"+yyyy+mm+dd+"/"
    data = urllib.request.urlopen(gongdir).read()
    data2 = data.split(b'\"')
    w = [str(a)[2:-1] for a in data2 if str(a)[2:4] == '20' and str(a)[-5:-1].lower() == fileend]
    fn = w[int(len(w)*percent)]
    gongfullpath = gongdir+fn
    return gongfullpath

def GetGongImageForDate(date, percent):
    #return getImage(GetGongImageURL(date, percent, "https://gong2.nso.edu/HA/hag/", '.jpg'))
    return getImageFz(GetGongImageURL(date, percent, "https://nispdata.nso.edu/ftp/HA/haf/", 's.fz'))

def readImage(fn):
  return cv.imread(cv.samples.findFile(fn), cv.IMREAD_UNCHANGED | cv.IMREAD_ANYDEPTH)

def uploadFile():
  keys = list(files.upload().keys())
  return keys[0] if keys else ""

def writeImage(im, fn, suffix):
  # strip full path after the last .
  withoutextension = fn[::-1].split('.',1)[1][::-1] # reverse, split first ., take 2nd part, reverse again
  outfn = withoutextension + '-' + suffix + '.png'
  cv.imwrite(outfn, im)
  return outfn

def downloadImage(im, fn, suffix):
  files.download(writeImage(im, fn, suffix))

def downloadButton(im, fn, suffix):
  if IN_COLAB:
    button = widgets.Button(description='Download Image')
    button.on_click(lambda x: downloadImage(im,fn,suffix))
    display(button)

def showRGB(im):
  if IN_COLAB:
    plt.imshow(im)
    plt.show()
  else:
    cv.imshow(str(id(im)),swapRB(shrink(im,3)))
    cv.waitKey(1)

def showFloat01(im):
  showRGB(colorize8RGB(im,1,1,1))

###
### Enhancement

def displayIntermediateResults(polar_image, meanimage, unwarpedmean, diff, normstddev, enhancefactor, enhance, fn, silent):
  print("Polar warp the image as an initial step to make a pseudoflat")
  if not silent:
    showFloat01(polar_image)

  print("Mean filter on polar warp image")
  if not silent:
    showFloat01(meanimage)

  print("Finally unwarp the mean image to get the pseudoflat:")
  if not silent:
    showFloat01(unwarpedmean)
  downloadButton(float01to16bit(unwarpedmean), fn, "unwarpedmean")

  print("Subtract pseudoflat from image:")
  if not silent:
    showFloat01(diff+0.5)
  downloadButton(float01to16bit(diff+0.5), fn, "diff")

  print("Result of standard deviation filter, to drive contrast enhancement")
  if not silent:
    showFloat01(normstddev)

  print("Enhanced contrast in diff image:")
  if not silent:
    showFloat01((diff*enhancefactor + 0.5).clip(min=0,max=1))
  downloadButton(float01to16bit((diff*enhancefactor + 0.5).clip(min=0,max=1)), fn, "diff-enhanced")

  print("Enhance contrast and add back to pseudoflat:")
  if not silent:
    showFloat01(enhance)
  downloadButton(float01to16bit(enhance), fn, "enhancedgray")


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
def CNRGF_Enhance(img, minrecip, maxrecip, fn, silent):
  # find mean and standard deviation image from polar-warped image, then unwarp
  polar_image = PolarWarp(img)
  (meanimage, stddevs) = GetMeanAndStddevImage(polar_image, 6)
  unwarpedmean = PolarUnwarp(meanimage, img.shape)
  unwarpedstddev = PolarUnwarp(stddevs, img.shape)

  # adjust range of standard deviation image to get preferred range of contrast enhancement
  normstddev = cv.normalize(unwarpedstddev, None, 1/maxrecip, 1/minrecip, cv.NORM_MINMAX)

  # subtract mean, divide by standard deviation, and add back mean
  enhanceFactor = np.reciprocal(normstddev)
  diff = img-unwarpedmean
  enhance = diff*enhanceFactor + unwarpedmean

  # final normalize and clip
  enhance = enhance.clip(min=0.01) # don't want sunspot pixels blowing up the normalize
  enhance = cv.normalize(enhance, None, 0,1, cv.NORM_MINMAX).clip(min=0).clip(max=1)

  displayIntermediateResults(polar_image, meanimage, unwarpedmean, diff, normstddev, enhanceFactor, enhance, fn, silent)
  return enhance

# returns a function that normalizes to within a given range
def GetStdDevScaler(minrecip, maxrecip):
  return lambda sd: cv.normalize(sd, None, 1/maxrecip, 1/minrecip, cv.NORM_MINMAX)

# CNRGF split into two parts, first part does expensive compute
def CNRGF_Enhance_part1(img, n):
  (meanimage, stddevs) = GetMeanAndStddevImage(PolarWarp(img), n)
  return (PolarUnwarp(meanimage, img.shape), PolarUnwarp(stddevs, img.shape))

# CNRGF split into two parts, second part is cheaper and has tunable parameters
# using scaleStdDev as a function that has tunable parameters baked into it.
def CNRGF_Enhance_part2(img, mean_and_stddev, scaleStdDev):
  (unwarpedmean, unwarpedstddev) = mean_and_stddev
  normstddev = scaleStdDev(unwarpedstddev)
  return (img-unwarpedmean)*np.reciprocal(normstddev) + unwarpedmean

# CNRGF combining the two parts in one go
def Enhance(img, n, minrecip, maxrecip, minclip):
  mean_and_stddev = CNRGF_Enhance_part1(img, n)
  enhance = CNRGF_Enhance_part2(img, mean_and_stddev, GetStdDevScaler(minrecip, maxrecip))
  return cv.normalize(enhance.clip(min=minclip), None, 0,1, cv.NORM_MINMAX).clip(min=0).clip(max=1)

###
### Interactive

def InteractiveAdjust(img, center, radius, disttoedge, minadj, maxadj, gamma, gammaweight, minclip):
  def on_changemin(val): nonlocal minadj; minadj = 1.0+val/10.0; update()
  def on_changemax(val): nonlocal maxadj; maxadj = 1.0+val/10.0; update()
  def on_changegamma(val): nonlocal gamma; gamma = val/100.0; update()
  def on_changegammaweight(val): nonlocal gammaweight; gammaweight = val/100.0; update()

  def update():
    enhance = Enhance(shrink(img,3), 6, minadj, maxadj, 0.01)
    enhance = brighten(enhance, gamma, gammaweight)
    enhance8 = swapRB(colorize8RGB(enhance, 0.5, 1.25, 3.75))
    cv.imshow('adjust', enhance8)

  update()
  cv.createTrackbar('min adjust', 'adjust', 7, 100, on_changemin)
  cv.createTrackbar('max adjust', 'adjust', 30, 100, on_changemax)
  cv.createTrackbar('gamma', 'adjust', 70, 100, on_changegamma)
  cv.createTrackbar('gammaweight', 'adjust', 50, 100, on_changegammaweight)
  cv.waitKey(0)
  return (minadj, maxadj, gamma, gammaweight, minclip)

###
### main - drive the high-level flow

def isUrl(filenameOrUrl):
  prefix = filenameOrUrl[:6]
  return prefix == "https:" or prefix == "http:/"

# fetch an image as 16 bit grayscale, given a local filename or URL
# also returns the filename on disk, in case image came from URL
def FetchImage(filenameOrUrl):
  fn = filenameOrUrl

  # if it's a URL, download it
  if isUrl(filenameOrUrl):
    fn = "tempsolarimage.tif"
    open(fn, 'wb').write(requests.get(filenameOrUrl, allow_redirects=True).content)

  # force to single-channel 16-bit grayscale
  src = force16Gray(readImage(fn))
  return src, fn

def FlipImage(im, horiz, vert):
  if horiz:
    im = cv.flip(im, 1)
  if vert:
    im = cv.flip(im, 0)
  return im

# align a single image, given a date to compare against
def AlignImage(im, date, silent):
  if not silent:
    print(f"Original image before alignment:")
    showFloat01(toFloat01from16bit(im))
    print(f"Aligning with GONG image from {date}. This might take a minute.")

  date = date.replace('-','/')
  best = FindBestAlignment(im, date, 0)
  percent, angle, flipped, similarity, gong_big, gong, inpu_big, inpu = best

  if not silent:
    flippedtext = 'and horizontally flipped' if flipped else ''
    print(f"Best angle is {angle} {flippedtext}")
    print(f"GONG image used for alignment:")
    showFloat01(gong)

  if flipped:
    im = cv.flip(im,1)
  im = sp.ndimage.rotate(im,angle)
  return im

# process a single image, silently
def SilentProcessImage(src, minrecip, maxrecip, brightengamma, gammaweight):
  (isValid, srccenter, radius) = findValidCircle(src)
  if(not isValid):
    return None

  # use a expanded/centered grayscale 0-1 float image for all calculations
  (center, centered) = CenterAndExpand(srccenter,src)
  img = toFloat01from16bit(centered)

  enhance = CNRGF_Enhance(img, minrecip, maxrecip, "", True)
  (center, enhance) = CropToDist(enhance, center, CalcMinDistToEdge(srccenter, src.shape))

  # brighten and colorize
  enhance = brighten(enhance, brightengamma, gammaweight)
  enhance16 = colorize16BGR(enhance, 0.5, 1.25, 3.75)
  return enhance16

# process a single image, with verbose output
def ProcessImage(src, minrecip, maxrecip, brightengamma, gammaweight, fn):
  # find the solar disk circle
  (isValid, srccenter, radius) = findValidCircle(src)
  if(not isValid):
    print("Couldn't find valid circle for solar disk!")
    return None

  # show original image as uploaded
  print(f"\nOriginal image size: {src.shape[1]},{src.shape[0]}  Circle found with radius {radius} and center {srccenter[0]},{srccenter[1]}")
  showFloat01(toFloat01from16bit(src))

  # use a expanded/centered grayscale 0-1 float image for all calculations
  (center, centered) = CenterAndExpand(srccenter,src)
  img = toFloat01from16bit(centered)
  print(f"centered image size: {centered.shape[1]},{centered.shape[0]}  New center: {center[0]},{center[1]}")
  showFloat01(img)
  downloadButton(centered, fn, "centered")

  # show image with circle drawn
  imageWithCircle = addCircle(gray2rgb(img), center, radius, (1,0,0), 3)
  solarRadiusInKm = 695700
  print(f"centered image with solar limb circle highlighted. Circle should be very close to the edge of the photosphere. Pixel size is about {solarRadiusInKm/radius:.1f}km")
  showRGB(imageWithCircle)
  imageWithCircle = addCircle(gray2rgb(img), center, radius, (0,0,1), 1)
  downloadButton(float01to16bit(imageWithCircle), fn, "withcircle")

  minclip = 0.01
  if not IN_COLAB:
    disttoedge = CalcMinDistToEdge(srccenter, src.shape)
    params = InteractiveAdjust(img, center, radius, disttoedge, minrecip, maxrecip, brightengamma, gammaweight, minclip)
    (minrecip, maxrecip, brightengamma, gammaweight, minclip) = params

  enhance = CNRGF_Enhance(img, minrecip, maxrecip, fn, False)
  #enhance = Enhance(img, 6, minrecip, maxrecip, minclip)
  (center, enhance) = CropToDist(enhance, center, CalcMinDistToEdge(srccenter, src.shape))

  # brighten and colorize
  enhance = brighten(enhance, brightengamma, gammaweight)
  print("Brighten image:")
  showFloat01(enhance)
  downloadButton(float01to16bit(enhance), fn, "enhancedgraybright")

  enhance8 = colorize8RGB(enhance, 0.5, 1.25, 3.75)
  enhance16 = colorize16BGR(enhance, 0.5, 1.25, 3.75)
  print("And finally colorize image:")
  showRGB(enhance8)
  downloadButton(enhance16, fn, "enhancedcolor")
  return enhance16

# process a single image - from filename or URL
def imageMain(filenameOrUrl, silent, hflip, vflip, align, date, mincontrastadjust, maxcontrastadjust, brightengamma, gammaweight):
  src, filename = FetchImage(filenameOrUrl)
  src = FlipImage(src, hflip, vflip)

  if align:
    src = AlignImage(src, date, silent)

  if silent:
    enhance16 = SilentProcessImage(src, mincontrastadjust, maxcontrastadjust, brightengamma, gammaweight)
  else:
    enhance16 = ProcessImage(src, mincontrastadjust, maxcontrastadjust, brightengamma, gammaweight, filename)

  return enhance16, filename

# process command line arguments to get parameters, and get list of files to process
def ProcessArgs():
    fnlist = []
    parser = argparse.ArgumentParser(description='Process solar images')
    #parser.add_argument('-d', action='store_true', help='Treat the argv[1] as a directory and process all files in it, implies -s')
    parser.add_argument('-t', '--type', type=str, default='tif', help='filetype to go along with -d, defaults to tif')
    parser.add_argument('-p', '--pattern', type=str, default='', help='String pattern to match for -d')
    parser.add_argument('-o', '--output', type=str, nargs='?', help='Output directory')
    parser.add_argument('-s', '--silent', action='store_true', help='run silently')
    parser.add_argument('-a', '--append', action='store_true', help='append the settings used for gamma, min contrast, and max contrast as part of the output filename')
    parser.add_argument('-f', '--flip', type=str, default='', choices=['h', 'v', 'hv'], help='rotate final images horizontally, vertically, or both')
    parser.add_argument('-g', '--gongalign', type=str, default='', help='Date of GONG image to compare for auto-align, YYYY-MM-DD')
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

    if len(fnlist) == 0:
      print(f"No files found, using sample image")
      fnlist.append("")

    if not args.output:
      output = directory
    else:
      output = args.output

    hflip = 'h' in args.flip
    vflip = 'v' in args.flip
    return fnlist, silent, directory, hflip, vflip, output, args.append, args.gongalign

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

  mincontrastadjust = 1.7 # @param {type:"number"}   # 1.6
  maxcontrastadjust = 3.0 # @param {type:"number"}   # 4.0
  brightengamma = 0.5 # @param {type:"number"}       # 0.7
  gammaweight = 0.5 # @param {type:"number")         # 0.5
  shouldAlignFirst = False # @param {type:"boolean"}
  dateIfAligning = "2023-12-17" # @param {type:"date"}
  shouldUseUrl = False # @param {type:"boolean"}
  urlToUse = "https://www.cloudynights.com/uploads/monthly_01_2023/post-412916-0-66576300-1674591059.jpg" # @param{type:"string"}
  fallbackUrl = 'https://www.cloudynights.com/uploads/gallery/album_24182/gallery_79290_24182_1973021.png'

  # get the solar disk image
  if IN_COLAB:
    fnlist, silent, directory, hflip, vflip, outputDirectory, append, gongAlignDate = [""], False, ".", False, False, ".", False, ""
    print("Upload full disk solar image now, or click cancel to use default test image")
    fnlist[0] = urlToUse if shouldUseUrl else uploadFile()
  else:
    fnlist, silent, directory, hflip, vflip, outputDirectory, append, gongAlignDate = ProcessArgs()

  suffix = f"minc_{str(mincontrastadjust)}_maxc_{str(maxcontrastadjust)}_g{str(brightengamma)}" if append else ""
  if gongAlignDate != "":
    shouldAlignFirst = True
    dateIfAligning = gongAlignDate

  if fnlist[0] == "":
    fnlist[0] = fallbackUrl

  for fn in fnlist:
    fullName = fn if isUrl(fn) else directory + '/' + fn
    enhance16, outfn = imageMain(fullName, silent, hflip, vflip, shouldAlignFirst, dateIfAligning, mincontrastadjust, maxcontrastadjust, brightengamma, gammaweight)
    if not IN_COLAB:
      outfn = outputDirectory + '/' + os.path.basename(outfn) # replace input dir without output dir
      writeImage(enhance16, outfn, "enhancedcolor" + suffix)
      writeImage(cv.cvtColor(enhance16, cv.COLOR_BGR2GRAY), outfn, "enhancedgray" + suffix)
      cv.waitKey(0)
      cv.destroyAllWindows()

main()