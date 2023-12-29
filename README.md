# SolarFinish
App to apply finishing touches to full disk solar images

Assumes a grayscale full solar disk input image that has been through typical stacking and sharpening using tools like Autostakkert and Imppg, but no or minimal adjustment of intensity curves. It's also best for the input image to be uncropped to avoid artifacts near edges and to allow arbitrary cropping at the end. That said, it takes what you give it and tries its best as long as it can find a big circle in the image.

Supports:
- Automatic rotational/flip alignment to GONG image given a date
- Manual horizontal or vertical flip
- Manual rotation
- Enhanced contrast using Convolutional Normalizing Radial Graded Filter
- Automatic centering
- Cropping to given solar radius multiple
- Brightening with a blended gamma curve: `weight*np.power(im, gamma) + (1-weight)*(1-np.power(1-im,1/gamma))`
- Simple colorization with linear RGB
- Batch processing images in a directory
- Processing image from filepath or remote URL

```
usage: SolarFinish.py [-h] [-t TYPE] [-p PATTERN] [-o [OUTPUT]] [-s] [-a]
                      [-f {h,v,hv}] [-g GONGALIGN] [-b BRIGHTEN]
                      [-w BRIGHTENWEIGHT] [-e ENHANCE] [-c CROP] [-r ROTATE]
                      [-d DARKCLIP] [-i]
                      [filename]

Process solar images

positional arguments:
  filename              Image file to process

options:
  -h, --help            show this help message and exit
  -t TYPE, --type TYPE  filetype to go along with -d, defaults to tif
  -p PATTERN, --pattern PATTERN
                        String pattern to match for -d
  -o [OUTPUT], --output [OUTPUT]
                        Output directory
  -s, --silent          run silently
  -a, --append          append the settings used for gamma, min contrast, and
                        max contrast as part of the output filename
  -f {h,v,hv}, --flip {h,v,hv}
                        rotate final images horizontally, vertically, or both
  -g GONGALIGN, --gongalign GONGALIGN
                        Date of GONG image to compare for auto-align, YYYY-MM-
                        DD
  -b BRIGHTEN, --brighten BRIGHTEN
                        gamma value to brighten by, 1 = none, 0.1 = extreme
                        bright, 2.0 darken
  -w BRIGHTENWEIGHT, --brightenweight BRIGHTENWEIGHT
                        weight to shift gamma brightening, 1 = use gamma
                        curve, 0 = less brightening of darker pixels
  -e ENHANCE, --enhance ENHANCE
                        contrast enhance min,max. 1 = no enhance, 5 = probably
                        too much
  -c CROP, --crop CROP  final crop radius in solar radii
  -r ROTATE, --rotate ROTATE
                        rotation in degrees
  -d DARKCLIP, --darkclip DARKCLIP
                        clip minimum after contrast enhancement and before
                        normalization
  -i, --interact        interactively adjust parameters
```
