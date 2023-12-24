# SolarFinish
App to apply finishing touches to full disk solar images

Assumes a grayscale full solar disk input image that has been through typical stacking and sharpening using tools like Autostakkert and Imppg, but no or minimal adjustment of intensity curves. It's also best for the input image to be uncropped to avoid artifacts near edges and to allow arbitrary cropping at the end. That said, it takes what you gives it and tries its best as long as it can find a big circle in the image.

Supports:
- Automatic rotational/flip alignment to GONG image given a date
- Manual horizontal or vertical flip
- Enhanced contrast using Convolutional Normalizing Radial Graded Filter
- Automatic centering
- Cropping to given solar radius multiple
- Brightening with a blended gamma curve: `weight*np.power(im, gamma) + (1-weight)*(1-np.power(1-im,1/gamma))`
- Simple colorization with linear RGB
- Batch processing images in a directory
