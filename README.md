# SolarFinish
App to apply finishing touches to full disk solar images

Supports:
- Automatic rotational/flip alignment to GONG image given a date
- Manual horizontal or vertical flip
- Enhanced contrast using Convolutional Normalizing Radial Graded Filter
- Automatic centering
- Cropping to given solar radius multiple
- Brightening with a blended gamma curve: `weight*np.power(im, gamma) + (1-weight)*(1-np.power(1-im,1/gamma))`
- Simple colorization with linear RGB
