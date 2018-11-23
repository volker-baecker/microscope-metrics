"""These are some possibly useful code snippets"""

# Thresholding and labeling
from skimage.filters import threshold_otsu, apply_hysteresis_threshold
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, cube, octahedron, ball
from scipy.spatial.distance import cdist
import numpy as np



def argolight_field(channel,
                    dimensions=('z','x', 'y'),
                    x_pixel_size_nm=None,
                    y_pixel_size_nm=None,
                    z_pixel_size_nm=None):
    """Analyzes an array of rings"""
    properties = list()

    thresh = threshold_otsu(channel)

    # We may try here hysteresis thresholding
    thresholded = apply_hysteresis_threshold(channel, low=(thresh * .9), high=(thresh * 1.2))

    bw = closing(thresholded, cube(2))
    cleared = clear_border(bw)
    label_image = label(cleared)
    regions = regionprops(label_image, channel)

    for region in regions:
        properties.append({'label': region.label,
                           'area': region.area,
                           'centroid': region.centroid,
                           'weighted_centroid': region.weighted_centroid,
                           'max_intensity': region.max_intensity,
                           'mean_intensity': region.mean_intensity,
                           'min_intensity': region.min_intensity
                           })

    return properties, label_image


def analise_distances_matrix(positions):
    """Calculates all possible distances between all channels and returns the imteresting values
    @:parameter positions: a numpy ndarray containing dimensions [channel, x, y, z] or [channel, x, y]"""

    if len(positions.shape) == 4:
        n_dim = 3
    elif len(positions.shape) == 3:
        n_dim = 2
    else:
        raise Exception('Not enough dimensions to do a distance measurement')

    if n_dim == 2:
        distances = cdist(positions[1, :, :], positions[2, :, :])
    if n_dim == 3:
        distances = cdist(positions[1, :, :, :], positions[2, :, :, :])

    return distances