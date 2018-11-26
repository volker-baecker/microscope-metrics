"""These are some possibly useful code snippets"""

# Thresholding and labeling
from skimage.filters import threshold_otsu, apply_hysteresis_threshold
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, cube, octahedron, ball
from scipy.spatial.distance import cdist
from itertools import permutations
import numpy as np



def argolight_field(channel,
                    pixel_size_nm=(None, None, None)):
    """Analyzes an array of rings"""
    properties = list()

    # If pixel sizes are not provided, we log the absence and we use 1
    if None in pixel_size_nm:
        print('Pixel sizes are not provided. The unity will be used')
        pixel_size_nm = (1., 1., 1.)

    # Calculate Otsu's threshold and use it with hysteresis:
    thresh = threshold_otsu(channel)
    thresholded = apply_hysteresis_threshold(channel, low=(thresh * .9), high=(thresh * 1.2))

    # We smooth the region detection using a growing and shrinking ball
    bw = closing(thresholded, cube(20))
    cleared = clear_border(bw)
    label_image = label(cleared)
    regions = regionprops(label_image, channel)

    for region in regions:
        properties.append({'label': region.label,
                           'area': region.area,
                           'centroid': tuple(l * r for l, r in zip(region.centroid, pixel_size_nm)),
                           'weighted_centroid': tuple(l * r for l, r in zip(region.weighted_centroid, pixel_size_nm)),
                           'max_intensity': region.max_intensity,
                           'mean_intensity': region.mean_intensity,
                           'min_intensity': region.min_intensity
                           })

    return properties, label_image


def analise_distances_matrix(positions):
    """Calculates all possible distances between all channels and returns the interesting values
    @:parameter positions: a list of numpy ndarray's containing dimensions [x, y, z] or [x, y]"""

    raw_distances_3d = list()
    raw_distances_1d = list()

    min_distances_3d = list()
    min_distances_1d = list()

    # Verify that there is more than 1 channel
    if len(positions) < 2:
        print('No distances found as there was only one channel.')
        return None

    # Verify the number of dimensions
    if positions[0].shape[1] == 3:
        n_dim = 3
    elif positions[0].shape[1] == 2:
        n_dim = 2
    else:
        raise Exception('Not enough dimensions to do a distance measurement')

    ch_permutations = permutations(positions, 2)
    ch_nr_permutatiions = permutations(range(len(positions)), 2)

    for pair in ch_permutations:
        raw_distances_3d.append(cdist(pair[0], pair[1]))
        # raw_distances_1d.append([cdist(pair[0][:, x], pair[1][:, x]) for x in range(pair[0].shape[1])])
        # raw_distances_1d.append(tuple([l - r for l, r in zip(pair[0], pair[1])]))

        # Get the smaller distances
        min_distances_3d.append(np.amin(raw_distances_3d[-1], 1))

    return min_distances_3d