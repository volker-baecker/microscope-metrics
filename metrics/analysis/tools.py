"""These are some possibly useful code snippets"""

from skimage.filters import threshold_otsu, apply_hysteresis_threshold, gaussian
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, cube, octahedron, ball
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from scipy import ndimage
import numpy as np
from itertools import permutations

# from dask import delayed

import logging

# Creating logging services
module_logger = logging.getLogger('metrics.analysis.tools')


def _segment_channel(channel, min_distance, sigma, method, low_corr_factor, high_corr_factor):
    """Segment a channel (3D numpy array)
    """
    threshold = threshold_otsu(channel)

    # TODO: Threshold be a sigma passed here
    if sigma is not None:
        channel = gaussian(image=channel,
                           multichannel=False,
                           sigma=sigma,
                           preserve_range=True)

    if method == 'hysteresis':  # We may try here hysteresis threshold
        thresholded = apply_hysteresis_threshold(channel,
                                                 low=threshold * low_corr_factor,
                                                 high=threshold * high_corr_factor
                                                 )

    elif method == 'local_max':  # We are applying a local maxima algorithm
        peaks = peak_local_max(channel,
                               min_distance=min_distance,
                               threshold_abs=(threshold * .5),
                               exclude_border=True,
                               indices=False)
        thresholded = np.copy(channel)
        thresholded[peaks] = thresholded.max()
        thresholded = apply_hysteresis_threshold(thresholded,
                                                 low=threshold * low_corr_factor,
                                                 high=threshold * high_corr_factor
                                                 )
    else:
        raise Exception('A valid segmentation method was not provided')

    closed = closing(thresholded, cube(min_distance))
    cleared = clear_border(closed)
    return label(cleared)


def segment_image(image,
                  min_distance=20,
                  sigma=None,
                  method='local_max',
                  low_corr_factors=None,
                  high_corr_factors=None):
    """Segment an image and return a labels object.
    Image must be provided as
    """
    module_logger.info('Image being segmented...')

    if low_corr_factors is None:
        low_corr_factors = [.95] * image.shape[-3]
        module_logger.warning('No low correction factor specified. Using defaults')
    if high_corr_factors is None:
        high_corr_factors = [1.05] * image.shape[-3]
        module_logger.warning('No high correction factor specified. Using defaults')

    # We create an empty array to store the output
    labels_image = np.zeros(image.shape, dtype=np.uint16)
    for c in range(image.shape[-3]):
        labels_image[..., c, :, :] = _segment_channel(image[..., c, :, :],
                                                      min_distance=min_distance,
                                                      sigma=sigma,
                                                      method=method,
                                                      low_corr_factor=low_corr_factors[c],
                                                      high_corr_factor=high_corr_factors[c])
    return labels_image


def _compute_channel_spots_properties(channel, label_channel, remove_center_cross=False, pixel_size=None):
    """Analyzes and extracts the properties of a single channel"""

    ch_properties = list()

    regions = regionprops(label_channel, channel)

    for region in regions:
        ch_properties.append({'label': region.label,
                              'area': region.area,
                              'centroid': region.centroid,
                              'weighted_centroid': region.weighted_centroid,
                              'max_intensity': region.max_intensity,
                              'mean_intensity': region.mean_intensity,
                              'min_intensity': region.min_intensity
                              })
    if remove_center_cross:  # Argolight spots pattern contains a central cross that we might want to remove
        largest_area = 0
        largest_region = None
        for region in ch_properties:
            if region['area'] > largest_area:  # We assume the cross is the largest area
                largest_area = region['area']
                largest_region = region
        if largest_region:
            ch_properties.remove(largest_region)
    ch_positions = np.array([x['weighted_centroid'] for x in ch_properties])
    if pixel_size:
        ch_positions = ch_positions[0:] * pixel_size

    return ch_properties, ch_positions


def compute_spots_properties(image, labels, remove_center_cross=True):
    """Computes a number of properties for the PSF-like spots found on an image provided they are segmented"""
    # TODO: Verify dimensions of image and labels are the same
    properties = list()
    positions = list()

    for c in range(image.shape[-3]):  # TODO: Deal with Time here
        pr, pos = _compute_channel_spots_properties(image[..., c, :, :],
                                                    labels[..., c, :, :],
                                                    remove_center_cross=remove_center_cross)
        properties.append(pr)
        positions.append(pos)

    return properties, positions


def compute_distances_matrix(positions, sigma, pixel_size=None):
    """Calculates Mutual Closest Neighbour distances between all channels and returns the values as
    a list of tuples where the first element is a tuple with the channel combination (ch_A, ch_B) and the second is
    a list of pairwise measurements where, for every spot s in ch_A:
    - Positions of s (s_x, s_y, s_z)
    - Weighted euclidean distance dst to the nearest spot in ch_B, t
    - Index t_index of the nearest spot in ch_B
    Like so:
    [((ch_A, ch_B), [[(s_x, s_y, s_z), dst, t_index],...]),...]
    """
    module_logger.info('Computing distances between spots')
    # TODO: Correct documentation
    # Container for results
    distances = list()

    if len(positions) < 2:
        raise Exception('Not enough dimensions to do a distance measurement')

    channel_permutations = list(permutations(range(len(positions)), 2))

    # Create a space to hold the distances. For every channel permutation (a, b) we want to store:
    # Coordinates of a
    # Distance to the closest spot in b
    # Index of the nearest spot in b

    if not pixel_size:
        pixel_size = np.array((1, 1, 1))
        module_logger.warning('No pixel size specified. Using the unit')
    else:
        pixel_size = np.array(pixel_size)

    for a, b in channel_permutations:
        # TODO: Try this
        # TODO: Make explicit arguments of cdist
        distances_matrix = cdist(positions[a], positions[b], w=pixel_size)

        pairwise_distances = {'channels': (a, b),
                              'coord_of_A': list(),
                              'dist_3d': list(),
                              'index_of_B': list()
                              }
        for p, d in zip(positions[a], distances_matrix):
            if d.min() < sigma:
                pairwise_distances['coord_of_A'].append(tuple(p))
                pairwise_distances['dist_3d'].append(d.min())
                pairwise_distances['index_of_B'].append(d.argmin())

        distances.append(pairwise_distances)

    return distances


def _radial_mean(image, bins=None):
    """Computes the radial mean from an input 2d image.
    Taken from scipy-lecture-notes 2.6.8.4
    """
    # TODO: workout a binning = image size
    if not bins:
        bins = 200
    size_x, size_y = image.shape
    x, y = np.ogrid[0:size_x, 0:size_y]

    r = np.hypot(x - size_x / 2, y - size_y / 2)

    rbin = (bins * r / r.max()).astype(np.int)
    radial_mean = ndimage.mean(image, labels=rbin, index=np.arange(1, rbin.max() + 1))

    return radial_mean


def _channel_fft_2d(channel):
    channel_fft = np.fft.rfft2(channel)
    channel_fft_magnitude = np.fft.fftshift(np.abs(channel_fft), axes=1)
    return channel_fft_magnitude


def fft_2d(image):
    # Create an empty array to contain the transform
    fft = np.zeros(shape=(image.shape[1],
                          image.shape[2],
                          image.shape[3] // 2 + 1),
                   dtype='float64')
    for c in range(image.shape[2]):
        fft[c, :, :] = _channel_fft_2d(image[..., c, :, :])

    return fft


def _channel_fft_3d(channel):
    channel_fft = np.fft.rfftn(channel)
    channel_fft_magnitude = np.fft.fftshift(np.abs(channel_fft), axes=1)
    return channel_fft_magnitude


def fft_3d(image):
    fft = np.zeros(shape=(image.shape[0],
                          image.shape[1],
                          image.shape[2],
                          image.shape[3],
                          image.shape[4] // 2 + 1),  # We only compute the real part of the FFT
                   dtype='float64')
    for c in range(image.shape[-3]):
        fft[..., c, :, :] = _channel_fft_3d(image[..., c, :, :])

    return fft
