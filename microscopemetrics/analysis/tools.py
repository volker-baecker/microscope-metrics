"""These are some possibly useful code snippets"""
from pandas import DataFrame
from skimage.filters import threshold_otsu, apply_hysteresis_threshold, gaussian
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, cube, octahedron, ball
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from scipy import ndimage
import numpy as np
from itertools import permutations

import logging

# Creating logging services
module_logger = logging.getLogger(__name__)


def _segment_channel(channel, min_distance, method,
                     threshold, sigma, low_corr_factor, high_corr_factor,
                     indices):
    """Segment a channel (3D numpy array)
    """
    if threshold is None:
        threshold = threshold_otsu(channel)

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
                               indices=indices)
        thresholded = np.copy(channel)
        thresholded[peaks] = thresholded.max()
        thresholded = apply_hysteresis_threshold(thresholded,
                                                 low=threshold * low_corr_factor,
                                                 high=threshold * high_corr_factor,
                                                 )
    else:
        raise Exception('A valid segmentation method was not provided')

    closed = closing(thresholded, cube(min_distance))
    cleared = clear_border(closed)
    return label(cleared)


def segment_image(image,
                  min_distance,
                  sigma=None,
                  method='local_max',
                  low_corr_factors=None,
                  high_corr_factors=None,
                  indices=False):
    """Segment an image and return a labels object.
    Image must be provided as zctxy numpy array
    """
    module_logger.info('Image being segmented...')

    if low_corr_factors is None:
        low_corr_factors = [.95] * image.shape[1]
        module_logger.warning('No low correction factor specified. Using defaults')
    if high_corr_factors is None:
        high_corr_factors = [1.05] * image.shape[1]
        module_logger.warning('No high correction factor specified. Using defaults')

    if len(high_corr_factors) != image.shape[1] or len(low_corr_factors) != image.shape[1]:
        raise Exception('The number of correction factors does not match the number of channels.')

    # We create an empty array to store the output
    labels_image = np.zeros(image.shape, dtype=np.uint16)
    for c in range(image.shape[1]):
        for t in range(image.shape[2]):
            labels_image[:, c, t, ...] = _segment_channel(image[:, c, t, ...],
                                                          min_distance=min_distance,
                                                          method=method,
                                                          threshold=None,
                                                          sigma=sigma,
                                                          low_corr_factor=low_corr_factors[c],
                                                          high_corr_factor=high_corr_factors[c],
                                                          indices=indices)
    return labels_image


def _compute_channel_spots_properties(channel, label_channel, remove_center_cross=False, pixel_size=None):
    """Analyzes and extracts the properties of a single channel"""

    regions = regionprops(label_channel, channel)

    ch_properties = [
        {
            'label': region.label,
            'area': region.area,
            'centroid': region.centroid,
            'weighted_centroid': region.weighted_centroid,
            'max_intensity': region.max_intensity,
            'mean_intensity': region.mean_intensity,
            'min_intensity': region.min_intensity,
            'integrated_intensity': region.mean_intensity * region.area,
        }
        for region in regions
    ]


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


def compute_spots_properties(image, labels, remove_center_cross=False, pixel_size=None):
    """Computes a number of properties for the PSF-like spots found on an image provided they are segmented"""
    properties = []
    positions = []

    for c in range(image.shape[1]):
        for t in range(image.shape[2]):
            pr, pos = _compute_channel_spots_properties(channel=image[:, c, t, ...],
                                                        label_channel=labels[:, c, t, ...],
                                                        remove_center_cross=remove_center_cross,
                                                        pixel_size=pixel_size)
            properties.append(pr)
            positions.append(pos)

    return properties, positions


def compute_distances_matrix(positions, max_distance, pixel_size=None):
    """Calculates Mutual Closest Neighbour distances between all channels and returns the values as
    """
    module_logger.info('Computing distances between spots')

    if len(positions) < 2:
        raise Exception('Not enough dimensions to do a distance measurement')

    channel_permutations = list(permutations(range(len(positions)), 2))

    if not pixel_size:  # TODO: make sure the units are corrected if no pixel size
        pixel_size = np.array((1, 1, 1))
        module_logger.warning('No pixel size specified. Using the unit')
    else:
        pixel_size = np.array(pixel_size)

    for a, b in channel_permutations:
        distances_matrix = cdist(positions[a], positions[b], w=pixel_size)

        distances_df = DataFrame()

        for i, (pos_A, d) in enumerate(zip(positions[a], distances_matrix)):
            if d.min() < max_distance:
                distances_df = distances_df.append({"channel_a": a,
                                                    "channel_b": b,
                                                    "z_coord_a": pos_A[0],
                                                    "y_coord_a": pos_A[1],
                                                    "x_coord_a": pos_A[2],
                                                    "z_coord_b": positions[b][d.argmin()][0],
                                                    "y_coord_b": positions[b][d.argmin()][1],
                                                    "x_coord_b": positions[b][d.argmin()][2],
                                                    "z_dist": pos_A[0] - positions[b][d.argmin()][0],
                                                    "y_dist": pos_A[1] - positions[b][d.argmin()][1],
                                                    "x_dist": pos_A[2] - positions[b][d.argmin()][2],
                                                    'dist_3d': d.min(),
                                                    "labels_a": i,
                                                    "labels_b": d.argmin()
                                                    }, ignore_index=True
                                                   )

    return distances_df


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

    return ndimage.mean(image, labels=rbin, index=np.arange(1, rbin.max() + 1))


def _channel_fft_2d(channel):
    channel_fft = np.fft.rfft2(channel)
    return np.fft.fftshift(np.abs(channel_fft), axes=1)


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
    return np.fft.fftshift(np.abs(channel_fft), axes=1)


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
