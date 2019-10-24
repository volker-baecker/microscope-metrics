"""These are some possibly useful code snippets"""

# Thresholding and labeling
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

# class BeadsField2D:
#     """
#     Docs in here
#     """
#     def __init__(self,
#                  image: np.ndarray,
#                  dimensions: tuple = ('z', 'c', 'x', 'y'),
#                  pixel_size_um = None,
#                  ):
#         self.image = image
#         self.dimensions = dimensions
#         self.pixel_size_um = pixel_size_um
#         self.labels_image = np.zeros(self.image.shape, dtype=np.uint16)
#         self.properties = list()
#         self.positions = list()
#         self.channel_permutations = list()
#         self.distances = list()
#         # TODO: reshape image if dimensions is not default
#         # TODO: implement drift for a time dimension


def _segment_single_channel(channel, min_distance, method='local_max', hysteresis_levels=(.7, 1.0)):
    """Segment a given channel (3D numpy array) to find PSF-like spots"""
    threshold = threshold_otsu(channel)

    # TODO: Thereshoudl be a sigma passed here
    gauss_filtered = gaussian(np.copy(channel), (3, 3, 1))

    if method == 'hysteresis':  # We may try here hysteresis threshold
        thresholded = apply_hysteresis_threshold(gauss_filtered,
                                                 low=threshold * hysteresis_levels[0],
                                                 high=threshold * hysteresis_levels[1]
                                                 )

    elif method == 'local_max':  # We are applying a local maxima algorithm
        peaks = peak_local_max(gauss_filtered,
                               min_distance=min_distance,
                               threshold_abs=(threshold * .2),
                               exclude_border=True,
                               indices=False
                               )
        thresholded = np.copy(gauss_filtered)
        thresholded[peaks] = thresholded.max()
        thresholded = apply_hysteresis_threshold(thresholded,
                                                 low=threshold * hysteresis_levels[0],
                                                 high=threshold * hysteresis_levels[1]
                                                 )
    else:
        raise Exception('A valid segmentation method was not provided')

    closed = closing(thresholded, cube(min_distance))
    cleared = clear_border(closed)
    return label(cleared)


def segment_image(image, min_distance=30, method='local_max', hysteresis_levels=(.7, 1.0)):
    """Segment an image and return a labels object"""
    # We create an empty array to store the output
    labels_image = np.zeros(image.shape, dtype=np.uint16)
    for c in range(image.shape[1]):
        labels_image[..., c] = _segment_single_channel(image[..., c],
                                                       min_distance,
                                                       method,
                                                       hysteresis_levels)
    return labels_image


def _compute_channel_spots_properties(channel, label_channel, pixel_size=None):
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
    ch_positions = np.array([x['weighted_centroid'] for x in ch_properties])
    if pixel_size:
        ch_positions = ch_positions[0:] * pixel_size

    return ch_properties, ch_positions


def compute_spots_properties(image, labels):
    """Computes a number of properties for the PSF-like spots found on an image provided they are segmented"""
    # TODO: Verify dimmensions of image and labels are the same
    properties = list()
    positions = list()

    for c in range(image.shape[1]):
        pr, pos = _compute_channel_spots_properties(image[:, c, ...], labels[:, c, ...])
        properties.append(pr)
        positions.append(pos)

    return properties, positions


def compute_distances_matrix(positions, pixel_size=None):
    """Calculates Mutual Closest Neighbour distances between all channels and returns the values as
    a list of tuples where the first element is a tuple with the channel combination (ch_A, ch_B) and the second is
    a list of pairwise measurements where, for every spot s in ch_A:
    - Positions of s (s_x, s_y, s_z)
    - Weighted euclidean distance dst to the nearest spot in ch_B, t
    - Index t_index of the nearest spot in ch_B
    Like so:
    [((ch_A, ch_B), [[(s_x, s_y, s_z), dst, t_index],...]),...]
    """

    # Container for results
    distances = list()

    if len(positions) < 2:
        raise Exception('Not enough dimensions to do a distance measurement')

    channel_permutations = list(permutations(range(len(positions)), 2))

    # Create a space to hold the distances. For every permutation (a, b) we want to store:
    # Positions of a (indexes 0:2)
    # Distance to the closest spot in b (index 3)
    # Index of the nearest spot in b (index 4)

    if not pixel_size:
        pixel_size = np.array((1, 1, 1))
        # TODO: log warning
    else:
        pixel_size = np.array(pixel_size)

    for a, b in channel_permutations:
        # TODO: Try this
        distances_matrix = cdist(positions[a], positions[b], w=pixel_size)

        pairwise_distances = list()
        for p, d in zip(positions[a], distances_matrix):
            single_distance = list()
            single_distance.append(tuple(p))  # Adding the coordinates of spot in ch_a
            single_distance.append(d.min())  # Appending the 3D distance
            single_distance.append(d.argmin())  # Appending the index of the closest spot in ch_b
            pairwise_distances.append(single_distance)

        distances.append(((a, b), pairwise_distances))

    return distances


def _radial_mean(image, bins=None):
    """Computes the radial mean from an input 2d image.
    Taken from scipy-lecture-notes 2.6.8.4
    """
    # TODO: workout a binning = image size
    if not bins:
        bins = 200
    size_x, size_y = image.shape
    X, Y = np.ogrid[0:size_x, 0:size_y]

    r = np.hypot(X - size_x / 2, Y - size_y / 2)

    rbin = (bins * r / r.max()).astype(np.int)
    radial_mean = ndimage.mean(image, labels=rbin, index=np.arange(1, rbin.max() + 1))

    return radial_mean


def _channel_fft_2d(channel):

    # channel_fft = np.fft.rfftn(channel)
    channel_fft = np.fft.rfft2(channel)
    channel_fft_magnitude = np.fft.fftshift(np.abs(channel_fft), 1)
    return channel_fft_magnitude


def fft_2d(image):

    # Create an empty array to contain the transform
    fft = np.zeros(shape=(image.shape[1],
                          image.shape[2],
                          image.shape[3] // 2 + 1),
                   dtype='float64')
    for c in range(image.shape[1]):
        fft[c, ...] = _channel_fft_2d(image[:, c, ...])

    return fft

def _channel_fft_3d(channel):
    channel_fft = np.fft.rfftn(channel)

def fft_3d():
    pass

