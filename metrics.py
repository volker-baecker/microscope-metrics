"""These are some possibly useful code snippets"""

# Thresholding and labeling
from skimage.filters import threshold_otsu, apply_hysteresis_threshold
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, cube, octahedron, ball
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
import numpy as np
from itertools import permutations


class BeadsField2D:
    """
    Docs in here
    """
    def __init__(self,
                 image: np.ndarray,
                 dimensions: tuple = ('z', 'c', 'x', 'y'),
                 pixel_size_um = None,
                 ):
        self.image = image
        self.dimensions = dimensions
        self.pixel_size_um = pixel_size_um
        self.labels_image = np.zeros(self.image.shape, dtype=np.uint16)
        self.properties = list()
        self.positions = list()
        self.channel_permutations = list()
        self.distances = list()
        # TODO: reshape image if dimensions is not default
        # TODO: implement drift for a time dimension

    def channel_local_max(self, channel):
        threshold = threshold_otsu(channel)

        peaks = peak_local_max(channel,
                               min_distance=10,
                               threshold_abs=threshold,
                               indices=False)

        return peaks


    def segment_channel(self, channel):
        threshold = threshold_otsu(channel)

        thresholded = apply_hysteresis_threshold(channel, low=(threshold * .9), high=(threshold * 1.5))

        bw = closing(thresholded, cube(50))
        cleared = clear_border(bw)
        return label(cleared)

    def segment_image(self):
        for c in range(self.image.shape[1]):
            self.labels_image[:, c] = self.segment_channel(self.image[:, c])

    def compute_channel_properties(self, channel, label_channel):
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
        if self.pixel_size_um:
            ch_positions = ch_positions[0:] * self.pixel_size_um

        return ch_properties, ch_positions

    def compute_image_properties(self):
        for c in range(self.image.shape[1]):
            pr, pos = self.compute_channel_properties(self.image[:, c], self.labels_image[:, c])
            self.properties.append(pr)
            self.positions.append(pos)

    def compute_distances_matrix(self):
        """Calculates all possible distances between all channels and returns the interesting values"""

        if self.image.shape[0] == 1:
            raise Exception('Image has one only channel and no distances matrix can be calculated')
            return None
        if len(self.properties) == 0:  # We guess that properties have not been calculated yet
            self.compute_image_properties()

        n_dim = len(self.positions) - 1

        if n_dim < 2:
            raise Exception('Not enough dimensions to do a distance measurement')

        self.channel_permutations = list(permutations(range(len(self.positions)), 2))

        # Create a space to hold the distances. For every permutation we want to store:
        # Positions of a (indexes 0:2)
        # Distance to the closest spot in b (index 3)
        # Index of the nearest spot in b (index 4)

        distances = list()

        for a, b in self.channel_permutations:
            distances_matrix = cdist(self.positions[a], self.positions[b])

            distance = None
            for p, d in zip(self.positions[a], distances_matrix[0]):
                distance = list(p[:])
                distance.append(d.min())
                distance.append(d.argmin())

            self.distances.append(distance)
