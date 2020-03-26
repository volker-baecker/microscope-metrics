
from metrics.analysis.tools import segment_image, compute_distances_matrix, compute_spots_properties
from metrics.plot import plot

import numpy as np
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from scipy.signal import find_peaks
from statistics import median

import logging

# Creating logging services
module_logger = logging.getLogger('metrics.samples.argolight')


# ___________________________________________
#
# ANALYZING 'SPOTS' MATRIX. PATTERNS XXX
# Computing chromatic shifts, homogeneity,...
# ___________________________________________

def analyze_spots(image, pixel_sizes, low_corr_factors, high_corr_factors):

    labels_stack = segment_image(image=image,
                                 min_distance=30,
                                 sigma=(1, 3, 3),
                                 method='local_max',
                                 low_corr_factors=low_corr_factors,
                                 high_corr_factors=high_corr_factors)

    spots_properties, spots_positions = compute_spots_properties(image, labels_stack, remove_center_cross=False)

    spots_distances = compute_distances_matrix(positions=spots_positions,
                                               sigma=2.0,
                                               pixel_size=pixel_sizes)

    # We want to save:
    # A table per image containing the following columns:
    # - source Image
    # - Per channel
    #   - RoiColumn(name='chXX_MaxIntegratedIntensityRoi', description='ROI with the highest integrated intensity.', values)
    #   - RoiColumn(name='chXX_MinIntegratedIntensityRoi', description='ROI with the lowest integrated intensity.', values)
    #   - LongArrayColumn(name='chXX_roiMaskLabels', description='Labels of the mask ROIs.', size=(verify size), values)
    #   - LongArrayColumn(name='chXX_roiCentroidLabels', description='Labels of the centroids ROIs.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_roiVolume', description='Volume of the ROIs.', size=(verify size), values)
    # - StringColumn(name='roiVolumeUnit', description='Volume units for the ROIs.', size=(max size), values)
    #   - FloatArrayColumn(name='chXX_roiMinIntensity', description='Minimum intensity of the ROIs.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_roiMaxIntensity', description='Maximum intensity of the ROIs.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_roiMeanIntensity', description='Mean intensity of the ROIs.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_roiIntegratedIntensity', description='Integrated intensity of the ROIs.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_roiXWeightedCentroid', description='Wighted Centroid X coordinates of the ROIs.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_roiYWeightedCentroid', description='Wighted Centroid Y coordinates of the ROIs.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_roiZWeightedCentroid', description='Wighted Centroid Z coordinates of the ROIs.', size=(verify size), values)
    # - StringColumn(name='roiWeightedCentroidUnits', description='Wighted Centroid coordinates units for the ROIs.', size=(max size), values)
    # - Per channel permutation
    #   - FloatArrayColumn(name='chXX_chYY_chARoiLabels', description='Labels of the ROIs in channel A.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_chYY_chBRoiLabels', description='Labels of the ROIs in channel B.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_chYY_XDistance', description='Distance in X between Weighted Centroids of mutually closest neighbouring ROIs in channels A and B.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_chYY_YDistance', description='Distance in Y between Weighted Centroids of mutually closest neighbouring ROIs in channels A and B.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_chYY_ZDistance', description='Distance in Z between Weighted Centroids of mutually closest neighbouring ROIs in channels A and B.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_chYY_3DDistance', description='Distance in 3D between Weighted Centroids of mutually closest neighbouring ROIs in channels A and B.', size=(verify size), values)
    # - StringColumn(name='DistanceUnits', description='Wighted Centroid coordinates units for the ROIs.', size=(max size), values)

    plot.plot_homogeneity_map(raw_stack=image,
                              spots_properties=spots_properties,
                              spots_positions=spots_positions,
                              labels_stack=labels_stack)

    plot.plot_distances_maps(distances=spots_distances,
                             x_dim=image.shape[-2],
                             y_dim=image.shape[-1])


# _____________________________________
#
# ANALYSING LINES PATTERN. PATTERNS XXX
# Computing resolution
# _____________________________________

def analyze_resolution(image, pixel_sizes, pixel_units, axis):
    profiles, \
    peaks, \
    peak_properties, \
    resolution_values, \
    resolution_indexes, \
    resolution_method = compute_resolution(image=image,
                                           axis=axis,
                                           prominence=.2,
                                           do_angle_refinement=False)
    # resolution in native units
    resolution_values = [x * pixel_sizes[axis] for x in resolution_values]

    # We want to save:
    # - the profiles as a 1 pixels image
    # A table per image containing the following columns
    # - source Image
    # - the profiles Image
    # - Per channel:
    #   - Per method:
    #     - Per line:
    #       - RoiColumn(name='chXX_method_Line_X', description='Measured line with method {method}.', values)
    #     - FloatColumn(name='chXX_method_resolution' description='Measured resolution value using method {method}.', values)
    #     - FloatColumn(name='chXX_method_resolution_angle' description='Angle at which resolution was measured using method {method}.', values)
    # - StringColumn(name='resolutionUnits', description='Measured resolution units.', size=(max size), values)
    # - StringColumn(name='resolutionAngleUnits', description='Measured resolution angle units.', size=(max size), values)

    plot.plot_peaks(profiles, peaks, peak_properties, resolution_values, resolution_indexes)


def _compute_channel_resolution(channel, axis, prominence, do_angle_refinement=False):
    """Computes the resolution on a pattern of lines with increasing separation"""
    # find the most contrasted z-slice
    z_stdev = np.std(channel, axis=(1, 2))
    z_focus = np.argmax(z_stdev)
    focus_slice = channel[z_focus]  # TODO: verify 2 dimensions

    # TODO: verify angle and correct
    if do_angle_refinement:
        # Set a precision of 0.1 degree.
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 1800)
        h, theta, d = hough_line(focus_slice, theta=tested_angles)

    # project parallel to the axis of interest
    # parallel_prj = np.mean(focus_slice, axis=axis)

    # get the center and width of the lines pattern
    # TODO: properly get the best of the lines

    # TODO: we have to do some fitting or interpolation to get more precision then the single pixel

    # Cut a band of that found peak
    # Best we can do now is just to cut a band in the center
    # We create a profiles along which we average signal
    axis_len = focus_slice.shape[axis]
    weight_profile = np.zeros(axis_len)
    weight_profile[int(axis_len / 2.5):int(axis_len / 1.5)] = 1
    profile = np.average(focus_slice,
                         axis=axis,
                         weights=weight_profile)

    normalized_profile = (profile - np.min(profile))/np.ptp(profile)

    # Find peaks: We implement Rayleigh limits that will be refined downstream
    peaks, properties = find_peaks(normalized_profile,
                                   height=.3,
                                   distance=2,
                                   prominence=prominence,
                                   )

    # Find the closest peaks to return it as a measure of resolution
    peaks_distances = [abs(a - b) for a, b in zip(peaks[0:-2], peaks[1:-1])]
    res = min(peaks_distances)
    res_indices = [i for i, x in enumerate(peaks_distances) if x == res]

    return normalized_profile, peaks, properties, res, res_indices


def compute_resolution(image, axis, prominence=0.1, do_angle_refinement=False):
    profiles = list()
    peaks = list()
    peaks_properties = list()
    resolution_values = list()
    resolution_indexes = list()
    resolution_method = 'Rayleigh'

    for c in range(image.shape[1]):  # TODO: Deal with Time here
        prof, pk, pr, res, res_ind = _compute_channel_resolution(channel=np.squeeze(image[:, c, ...]),
                                                                 axis=axis,
                                                                 prominence=prominence,
                                                                 do_angle_refinement=do_angle_refinement)
        profiles.append(prof)
        peaks.append(pk)
        peaks_properties.append(pr)
        resolution_values.append(res)
        resolution_indexes.append(res_ind)

    return profiles, peaks, peaks_properties, resolution_values, resolution_indexes, resolution_method

# Calculate 2D FFT
# slice_2d = raw_img[17, ...].reshape([1, n_channels, x_size, y_size])
# fft_2D = fft_2d(slice_2d)

# Calculate 3D FFT
# fft_3D = fft_3d(spots_image)
#
# plt.imshow(np.log(fft_3D[2, :, :, 1]))  # , cmap='hot')
# # plt.imshow(np.log(fft_3D[2, 23, :, :]))  # , cmap='hot')
# plt.show()
#

