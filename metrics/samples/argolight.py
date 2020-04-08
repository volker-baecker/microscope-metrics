
from metrics.analysis.tools import segment_image, compute_distances_matrix, compute_spots_properties
from metrics.utils.utils import multi_airy_fun
# from metrics.plot import plot

import numpy as np
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from statistics import median

# Creating logging services
import logging
module_logger = logging.getLogger('metrics.samples.argolight')


# ___________________________________________
#
# ANALYZING 'SPOTS' MATRIX. PATTERNS XXX
# Computing chromatic shifts, homogeneity,...
# ___________________________________________

def analyze_spots(image, pixel_size, pixel_size_units, low_corr_factors, high_corr_factors):
    labels = segment_image(image=image,
                           min_distance=30,
                           sigma=(1, 3, 3),
                           method='local_max',
                           low_corr_factors=low_corr_factors,
                           high_corr_factors=high_corr_factors)

    spots_properties, spots_positions = compute_spots_properties(image=image,
                                                                 labels=labels,
                                                                 remove_center_cross=False)

    spots_distances = compute_distances_matrix(positions=spots_positions,
                                               sigma=2.0,
                                               pixel_size=pixel_size)

    # Return some key-value pairs
    key_values = dict()
    # TODO: Median z-distance

    # Return a table with detailed data
    table_col_names = list()
    table_col_desc = list()
    table_data = list()

    # Populating the data
    table_col_names.append('roiVolumeUnit')
    table_col_desc.append('Volume units for the ROIs.')
    table_data.append(['VOXEL'])

    table_col_names.append('roiWeightedCentroidUnits')
    table_col_desc.append('Weighted Centroid coordinates units for the ROIs.')
    table_data.append(['PIXEL'])

    table_col_names.append('Distance3dUnits')
    table_col_desc.append('Weighted Centroid 3d distances units for the ROIs.')
    table_data.append([pixel_size_units[0]])

    for c, ch_spot_prop in enumerate(spots_properties):
        key_values[f'Nr_of_spots_ch{c:02d}'] = len(ch_spot_prop)

        key_values[f'Max_Intensity_ch{c:02d}'] = max(x['integrated_intensity'] for x in ch_spot_prop)
        key_values[f'Max_Intensity_Roi_ch{c:02d}'] = \
            ch_spot_prop[[x['integrated_intensity'] for x in ch_spot_prop].index(key_values[f'Max_Intensity_ch{c:02d}'])]['label']

        key_values[f'Min_Intensity_ch{c:02d}'] = min(x['integrated_intensity'] for x in ch_spot_prop)
        key_values[f'Min_Intensity_Roi_ch{c:02d}'] = \
            ch_spot_prop[[x['integrated_intensity'] for x in ch_spot_prop].index(key_values[f'Min_Intensity_ch{c:02d}'])]['label']

        key_values[f'Min-Max_intensity_ratio_ch{c:02d}'] = key_values[f'Min_Intensity_ch{c:02d}'] / key_values[f'Max_Intensity_ch{c:02d}']

    for c, ch_spot_prop in enumerate(spots_properties):
        table_col_names.append(f'ch{c:02d}_MaskLabels')
        table_col_desc.append('Labels of the mask ROIs.')
        table_data.append([[x['label'] for x in ch_spot_prop]])

        table_col_names.append(f'ch{c:02d}_Volume')
        table_col_desc.append('Volume of the ROIs.')
        table_data.append([[x['area'].item() for x in ch_spot_prop]])

        table_col_names.append(f'ch{c:02d}_MaxIntensity')
        table_col_desc.append('Maximum intensity of the ROIs.')
        table_data.append([[x['max_intensity'].item() for x in ch_spot_prop]])

        table_col_names.append(f'ch{c:02d}_MinIntensity')
        table_col_desc.append('Minimum intensity of the ROIs.')
        table_data.append([[x['min_intensity'].item() for x in ch_spot_prop]])

        table_col_names.append(f'ch{c:02d}_MeanIntensity')
        table_col_desc.append('Mean intensity of the ROIs.')
        table_data.append([[x['mean_intensity'].item() for x in ch_spot_prop]])

        table_col_names.append(f'ch{c:02d}_IntegratedIntensity')
        table_col_desc.append('Integrated intensity of the ROIs.')
        table_data.append([[x['integrated_intensity'].item() for x in ch_spot_prop]])

        table_col_names.append(f'ch{c:02d}_XWeightedCentroid')
        table_col_desc.append('Weighted Centroid X coordinates of the ROIs.')
        table_data.append([[x['weighted_centroid'][2].item() for x in ch_spot_prop]])

        table_col_names.append(f'ch{c:02d}_YWeightedCentroid')
        table_col_desc.append('Weighted Centroid Y coordinates of the ROIs.')
        table_data.append([[x['weighted_centroid'][1].item() for x in ch_spot_prop]])

        table_col_names.append(f'ch{c:02d}_ZWeightedCentroid')
        table_col_desc.append('Weighted Centroid Z coordinates of the ROIs.')
        table_data.append([[x['weighted_centroid'][0].item() for x in ch_spot_prop]])

    for c, chs_dist in enumerate(spots_distances):
        table_col_names.append(f'ch{chs_dist["channels"][0]:02d}_ch{chs_dist["channels"][1]:02d}_chARoiLabels')
        table_col_desc.append('Labels of the ROIs in channel A.')
        table_data.append([chs_dist['labels_of_A']])

        table_col_names.append(f'ch{chs_dist["channels"][0]:02d}_ch{chs_dist["channels"][1]:02d}_chBRoiLabels')
        table_col_desc.append('Labels of the ROIs in channel B.')
        table_data.append([chs_dist['labels_of_B']])

        table_col_names.append(f'ch{chs_dist["channels"][0]:02d}_ch{chs_dist["channels"][1]:02d}_3dDistance')
        table_col_desc.append(
            'Distance in 3d between Weighted Centroids of mutually closest neighbouring ROIs in channels A and B.')
        table_data.append([[x.item() for x in chs_dist['dist_3d']]])

        key_values[f'Median_3d_dist_ch{chs_dist["channels"][0]:02d}_ch{chs_dist["channels"][1]:02d}'] = \
            median(table_data[-1][-1])

    key_values['Distance_units'] = pixel_size_units[0]

    # plot.plot_homogeneity_map(raw_stack=image,
    #                           spots_properties=spots_properties,
    #                           spots_positions=spots_positions,
    #                           labels_stack=labels_stack)
    #
    # plot.plot_distances_maps(distances=spots_distances,
    #                          x_dim=image.shape[-2],
    #                          y_dim=image.shape[-1])

    return labels, table_col_names, table_col_desc, table_data, key_values


# _____________________________________
#
# ANALYSING LINES PATTERN. PATTERNS XXX
# Computing resolution
# _____________________________________


def analyze_resolution(image, pixel_size, pixel_units, axis, measured_band=.4):
    profiles, \
        z_planes, \
        peak_positions, \
        peak_heights, \
        resolution_values, \
        resolution_indexes, \
        resolution_method = compute_resolution(image=image,
                                               axis=axis,
                                               measured_band=measured_band,
                                               prominence=.2,
                                               do_angle_refinement=False)
    # resolution in native units
    resolution_values = [x * pixel_size[axis] for x in resolution_values]

    key_values = dict()

    for c, res in enumerate(resolution_values):
        key_values[f'ch{c:02d}_{resolution_method}_resolution'] = res.item()

    key_values['resolution_units'] = pixel_units[0]
    key_values['resolution_axis'] = axis
    key_values['measured_band'] = measured_band

    for c, indexes in enumerate(resolution_indexes):
        key_values[f'ch{c:02d}_peak_positions'] = [(peak_positions[c][ind].item(), peak_positions[c][ind + 1].item()) for ind in indexes]
        key_values[f'ch{c:02d}_peak_heights'] = [(peak_heights[c][ind].item(), peak_heights[c][ind + 1].item()) for ind in indexes]
        key_values[f'ch{c:02d}_focus'] = z_planes[c].item()

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

    # plot.plot_peaks(profiles, peaks, peak_properties, resolution_values, resolution_indexes)

    return profiles, key_values


def _fit(profile, peaks_guess, amp=4, lower_amp=2, upper_amp=5, center_tolerance=1):
    guess = list()
    lower_bounds = list()
    upper_bounds = list()
    for p in peaks_guess:
        guess.append(p)  # pead center
        guess.append(amp)
        lower_bounds.append(p - center_tolerance)
        lower_bounds.append(lower_amp)
        upper_bounds.append(p + center_tolerance)
        upper_bounds.append(upper_amp)

    x = np.linspace(0, profile.shape[0], profile.shape[0])

    popt, pcov = curve_fit(multi_airy_fun, x, profile, p0=guess, bounds=(lower_bounds, upper_bounds))

    opt_peaks = popt[::2]
    opt_amps = [a / 4 for a in popt[1::2]]  # We normalize back the amplitudes to the unity

    return opt_peaks, opt_amps


def _compute_channel_resolution(channel, axis, prominence, measured_band, do_fitting=True, do_angle_refinement=False):
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
    axis_len = focus_slice.shape[-axis]
    weight_profile = np.zeros(axis_len)
    # Calculates a band of relative width 'image_fraction' to integrate the profile
    weight_profile[int((axis_len / 2) - (axis_len * measured_band / 2)):int((axis_len / 2) + (axis_len * measured_band / 2))] = 1
    profile = np.average(focus_slice,
                         axis=-axis,
                         weights=weight_profile)

    normalized_profile = (profile - np.min(profile)) / np.ptp(profile)

    # Find peaks: We implement Rayleigh limits that will be refined downstream
    peak_positions, properties = find_peaks(normalized_profile,
                                            height=.3,
                                            distance=2,
                                            prominence=prominence / 4,
                                            )

    # From the properties we are interested in teh amplitude
    # peak_heights = [h for h in properties['peak_heights']]
    ray_filtered_peak_pos = []
    ray_filtered_peak_heights = []

    for peak, height, prom in zip(peak_positions, properties['peak_heights'], properties['prominences']):
        if (prom / height) > prominence:
            ray_filtered_peak_pos.append(peak)
            ray_filtered_peak_heights.append(height)

    peak_positions = ray_filtered_peak_pos
    peak_heights = ray_filtered_peak_heights

    # TODO: We have to filter by relative prominences.
    if do_fitting:
        peak_positions, peak_heights = _fit(normalized_profile, peak_positions)

    # Find the closest peaks to return it as a measure of resolution
    peaks_distances = [abs(a - b) for a, b in zip(peak_positions[0:-2], peak_positions[1:-1])]
    res = min(peaks_distances)
    res_indices = [i for i, x in enumerate(peaks_distances) if x == res]

    return normalized_profile, z_focus, peak_positions, peak_heights, res, res_indices


def compute_resolution(image, axis, measured_band, prominence=0.1, do_angle_refinement=False):
    profiles = list()
    z_planes = list()
    peaks_positions = list()
    peaks_heights = list()
    resolution_values = list()
    resolution_indexes = list()
    resolution_method = 'Rayleigh'

    for c in range(image.shape[1]):  # TODO: Deal with Time here
        prof, zp, pk_pos, pk_heights, res, res_ind = _compute_channel_resolution(channel=np.squeeze(image[:, c, ...]),
                                                                                 axis=axis,
                                                                                 prominence=prominence,
                                                                                 measured_band=measured_band,
                                                                                 do_angle_refinement=do_angle_refinement)
        profiles.append(prof)
        z_planes.append(zp)
        peaks_positions.append(pk_pos)
        peaks_heights.append(pk_heights)
        resolution_values.append(res)
        resolution_indexes.append(res_ind)

    return profiles, z_planes, peaks_positions, peaks_heights, resolution_values, resolution_indexes, resolution_method

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
