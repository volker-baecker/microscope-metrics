
from metrics.analysis.tools import segment_image, compute_distances_matrix, compute_spots_properties
from metrics.utils.utils import multi_airy_fun, airy_fun, gaussian_fun

import numpy as np
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from statistics import median
import datetime

# Creating logging services
import logging

module_logger = logging.getLogger('metrics.samples.psf_beads')


# ___________________________________________
# ANALYZING PSF BEADS
# ___________________________________________
# TODO: Implemented single channel only


def estimate_min_bead_distance(NA, pixel_size):
    return 50


def calculate_nyquist():
    pass


def calculate_theoretcal_resolution():
    pass


def _fit_gaussian(profile, guess=None):
    if guess is None:
        guess = [profile.min(), profile.max(), profile.argmax(), .8]
    x = np.linspace(0, profile.shape[0], profile.shape[0], endpoint=False)
    popt, pcov = curve_fit(gaussian_fun, x, profile, guess)

    fitted_profile = gaussian_fun(x, popt[0], popt[1], popt[2], popt[3])
    fwhm = popt[3] * 2.35482

    return fitted_profile, fwhm

def analize_bead(image):

    # Find the strongest sections to generate profiles
    x_max = np.max(image, axis=(0, 1))
    x_focus = np.argmax(x_max)
    y_max = np.max(image, axis=(0, 2))
    y_focus = np.argmax(y_max)
    z_max = np.max(image, axis=(1, 2))
    z_focus = np.argmax(z_max)

    # Generate profiles
    x_profile = np.squeeze(image[z_focus, y_focus, :])
    y_profile = np.squeeze(image[z_focus, :, x_focus])
    z_profile = np.squeeze(image[:, y_focus, x_focus])

    # Fitting the profiles
    x_fitted_profile, x_fwhm = _fit_gaussian(x_profile)
    y_fitted_profile, y_fwhm = _fit_gaussian(y_profile)
    z_fitted_profile, z_fwhm = _fit_gaussian(z_profile)

    return (x_profile, y_profile, z_profile), \
           (x_fitted_profile, y_fitted_profile, z_fitted_profile), \
           (x_fwhm, y_fwhm, z_fwhm)


    return x_profile, y_profile, z_profile

def find_beads(image, pixel_size, NA, min_distance=None, sigma=None):  # , low_corr_factors, high_corr_factors):

    if min_distance is None:
        min_distance = estimate_min_bead_distance(NA, pixel_size)

    image_mip = np.max(image, axis=0)

    if sigma is not None:
        image_mip = gaussian(image=image_mip,
                             multichannel=False,
                             sigma=(sigma, sigma),
                             preserve_range=True)

    # Find bead centers
    positions = peak_local_max(image=image_mip,
                               threshold_rel=0.2,
                               indices=True)

    nr_beads = positions.shape[0]
    module_logger.info(f'Beads found: {nr_beads}')

    # Exclude beads too close to the edge
    edge_keep_map = (positions[:,0] > min_distance) & \
                    (positions[:,0] < image_mip.shape[0] - min_distance) & \
                    (positions[:,1] > min_distance) & \
                    (positions[:,1] < image_mip.shape[1] - min_distance)
    module_logger.info(f'Beads too close to the edge: {nr_beads - np.sum(edge_keep_map)}')

    # Exclude beads too close to eachother
    proximity_keep_map = np.ones((nr_beads, nr_beads),dtype=bool)
    for i, pos in enumerate(positions):
        proximity_keep_map[i] = (abs(positions[:, 0] - pos[0]) > min_distance) |  \
                                (abs(positions[:, 1] - pos[1]) > min_distance)
        proximity_keep_map[i,i] = True  # Correcting the diagonal
    proximity_keep_map = np.all(proximity_keep_map, axis=0)
    module_logger.info(f'Beads too close to eachother: {nr_beads - np.sum(proximity_keep_map)}')

    # Exclude beads too intense or too weak
    # TODO: Exclude those.

    keep_map = edge_keep_map & proximity_keep_map
    module_logger.info(f'Beads either too close to the edge or to eachother: {nr_beads - np.sum(keep_map)}')

    positions = positions[keep_map,:]
    module_logger.info(f'Beads kept: {positions.shape[0]}')

    frames = list()
    for pos in positions:
        frames.append(image[:,  \
                            (pos[0]-(min_distance//2)):(pos[0]+(min_distance//2)),  \
                            (pos[1]-(min_distance//2)):(pos[1]+(min_distance//2))]
                      )

    profiles = list()
    for frame in frames:
        profiles.append(analize_bead(frame))

    return positions, profiles


    #

    #
    #
    # spots_properties, spots_positions = compute_spots_properties(image=image,
    #                                                              labels=labels,
    #                                                              )
    #
    # # Return some key-value pairs
    # key_values = dict()
    #
    # # Return some tables
    # properties = [{'name': 'roi_volume_units',
    #                'desc': 'Volume units for the ROIs.',
    #                'getter': lambda x, props: [x for n in range(len(props))],
    #                'data': list(),
    #                },
    #               {'name': 'roi_weighted_centroid_units',
    #                'desc': 'Weighted Centroid coordinates units for the ROIs.',
    #                'getter': lambda x, props: [x for n in range(len(props))],
    #                'data': list(),
    #                },
    #               {'name': 'channel',
    #                'desc': 'Channel.',
    #                'getter': lambda ch, props: [ch for x in props],
    #                'data': list(),
    #                },
    #               {'name': 'mask_labels',
    #                'desc': 'Labels of the mask ROIs.',
    #                'getter': lambda ch, props: [p['label'] for p in props],
    #                'data': list(),
    #                },
    #               {'name': 'volume',
    #                'desc': 'Volume of the ROIs.',
    #                'getter': lambda ch, props: [p['area'].item() for p in props],
    #                'data': list(),
    #                },
    #               {'name': 'max_intensity',
    #                'desc': 'Maximum intensity of the ROIs.',
    #                'getter': lambda ch, props: [p['max_intensity'].item() for p in props],
    #                'data': list(),
    #                },
    #               {'name': 'min_intensity',
    #                'desc': 'Minimum intensity of the ROIs.',
    #                'getter': lambda ch, props: [p['min_intensity'].item() for p in props],
    #                'data': list(),
    #                },
    #               {'name': 'mean_intensity',
    #                'desc': 'Mean intensity of the ROIs.',
    #                'getter': lambda ch, props: [p['mean_intensity'].item() for p in props],
    #                'data': list(),
    #                },
    #               {'name': 'integrated_intensity',
    #                'desc': 'Integrated intensity of the ROIs.',
    #                'getter': lambda ch, props: [p['integrated_intensity'].item() for p in props],
    #                'data': list(),
    #                },
    #               {'name': 'x_weighted_centroid',
    #                'desc': 'Weighted Centroid X coordinates of the ROIs.',
    #                'getter': lambda ch, props: [p['weighted_centroid'][2].item() for p in props],
    #                'data': list(),
    #                },
    #               {'name': 'y_weighted_centroid',
    #                'desc': 'Weighted Centroid Y coordinates of the ROIs.',
    #                'getter': lambda ch, props: [p['weighted_centroid'][1].item() for p in props],
    #                'data': list(),
    #                },
    #               {'name': 'z_weighted_centroid',
    #                'desc': 'Weighted Centroid Z coordinates of the ROIs.',
    #                'getter': lambda ch, props: [p['weighted_centroid'][0].item() for p in props],
    #                'data': list(),
    #                },
    #               ]
    #
    # # Populate the data
    # key_values['Analysis_date_time'] = str(datetime.datetime.now())
    #
    # for ch, ch_spot_prop in enumerate(spots_properties):
    #     key_values[f'Nr_of_spots_ch{ch:02d}'] = len(ch_spot_prop)
    #
    #     key_values[f'Max_Intensity_ch{ch:02d}'] = max(x['integrated_intensity'] for x in ch_spot_prop)
    #     key_values[f'Max_Intensity_Roi_ch{ch:02d}'] = \
    #         ch_spot_prop[
    #             [x['integrated_intensity'] for x in ch_spot_prop].index(key_values[f'Max_Intensity_ch{ch:02d}'])][
    #             'label']
    #
    #     key_values[f'Min_Intensity_ch{ch:02d}'] = min(x['integrated_intensity'] for x in ch_spot_prop)
    #     key_values[f'Min_Intensity_Roi_ch{ch:02d}'] = \
    #         ch_spot_prop[
    #             [x['integrated_intensity'] for x in ch_spot_prop].index(key_values[f'Min_Intensity_ch{ch:02d}'])][
    #             'label']
    #
    #     key_values[f'Min-Max_intensity_ratio_ch{ch:02d}'] = key_values[f'Min_Intensity_ch{ch:02d}'] / key_values[
    #         f'Max_Intensity_ch{ch:02d}']
    #
    #     for prop in properties:
    #         if prop['name'] == 'roi_volume_units':
    #             prop['data'].extend(prop['getter']('VOXEL', ch_spot_prop))
    #         elif prop['name'] == 'roi_weighted_centroid_units':
    #             prop['data'].extend(prop['getter']('PIXEL', ch_spot_prop))
    #         else:
    #             prop['data'].extend(prop['getter'](ch, ch_spot_prop))
    #
    # return labels, properties, key_values


# _____________________________________
#
# ANALYSING LINES PATTERN. PATTERNS XXX
# Computing resolution
# _____________________________________


def analyze_resolution(image, pixel_size, pixel_units, axis, measured_band=.4, precision=None):
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
    if precision is not None:
        resolution_values = [round(x * pixel_size[axis], precision) for x in resolution_values]
    else:
        resolution_values = [x * pixel_size[axis] for x in resolution_values]

    key_values = dict()

    key_values['Analysis_date_time'] = str(datetime.datetime.now())

    for c, res in enumerate(resolution_values):
        key_values[f'ch{c:02d}_{resolution_method}_resolution'] = res.item()

    key_values['resolution_units'] = pixel_units[0]
    key_values['resolution_axis'] = axis
    key_values['measured_band'] = measured_band

    for c, indexes in enumerate(resolution_indexes):
        key_values[f'ch{c:02d}_peak_positions'] = [(peak_positions[c][ind].item(), peak_positions[c][ind + 1].item())
                                                   for ind in indexes]
        key_values[f'ch{c:02d}_peak_heights'] = [(peak_heights[c][ind].item(), peak_heights[c][ind + 1].item()) for ind
                                                 in indexes]
        key_values[f'ch{c:02d}_focus'] = z_planes[c].item()

    # plot.plot_peaks(profiles, peaks, peak_properties, resolution_values, resolution_indexes)

    return profiles, key_values


def _fit(profile, peaks_guess, amp=4, lower_amp=2, upper_amp=5, center_tolerance=1):
    guess = list()
    lower_bounds = list()
    upper_bounds = list()
    for p in peaks_guess:
        guess.append(p)  # peak center
        guess.append(amp)
        lower_bounds.append(p - center_tolerance)
        lower_bounds.append(lower_amp)
        upper_bounds.append(p + center_tolerance)
        upper_bounds.append(upper_amp)

    x = np.linspace(0, profile.shape[0] - 1, profile.shape[0])

    popt, pcov = curve_fit(multi_airy_fun, x, profile, p0=guess, bounds=(lower_bounds, upper_bounds))

    opt_peaks = popt[::2]
    opt_amps = [a / 4 for a in popt[1::2]]  # We normalize back the amplitudes to the unity

    return opt_peaks, opt_amps


def _compute_channel_resolution(channel, axis, prominence, measured_band, do_fitting=True, do_angle_refinement=False):
    """Computes the resolution on a pattern of lines with increasing separation"""
    # find the most contrasted z-slice
    z_stdev = np.std(channel, axis=(1, 2))
    z_focus = np.argmax(z_stdev)
    focus_slice = channel[z_focus]

    # TODO: verify angle and correct
    if do_angle_refinement:
        # Set a precision of 0.1 degree.
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 1800)
        h, theta, d = hough_line(focus_slice, theta=tested_angles)

    # Cut a band of that found peak
    # Best we can do now is just to cut a band in the center
    # We create a profiles along which we average signal
    axis_len = focus_slice.shape[-axis]
    weight_profile = np.zeros(axis_len)
    # Calculates a band of relative width 'image_fraction' to integrate the profile
    weight_profile[
    int((axis_len / 2) - (axis_len * measured_band / 2)):int((axis_len / 2) + (axis_len * measured_band / 2))] = 1
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
        if (prom / height) > prominence:  # This is calculating the prominence in relation to the local intensity
            ray_filtered_peak_pos.append(peak)
            ray_filtered_peak_heights.append(height)

    peak_positions = ray_filtered_peak_pos
    peak_heights = ray_filtered_peak_heights

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
