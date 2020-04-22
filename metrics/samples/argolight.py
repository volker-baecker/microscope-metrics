
from metrics.analysis.tools import segment_image, compute_distances_matrix, compute_spots_properties
from metrics.utils.utils import multi_airy_fun
# from metrics.plot import plot

import numpy as np
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from statistics import median
import datetime

# Creating logging services
import logging

module_logger = logging.getLogger('metrics.samples.argolight')


# ___________________________________________
# ANALYZING 'SPOTS' MATRIX. PATTERNS XXX
# Computing chromatic shifts, homogeneity,...
# ___________________________________________

def analyze_spots(image, pixel_size, pixel_size_units, low_corr_factors, high_corr_factors):
    labels = segment_image(image=image,
                           min_distance=30,
                           sigma=(1, 3, 3),
                           method='local_max',
                           low_corr_factors=low_corr_factors,
                           high_corr_factors=high_corr_factors,
                           )

    spots_properties, spots_positions = compute_spots_properties(image=image,
                                                                 labels=labels,
                                                                 remove_center_cross=False,
                                                                 )

    spots_distances = compute_distances_matrix(positions=spots_positions,
                                               sigma=2.0,
                                               pixel_size=pixel_size,
                                               )

    # Prepare key-value pairs
    key_values = dict()

    # Prepare tables
    properties = [{'name': 'channel',
                   'desc': 'Channel.',
                   'getter': lambda ch, props: [ch for x in props],
                   'data': list(),
                   },
                  {'name': 'mask_labels',
                   'desc': 'Labels of the mask ROIs.',
                   'getter': lambda ch, props: [p['label'] for p in props],
                   'data': list(),
                   },
                  {'name': 'volume',
                   'desc': 'Volume of the ROIs.',
                   'getter': lambda ch, props: [p['area'].item() for p in props],
                   'data': list(),
                   },
                  {'name': 'roi_volume_units',
                   'desc': 'Volume units for the ROIs.',
                   'getter': lambda ch, props: ['VOXEL' for n in range(len(props))],
                   'data': list(),
                   },
                  {'name': 'max_intensity',
                   'desc': 'Maximum intensity of the ROIs.',
                   'getter': lambda ch, props: [p['max_intensity'].item() for p in props],
                   'data': list(),
                   },
                  {'name': 'min_intensity',
                   'desc': 'Minimum intensity of the ROIs.',
                   'getter': lambda ch, props: [p['min_intensity'].item() for p in props],
                   'data': list(),
                   },
                  {'name': 'mean_intensity',
                   'desc': 'Mean intensity of the ROIs.',
                   'getter': lambda ch, props: [p['mean_intensity'].item() for p in props],
                   'data': list(),
                   },
                  {'name': 'integrated_intensity',
                   'desc': 'Integrated intensity of the ROIs.',
                   'getter': lambda ch, props: [p['integrated_intensity'].item() for p in props],
                   'data': list(),
                   },
                  {'name': 'x_weighted_centroid',
                   'desc': 'Weighted Centroid X coordinates of the ROIs.',
                   'getter': lambda ch, props: [p['weighted_centroid'][2].item() for p in props],
                   'data': list(),
                   },
                  {'name': 'y_weighted_centroid',
                   'desc': 'Weighted Centroid Y coordinates of the ROIs.',
                   'getter': lambda ch, props: [p['weighted_centroid'][1].item() for p in props],
                   'data': list(),
                   },
                  {'name': 'z_weighted_centroid',
                   'desc': 'Weighted Centroid Z coordinates of the ROIs.',
                   'getter': lambda ch, props: [p['weighted_centroid'][0].item() for p in props],
                   'data': list(),
                   },
                  {'name': 'roi_weighted_centroid_units',
                   'desc': 'Weighted centroid coordinates units for the ROIs.',
                   'getter': lambda ch, props: ['PIXEL' for n in range(len(props))],
                   'data': list(),
                   },
                  ]

    distances = [{'name': 'channel_A',
                  'desc': 'Channel A.',
                  'getter': lambda props: [props['channels'][0] for p in props['dist_3d']],
                  'data': list(),
                  },
                 {'name': 'channel_B',
                  'desc': 'Channel B.',
                  'getter': lambda props: [props['channels'][1] for p in props['dist_3d']],
                  'data': list(),
                  },
                 {'name': 'ch_A_roi_labels',
                  'desc': 'Labels of the ROIs in channel A.',
                  'getter': lambda props: props['labels_of_A'],
                  'data': list(),
                  },
                 {'name': 'ch_B_roi_labels',
                  'desc': 'Labels of the ROIs in channel B.',
                  'getter': lambda props: props['labels_of_B'],
                  'data': list(),
                  },
                 {'name': 'distance_3d',
                  'desc': 'Distance in 3d between Weighted Centroids of mutually closest neighbouring ROIs in channels A and B.',
                  'getter': lambda props: [p.item() for p in props['dist_3d']],
                  'data': list(),
                  },
                 {'name': 'distance_x',
                  'desc': 'Distance in X between Weighted Centroids of mutually closest neighbouring ROIs in channels A and B.',
                  'getter': lambda props: [p[2].item() for p in props['dist_zxy']],
                  'data': list(),
                  },
                 {'name': 'distance_y',
                  'desc': 'Distance in Y between Weighted Centroids of mutually closest neighbouring ROIs in channels A and B.',
                  'getter': lambda props: [p[1].item() for p in props['dist_zxy']],
                  'data': list(),
                  },
                 {'name': 'distance_z',
                  'desc': 'Distance in Z between Weighted Centroids of mutually closest neighbouring ROIs in channels A and B.',
                  'getter': lambda props: [p[0].item() for p in props['dist_zxy']],
                  'data': list(),
                  },
                 {'name': 'distances_units',
                  'desc': 'Weighted Centroid distances units.',
                  'getter': lambda props: [pixel_size_units[0] for n in props['dist_3d']],
                  'data': list(),
                  },
                 ]

    # Populate the data
    key_values['Analysis_date_time'] = str(datetime.datetime.now())

    for ch, ch_spot_prop in enumerate(spots_properties):

        key_values[f'Nr_of_spots_ch{ch:02d}'] = len(ch_spot_prop)

        key_values[f'Max_Intensity_ch{ch:02d}'] = max(x['integrated_intensity'] for x in ch_spot_prop)
        key_values[f'Max_Intensity_Roi_ch{ch:02d}'] = \
            ch_spot_prop[
                [x['integrated_intensity'] for x in ch_spot_prop].index(key_values[f'Max_Intensity_ch{ch:02d}'])][
                'label']

        key_values[f'Min_Intensity_ch{ch:02d}'] = min(x['integrated_intensity'] for x in ch_spot_prop)
        key_values[f'Min_Intensity_Roi_ch{ch:02d}'] = \
            ch_spot_prop[
                [x['integrated_intensity'] for x in ch_spot_prop].index(key_values[f'Min_Intensity_ch{ch:02d}'])][
                'label']

        key_values[f'Min-Max_intensity_ratio_ch{ch:02d}'] = key_values[f'Min_Intensity_ch{ch:02d}'] / key_values[
            f'Max_Intensity_ch{ch:02d}']

        for prop in properties:
            prop['data'].extend(prop['getter'](ch, ch_spot_prop))

    for ch, chs_dist in enumerate(spots_distances):
        for dists in distances:
            dists['data'].extend(dists['getter'](chs_dist))

            if dists['name'] == 'distance_3d':
                key_values[f'Median_3d_dist_ch{chs_dist["channels"][0]:02d}_ch{chs_dist["channels"][1]:02d}'] = \
                    median([d.item() for d in chs_dist['dist_3d']])

            if dists['name'] == 'distance_z':
                key_values[f'Median_z_dist_ch{chs_dist["channels"][0]:02d}_ch{chs_dist["channels"][1]:02d}'] = \
                    median([d[0].item() for d in chs_dist['dist_zxy']])

    key_values['Distance_units'] = pixel_size_units[0]

    return labels, properties, distances, key_values


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
