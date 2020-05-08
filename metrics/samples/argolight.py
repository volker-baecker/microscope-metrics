
from metrics.analysis.tools import segment_image, compute_distances_matrix, compute_spots_properties
from metrics.utils.utils import multi_airy_fun, airy_fun
# from metrics.plot import plot

import numpy as np
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from statistics import median
import datetime

# Import sample superclass
from .samples import Sample

# Creating logging services
import logging

module_logger = logging.getLogger('metrics.samples.argolight')


class ArgolightSample(Sample):
    """This class handles the Argolight sample:
    - Defines the logic of the associated analyses
    - Defines the creation of reports"""

    def __init__(self, config=None):
        analysis_to_func = {'do_spots': self.analyze_spots,
                            'do_resolution': self.analyze_resolution}
        super().__init__(config=config, analysis_to_func=analysis_to_func)

    def analyze_spots(self, image, config):
        """Analyzes 'SPOTS' matrix pattern from the argolight sample. It computes chromatic shifts, homogeneity,..

        :param image: image instance
        :param config: MetricsConfig instance defining analysis configuration.
                       Must contain the analysis parameters defined by the configurator

        :returns a list of images
                 a list of rois
                 a list of tags
                 a list of dicts
                 a dict containing table_names and tables
        """
        module_logger.info(f'Analyzing spots image...')

        # Calculating the distance between spots in pixels with a security margin
        min_distance = round((config.getfloat('spots_distance') * 0.3) / max(image['pixel_size'][-2:]))

        # Calculating the maximum tolerated distance in microns for the same spot in a different channels
        max_distance = config.getfloat('spots_distance') * .4

        labels = segment_image(image=image['image_data'],
                               min_distance=min_distance,
                               sigma=config.getlistfloat('sigma', '[1, 3, 3]'),
                               method='local_max',
                               low_corr_factors=config.getlistfloat('lower_threshold_correction_factors'),
                               high_corr_factors=config.getlistfloat('upper_threshold_correction_factors'),
                               )

        spots_properties, spots_positions = compute_spots_properties(image=image['image_data'],
                                                                     labels=labels,
                                                                     remove_center_cross=False,
                                                                     )

        spots_distances = compute_distances_matrix(positions=spots_positions,
                                                   max_distance=max_distance,
                                                   pixel_size=image['pixel_size'],
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
                      'getter': lambda props: [image['pixel_size_units'][0] for n in props['dist_3d']],
                      'data': list(),
                      },
                     ]

        # Create some rois to mark the positions
        out_rois = list()

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

            channel_shapes = [self._create_shape(shape_type='point',
                                                 name=f'{p["label"]}',
                                                 x_pos=p['weighted_centroid'][2].item(),
                                                 y_pos=p['weighted_centroid'][1].item(),
                                                 z_pos=p['weighted_centroid'][0].item(),
                                                 c_pos=ch,
                                                 t_pos=0,
                                                 ) for p in ch_spot_prop]

            out_rois.append(self._create_roi(shapes=channel_shapes,
                                             name=f'{ch}',
                                             description=f'weighted centroids channel {ch}'))

        # TODO: match roi labels with mask labels
        for ch, chs_dist in enumerate(spots_distances):
            for dists in distances:
                dists['data'].extend(dists['getter'](chs_dist))

                if dists['name'] == 'distance_3d':
                    key_values[f'Median_3d_dist_ch{chs_dist["channels"][0]:02d}_ch{chs_dist["channels"][1]:02d}'] = \
                        median([d.item() for d in chs_dist['dist_3d']])

                if dists['name'] == 'distance_z':
                    key_values[f'Median_z_dist_ch{chs_dist["channels"][0]:02d}_ch{chs_dist["channels"][1]:02d}'] = \
                        median([d[0].item() for d in chs_dist['dist_zxy']])

        key_values['Distance_units'] = image['pixel_size_units'][0]

        # We need to add a time dimension to the labels image
        labels = np.expand_dims(labels, 2)
        out_images = [{'image_data': labels,
                       'image_name': f'{image["image_name"]}_roi_masks',
                       'image_desc': f'Image with detected spots labels. Image intensities correspond to roi labels.'}]

        out_tags = []
        out_dicts = [key_values]
        out_tables = {'Analysis_argolight_D_properties': properties,
                      'Analysis_argolight_D_distances': distances}

        return out_images, out_rois, out_tags, out_dicts, out_tables

    def analyze_vertical_resolution(self, image, config):
        """A intermediate function to specify the axis to be analyzed"""
        return self.analyze_resolution(image=image, axis=1, config=config)

    def analyze_horizontal_resolution(self, image, config):
        """A intermediate function to specify the axis to be analyzed"""
        return self.analyze_resolution(image=image, axis=2, config=config)

    def analyze_resolution(self, image, axis, config):  # axis, measured_band=.4, **kwargs):
        """Analyzes 'LINES' pattern from the argolight sample. It computes resolution along a specific axis,..

        :param image: image instance
        :param axis: axis in which resolution is measured
        :param config: MetricsConfig instance defining analysis configuration.
                       Must contain the analysis parameters defined by the configurator

        :returns a list of images
                 a list of rois
                 a list of tags
                 a list of dicts
                 a dict containing table_names and tables
        """
        profiles, \
            z_planes, \
            peak_positions, \
            peak_heights, \
            resolution_values, \
            resolution_indexes, \
            resolution_method = _compute_resolution(image=image['image_data'],
                                                    axis=axis,
                                                    measured_band=config.getfloat('res_measurement_band'),
                                                    prominence=.264,
                                                    do_angle_refinement=False)
        # resolution in native units
        resolution_values = [x * image['pixel_size'][axis] for x in resolution_values]

        key_values = dict()

        key_values['Analysis_date_time'] = str(datetime.datetime.now())

        for ch, res in enumerate(resolution_values):
            key_values[f'ch{ch:02d}_{resolution_method}_resolution'] = res.item()

        key_values['resolution_units'] = image['pixel_size_units'][axis]
        key_values['resolution_axis'] = axis
        key_values['measured_band'] = config['res_measurement_band']

        for ch, indexes in enumerate(resolution_indexes):
            key_values[f'ch{ch:02d}_peak_positions'] = [(peak_positions[ch][ind].item(), peak_positions[ch][ind + 1].item())
                                                       for ind in indexes]
            key_values[f'ch{ch:02d}_peak_heights'] = [(peak_heights[ch][ind].item(), peak_heights[ch][ind + 1].item()) for ind
                                                     in indexes]
            key_values[f'ch{ch:02d}_focus'] = z_planes[ch].item()

        out_images = []
        out_rois = []
        out_tables = {}

        # Populate tables and rois
        for ch, profile in enumerate(profiles):
            out_tables.update({f'Analysis_argolight_E_ch{ch:02d}': _profile_to_table(profile)})
            shapes = list()
            for pos in key_values[f'ch{ch:02d}_peak_positions']:
                for peak in pos:
                    # Measurements are taken at center of pixel so we add .5 pixel to peak positions
                    if axis == 1:  # Y resolution -> horizontal rois
                        axis_len = image['image_data'].shape[-2]
                        x1_pos = int((axis_len / 2) - (axis_len * config.getfloat('res_measurement_band') / 2))
                        y1_pos = peak + .5
                        x2_pos = int((axis_len / 2) + (axis_len * config.getfloat('res_measurement_band') / 2))
                        y2_pos = peak + .5
                    elif axis == 2:  # X resolution -> vertical rois
                        axis_len = image['image_data'].shape[-1]
                        y1_pos = int((axis_len / 2) - (axis_len * config.getfloat('res_measurement_band') / 2))
                        x1_pos = peak + .5
                        y2_pos = int((axis_len / 2) + (axis_len * config.getfloat('res_measurement_band') / 2))
                        x2_pos = peak + .5

                    shapes.append(self._create_shape(shape_type='line',
                                                     x1_pos=x1_pos, y1_pos=y1_pos, x2_pos=x2_pos, y2_pos=y2_pos,
                                                     c_pos=ch, z_pos=z_planes[ch],
                                                     stroke_color=(255, 255, 255, 150)))
            out_rois.append(self._create_roi(shapes=shapes, name=f'ch{ch:02d}',
                                             description=f'Lines where highest Rayleigh resolution was found in channel {ch}'))

        out_tags = []
        out_dicts = [key_values]

        return out_images, out_rois, out_tags, out_dicts, out_tables


def _profile_to_table(profile):
    table = list()
    table.append({'name': 'raw_profile',
                  'desc': f'Average intensity profile on measured band along measured axis and highest contrast z plane.',
                  'data': [v.item() for v in profile[0,:]]
                  })
    for p in range(1, profile.shape[0]):
        table.append({'name': f'fitted_profile_peak{p:02d}',
                      'desc': f'Fitted Airy function profile at peak {p:02d}',
                      'data': [v.item() for v in profile[p, :]]})

    return table


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
    # opt_amps = [a / 4 for a in popt[1::2]]  # We normalize back the amplitudes to the unity
    opt_amps = popt[1::2]

    fitted_profiles = np.zeros((len(peaks_guess), profile.shape[0]))
    for i, (centre, amplitude) in enumerate(zip(opt_peaks, opt_amps)):
        fitted_profiles[i, :] = airy_fun(x, centre=centre, amp=amplitude)

    return opt_peaks, opt_amps, fitted_profiles


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
    weight_profile[ \
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
        peak_positions, peak_heights, fitted_profiles = _fit(normalized_profile, peak_positions)
        normalized_profile = np.append(np.expand_dims(normalized_profile, 0), fitted_profiles, axis=0)

    # Find the closest peaks to return it as a measure of resolution
    peaks_distances = [abs(a - b) for a, b in zip(peak_positions[0:-2], peak_positions[1:-1])]
    res = min(peaks_distances)
    res_indices = [i for i, x in enumerate(peaks_distances) if x == res]

    return normalized_profile, z_focus, peak_positions, peak_heights, res, res_indices


def _compute_resolution(image, axis, measured_band, prominence, do_angle_refinement=False):
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
