
from metrics.analysis.tools import segment_image, compute_distances_matrix, compute_spots_properties
from metrics.utils.utils import multi_airy_fun, airy_fun, gaussian_fun

import numpy as np
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from statistics import median, mean
import datetime
from math import sin, asin, cos

# Import sample superclass
from .samples import Analyzer, Configurator

# Creating logging services
import logging

module_logger = logging.getLogger('metrics.samples.psf_beads')


def estimate_min_bead_distance(na, pixel_size):
    return 50


def calculate_nyquist(microscope_type, na, refractive_index, emission_wave, excitation_wave=None):
    if refractive_index is None:
        module_logger.warning('Refractive index is being guessed. Nyquist criteria will not be correct.')
        if na > .8:
            refractive_index = 1.5
        else:
            refractive_index = 1.0
    alpha = asin(na/refractive_index)
    # Theoretical resolutions for confocal microscope are only attained with very closed pinhole
    # We add a tolerance factor to render theoretical resolution more realistic
    tolerance = 1.6
    nyquist_delta = {}
    if microscope_type is None:
        module_logger.warning('Microscope type undefined to calculate Nyquist criterion. Falling back into Wide-Field')
        nyquist_delta = calculate_nyquist('wf', na, refractive_index, emission_wave, excitation_wave)
    elif microscope_type.lower() in ['wf', 'wide-field', 'widefield']:
        nyquist_delta['lateral'] = emission_wave / (4 * refractive_index * sin(alpha))
        nyquist_delta['axial'] = emission_wave / (2 * refractive_index * (1 - cos(alpha)))
        nyquist_delta['units'] = 'NANOMETER'
    elif microscope_type.lower() in ['confocal']:
        nyquist_delta['lateral'] = tolerance * excitation_wave / (8 * refractive_index * sin(alpha))
        nyquist_delta['axial'] = tolerance * excitation_wave / (4 * refractive_index * (1 - cos(alpha)))
        nyquist_delta['units'] = 'NANOMETER'
    else:
        module_logger.warning('Could not find microscope type to calculate Nyquist criterion. Falling back into Wide-Field')
        nyquist_delta = calculate_nyquist('wf', na, refractive_index, emission_wave, excitation_wave)

    return nyquist_delta


def calculate_theoretcal_resolution(microscope_type, na, refractive_index, emission_wave, excitation_wave=None):
    if refractive_index is None:
        module_logger.warning('Refractive index is being guessed. Theoretical resolutions will not be correct.')
        if na > .8:
            refractive_index = 1.5
        else:
            refractive_index = 1.0
    theoretical_res = {}
    if microscope_type is None:
        module_logger.warning('Microscope type undefined to calculate theoretical resolution. Falling back into Wide-Field')
        theoretical_res = calculate_theoretcal_resolution('wf', na, refractive_index, emission_wave, excitation_wave=excitation_wave)
    elif microscope_type.lower() in ['wf', 'wide-field', 'widefield']:
        theoretical_res['FWHM_lateral'] = .353 * emission_wave / na
        theoretical_res['Rayleigh_lateral'] = .61 * emission_wave / na
        theoretical_res['Rayleigh_axial'] = 2 * emission_wave * refractive_index / na ** 2
        theoretical_res['units'] = 'NANOMETER'
    elif microscope_type.lower() in ['confocal']:
        theoretical_res['FWHM_lateral'] = .353 * emission_wave / na
        theoretical_res['Rayleigh_lateral'] = .4 * emission_wave / na
        theoretical_res['Rayleigh_axial'] = 1.4 * emission_wave * refractive_index / na ** 2
        theoretical_res['units'] = 'NANOMETER'
    else:
        module_logger.warning('Could not find microscope type to calculate theoretical resolution. Falling back into Wide-Field')
        theoretical_res = calculate_theoretcal_resolution('wf', na, refractive_index, emission_wave, excitation_wave=excitation_wave)

    return theoretical_res


def _fit_gaussian(profile, guess=None):
    if guess is None:
        guess = [profile.min(), profile.max(), profile.argmax(), .8]
    x = np.linspace(0, profile.shape[0], profile.shape[0], endpoint=False)
    popt, pcov = curve_fit(gaussian_fun, x, profile, guess)

    fitted_profile = gaussian_fun(x, popt[0], popt[1], popt[2], popt[3])
    fwhm = popt[3] * 2.35482

    return fitted_profile, fwhm


class PSFBeadsConfigurator(Configurator):
    """This class handles the configuration properties of the psf_beads sample
    - Defines configuration properties
    - Helps in the generation of config files"""
    CONFIG_SECTION = 'PSF_BEADS'
    ANALYSES = ['beads']

    def __init__(self, config):
        super().__init__(config)


@PSFBeadsConfigurator.register_sample
class PSFBeadsAnalyzer(Analyzer):
    """This class handles a PSF beads sample:
    - Defines the logic of the associated analyses
    - Defines the creation of reports"""
    # TODO: Implemented multichannel

    def __init__(self, config=None):
        image_analysis_to_func = {'beads': self.analyze_beads}
        self.configurator = PSFBeadsConfigurator(config)
        super().__init__(config=config, image_analysis_to_func=image_analysis_to_func)

    @staticmethod
    def _analyze_bead(image):
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

        return (z_profile, y_profile, x_profile), \
               (z_fitted_profile, y_fitted_profile, x_fitted_profile), \
               (z_fwhm, y_fwhm, x_fwhm)

    @staticmethod
    def _find_beads(image, pixel_size, na, min_distance=None, sigma=None):  # , low_corr_factors, high_corr_factors):

        if min_distance is None:
            min_distance = estimate_min_bead_distance(na, pixel_size)

        image = np.squeeze(image)
        image_mip = np.max(image, axis=0)

        if sigma is not None:
            image_mip = gaussian(image=image_mip,
                                 multichannel=False,
                                 sigma=sigma,
                                 preserve_range=True)

        # Find bead centers
        positions_2d = peak_local_max(image=image_mip,
                                      threshold_rel=0.2,
                                      min_distance=5,
                                      indices=True)
        # Add the mas intensity value in z
        positions_3d = np.insert(positions_2d[:], 0, np.argmax(image[:, positions_2d[:, 0], positions_2d[:, 1]], axis=0), axis=1)

        nr_beads = positions_2d.shape[0]
        module_logger.info(f'Beads found: {nr_beads}')

        # Exclude beads too close to the edge
        edge_keep_mask = (positions_2d[:, 0] > min_distance) & \
                         (positions_2d[:, 0] < image_mip.shape[0] - min_distance) & \
                         (positions_2d[:, 1] > min_distance) & \
                         (positions_2d[:, 1] < image_mip.shape[1] - min_distance)
        module_logger.info(f'Beads too close to the edge: {nr_beads - np.sum(edge_keep_mask)}')

        # Exclude beads too close to each other
        proximity_keep_mask = np.ones((nr_beads, nr_beads),dtype=bool)
        for i, pos in enumerate(positions_2d):
            proximity_keep_mask[i] = (abs(positions_2d[:, 0] - pos[0]) > min_distance) |  \
                                    (abs(positions_2d[:, 1] - pos[1]) > min_distance)
            proximity_keep_mask[i,i] = True  # Correcting the diagonal
        proximity_keep_mask = np.all(proximity_keep_mask, axis=0)
        module_logger.info(f'Beads too close to each other: {nr_beads - np.sum(proximity_keep_mask)}')

        # Exclude beads too intense or too weak
        intensity_keep_mask = np.ones(nr_beads, dtype=bool)
        # TODO: Implement beads intensity filter
        module_logger.info(f'Beads too intense (probably more than one bead): {nr_beads - np.sum(intensity_keep_mask)}')

        keep_mask = edge_keep_mask & proximity_keep_mask & intensity_keep_mask
        module_logger.info(f'Beads kept for analysis: {np.sum(keep_mask)}')

        positions = positions_3d[keep_mask, :]
        pos_edge_disc = positions_3d[np.logical_not(edge_keep_mask), :]
        pos_proximity_disc = positions_3d[np.logical_not(proximity_keep_mask), :]
        pos_intensity_disc = positions_3d[np.logical_not(intensity_keep_mask), :]

        bead_images = list()
        for pos in positions:
            bead_images.append(image[:,
                                     (pos[1]-(min_distance//2)):(pos[1]+(min_distance//2)),
                                     (pos[2]-(min_distance//2)):(pos[2]+(min_distance//2))
                                     ]
                               )
        return bead_images, positions, pos_edge_disc, pos_proximity_disc, pos_intensity_disc

    def analyze_beads(self, image, config):
        """Analyzes images of sub-resolution beads in order to extract data on the optical performance of the microscope.

        :param image: image instance
        :param config: MetricsConfig instance defining analysis configuration.
                       Must contain the analysis parameters defined by the configurator

        :returns a list of images
                 a list of rois
                 a list of tags
                 a list of dicts
                 a dict containing table_names and tables
        """
        # Get some config parameters
        pixel_size_units = image['pixel_size_units']
        pixel_size = image['pixel_size']
        na = image['lens_na']
        refractive_index = image['refractive_index']
        magnification = image['lens_magnification']
        excitation_waves = image['excitation_waves']
        emission_waves = image['emission_waves']

        # TODO: Include microscope type into config
        # Get resolution parameters
        theoretical_resolution = calculate_theoretcal_resolution(microscope_type=None,
                                                                 na=na,
                                                                 refractive_index=refractive_index,
                                                                 emission_wave=emission_waves[0],
                                                                 excitation_wave=excitation_waves[0])
        nyquist = calculate_nyquist(microscope_type=None,
                                    na=na,
                                    refractive_index=refractive_index,
                                    emission_wave=emission_waves[0],
                                    excitation_wave=emission_waves[0])
        # TODO: validate units
        # Validating nyquist
        if pixel_size[1] > nyquist['lateral']:
            module_logger.warning('Nyquist criterion is not fulfilled in the lateral direction')
        if pixel_size[0] > nyquist['axial']:
            module_logger.warning('Nyquist criterion is not fulfilled in the axial direction')

        # Find the beads
        bead_images, \
            positions, \
            positions_edge_discarded, \
            positions_proximity_discarded, \
            positions_intensity_discarded = self._find_beads(image=image['image_data'],
                                                             pixel_size=pixel_size,
                                                             na=na,
                                                             min_distance=config.getint('min_distance', None),
                                                             sigma=config.getint('sigma', None))

        # Generate profiles and measure FWHM
        original_profiles = list()
        fitted_profiles = list()
        fwhm_values = list()
        for bead_image in bead_images:
            opr, fpr, fwhm = self._analyze_bead(bead_image)
            original_profiles.append(opr)
            fitted_profiles.append(fpr)
            fwhm = tuple([f * p for f, p in zip(fwhm, pixel_size)])
            fwhm_values.append(fwhm)

        # Prepare key-value pairs
        key_values = dict()

        # Prepare tables
        properties = [{'name': 'bead_image',
                       'desc': 'Reference to bead image.',
                       'getter': lambda i, pos, fwhm, bead_image: [0],  # We leave empty as it is populated after roi creation
                       'data': list(),
                       },
                      {'name': 'bead_roi',
                       'desc': 'Reference to bead roi.',
                       'getter': lambda i, pos, fwhm, bead_image: [0],  # We leave empty as it is populated after roi creation
                       'data': list(),
                       },
                      {'name': 'roi_label',
                       'desc': 'Label of the bead roi.',
                       'getter': lambda i, pos, fwhm, bead_image: [i],
                       'data': list(),
                       },
                      {'name': 'max_intensity',
                       'desc': 'Maximum intensity of the bead.',
                       'getter': lambda i, pos, fwhm, bead_image: [bead_image.max().item()],
                       'data': list(),
                       },
                      {'name': 'x_centroid',
                       'desc': 'Centroid X coordinate of the bead.',
                       'getter': lambda i, pos, fwhm, bead_image: [pos[2].item()],
                       'data': list(),
                       },
                      {'name': 'y_centroid',
                       'desc': 'Centroid Y coordinate of the bead.',
                       'getter': lambda i, pos, fwhm, bead_image: [pos[1].item()],
                       'data': list(),
                       },
                      {'name': 'z_centroid',
                       'desc': 'Centroid Z coordinate of the bead.',
                       'getter': lambda i, pos, fwhm, bead_image: [pos[0].item()],
                       'data': list(),
                       },
                      {'name': 'bead_centroid_units',
                       'desc': 'Centroid coordinates units for the bead.',
                       'getter': lambda i, pos, fwhm, bead_image: ['PIXEL'],
                       'data': list(),
                       },
                      {'name': 'x_fwhm',
                       'desc': 'FWHM in the X axis through the max intensity point of the bead.',
                       'getter': lambda i, pos, fwhm, bead_image: [fwhm[2].item()],
                       'data': list(),
                       },
                      {'name': 'y_fwhm',
                       'desc': 'FWHM in the Y axis through the max intensity point of the bead.',
                       'getter': lambda i, pos, fwhm, bead_image: [fwhm[1].item()],
                       'data': list(),
                       },
                      {'name': 'z_fwhm',
                       'desc': 'FWHM in the Z axis through the max intensity point of the bead.',
                       'getter': lambda i, pos, fwhm, bead_image: [fwhm[0].item()],
                       'data': list(),
                       },
                      {'name': 'fwhm_units',
                       'desc': 'FWHM units.',
                       'getter': lambda i, pos, fwhm, bead_image: [pixel_size_units[0]],
                       'data': list(),
                       },
                      ]

        # creating a table to store the profiles
        profiles_x = list()
        profiles_y = list()
        profiles_z = list()
        for i, (original_profile, fitted_profile) in enumerate(zip(original_profiles, fitted_profiles)):
            profiles_x.extend([{'name': f'raw_x_profile_bead_{i:02d}',
                                'desc': f'Intensity profile along x axis of bead {i:02d}.',
                                'getter': None,
                                'data': [p.item() for p in original_profile[2]]
                                },
                               {'name': f'fitted_x_profile_bead_{i:02d}',
                                'desc': f'Intensity profile along x axis of bead {i:02d}.',
                                'getter': None,
                                'data': [p.item() for p in fitted_profile[2]]
                                }
                               ]
                              )
            profiles_y.extend([{'name': f'raw_y_profile_bead_{i:02d}',
                                'desc': f'Intensity profile along x axis of bead {i:02d}.',
                                'getter': None,
                                'data': [p.item() for p in original_profile[1]]
                                },
                               {'name': f'fitted_y_profile_bead_{i:02d}',
                                'desc': f'Intensity profile along x axis of bead {i:02d}.',
                                'getter': None,
                                'data': [p.item() for p in fitted_profile[1]]
                                }
                               ]
                              )
            profiles_z.extend([{'name': f'raw_z_profile_bead_{i:02d}',
                                'desc': f'Intensity profile along x axis of bead {i:02d}.',
                                'getter': None,
                                'data': [p.item() for p in original_profile[0]]
                                },
                               {'name': f'fitted_z_profile_bead_{i:02d}',
                                'desc': f'Intensity profile along x axis of bead {i:02d}.',
                                'getter': None,
                                'data': [p.item() for p in fitted_profile[0]]
                                }
                               ]
                              )

        out_images = list()
        for i, img in enumerate(bead_images):
            out_images.append({'image_data': np.expand_dims(img, axis=(1, 2)),
                               'image_name': f'{image["image_name"]}_bead_{i}',
                               'image_desc': f'Bead crop.\nSource Image Id: {image["image_id"]}'})

        out_rois = []

        # Populate the data
        key_values['Analysis_date_time'] = str(datetime.datetime.now())

        for i, (pos, fwhm, bead_image) in enumerate(zip(positions, fwhm_values, bead_images)):
            for prop in properties:
                prop['data'].extend(prop['getter'](i, pos, fwhm, bead_image))

        out_rois.extend(self._pos_to_point_rois(positions_list=positions,
                                                description='Bead weighted centroid',
                                                individual_rois=True,
                                                stroke_color=(0, 255, 0, 150), fill_color=(50, 255, 50, 20)))
        out_rois.extend(self._pos_to_point_rois(positions_list=positions_edge_discarded,
                                                description='Discarded: edge',
                                                individual_rois=False,
                                                stroke_color=(255, 0, 0, 150), fill_color=(255, 50, 50, 20)))
        out_rois.extend(self._pos_to_point_rois(positions_list=positions_proximity_discarded,
                                                description='Discarded: proximity',
                                                individual_rois=False,
                                                stroke_color=(255, 0, 0, 150), fill_color=(255, 50, 50, 20)))
        out_rois.extend(self._pos_to_point_rois(positions_list=positions_intensity_discarded,
                                                description='Discarded: intensity',
                                                individual_rois=False,
                                                stroke_color=(255, 0, 0, 150), fill_color=(255, 50, 50, 20)))

        key_values['Nr_of_beads_analyzed'] = positions.shape[0]
        if positions.shape[0] == 0:
            key_values['Mean_X_FWHM'] = 'No mean could be calculated'
            key_values['Mean_Y_FWHM'] = 'No mean could be calculated'
            key_values['Mean_Z_FWHM'] = 'No mean could be calculated'
            key_values['FWHM_units'] = 'No mean could be calculated'
        else:
            key_values['Mean_X_FWHM'] = mean(properties[[p['name'] for p in properties].index('x_fwhm')]['data'])
            key_values['Mean_Y_FWHM'] = mean(properties[[p['name'] for p in properties].index('y_fwhm')]['data'])
            key_values['Mean_Z_FWHM'] = mean(properties[[p['name'] for p in properties].index('z_fwhm')]['data'])
            key_values['FWHM_units'] = properties[[p['name'] for p in properties].index('fwhm_units')]['data'][0]
        key_values['Theoretical_Rayleigh_lateral_resolution'] = theoretical_resolution['Rayleigh_lateral']
        key_values['Theoretical_Rayleigh_axial_resolution'] = theoretical_resolution['Rayleigh_axial']
        key_values['Theoretical_resolution_units'] = theoretical_resolution['units']

        out_tags = list()
        out_dicts = [key_values]
        out_tables = dict()
        if len(properties[0]['data']) > 0:
            out_tables.update({'Analysis_PSF_properties': properties,
                               'Analysis_PSF_X_profiles': profiles_x,
                               'Analysis_PSF_Y_profiles': profiles_y,
                               'Analysis_PSF_Z_profiles': profiles_z,
                               })

        return out_images, out_rois, out_tags, out_dicts, out_tables

    def _pos_to_point_rois(self, positions_list, description, individual_rois=True, stroke_color=(255, 255, 255, 255), fill_color=(255, 255, 255, 10)):
        roi_list = list()
        shapes = list()
        for i, pos in enumerate(positions_list):
            shape = self._create_shape(shape_type='point',
                                       x_pos=pos[2].item(),
                                       y_pos=pos[1].item(),
                                       z_pos=pos[0].item(),
                                       stroke_color=stroke_color,
                                       fill_color=fill_color,
                                       stroke_width=2)
            if individual_rois:
                roi_list.append(self._create_roi(shapes=[shape],
                                                 name=str(i),
                                                 description=description))
            else:
                shapes.append(shape)
        if not individual_rois:
            roi_list.append(self._create_roi(shapes=shapes,
                                             description=description))
        return roi_list


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
