import numpy as np
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from scipy.signal import find_peaks
from statistics import median


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

    # Cut a band of that found peak
    # Best we can do now is just to cut a band in the center
    # We create a profiles along which we average signal
    axis_len = focus_slice.shape[-axis]
    weight_profile = np.zeros(axis_len)
    weight_profile[int(axis_len / 2.5):int(axis_len / 1.5)] = 1
    profile = np.average(focus_slice,
                         axis=-axis,
                         weights=weight_profile)

    normalized_profile = (profile - np.min(profile))/np.ptp(profile)

    # Find peaks: We implement Rayleigh limits that will be refined downstream
    peaks, properties = find_peaks(normalized_profile,
                                   height=.3,
                                   distance=2,
                                   prominence=.1,
                                   )

    return normalized_profile, peaks, properties


def compute_resolution(image, axis, prominence=0.2):
    profiles = list()
    peaks = list()
    peaks_properties = list()
    resolution_values = list()
    resolution_method = 'Rayleigh'

    for c in range(image.shape[2]):  # TODO: Deal with Time here
        prof, pk, pr, res = _compute_channel_resolution(image[0, :, c, :, :],
                                                        axis=axis,
                                                        prominence)
        profiles.append(prof)
        peaks.append(pk)
        peaks_properties.append(pr)
        resolution_values.append(res)



    return profiles, peaks, peaks_properties
