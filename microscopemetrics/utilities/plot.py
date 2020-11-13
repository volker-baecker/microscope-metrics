import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.interpolate import griddata


def plot_distances_maps(distances, x_dim, y_dim):
    """[((ch_A, ch_B), [[(s_x, s_y, s_z), dst, t_index],...]),...]"""
    nb_of_channels = 4
    fig, axes = plt.subplots(ncols=nb_of_channels, nrows=nb_of_channels, squeeze=False, figsize=(12, 12))

    for p in distances:
        positions_map = np.asarray([(x, y) for (z, x, y) in p['coord_of_A']])
        distances_map = np.asarray(p['dist_3d'])

        grid_x, grid_y = np.mgrid[0:x_dim:1, 0:y_dim:1]
        interpolated = griddata(positions_map, distances_map, (grid_x, grid_y), method='cubic')

        ax = axes.ravel()
        ax[(p['channels'][0] * 4) + p['channels'][1]].imshow(np.flipud(interpolated),
                                                             extent=(0, x_dim, y_dim, 0),
                                                             origin='lower',
                                                             cmap=cm.hot,
                                                             # vmin=np.amin(raw_stack[0, :, c, :, :]),
                                                             # vmax=np.amax(raw_stack[0, :, c, :, :])
                                                             )

    plt.show()


def plot_homogeneity_map(raw_stack, spots_properties, spots_positions, labels_stack):

    nb_of_channels = raw_stack.shape[1]
    x_dim = raw_stack.shape[-2]
    y_dim = raw_stack.shape[-1]

    fig, axes = plt.subplots(ncols=nb_of_channels, nrows=3, squeeze=False, figsize=(12, 6))

    for c in range(nb_of_channels):
        weighted_centroid = np.array([x['weighted_centroid'][0] for x in spots_properties[c]])
        areas = np.array([x['area'] for x in spots_properties[c]])
        max_intensity = np.array([x['max_intensity'] for x in spots_properties[c]])
        grid_x, grid_y = np.mgrid[0:x_dim, 0:y_dim]
        try:
            interpolated = griddata(spots_positions[c][:, 1:], max_intensity, (grid_x, grid_y), method='linear')
        except Exception as e:
            # TODO: Log a warning
            interpolated = np.zeros((256, 256))

        ax = axes.ravel()
        ax[c] = plt.subplot(3, 4, c + 1)

        ax[c].imshow(raw_stack[..., c, :, :].max(0), cmap='gray')
        ax[c].set_title('raw_channel_' + str(c))

        ax[c + nb_of_channels].imshow(labels_stack[..., c, :, :].max(0))
        ax[c + nb_of_channels].set_title('segmented_channel_' + str(c))

        ax[c + 2 * nb_of_channels].imshow(np.flipud(interpolated),
                                          extent=(0, x_dim, y_dim, 0),
                                          origin='lower',
                                          cmap=cm.hot,
                                          vmin=np.amin(raw_stack[..., c, :, :]),
                                          vmax=np.amax(raw_stack[..., c, :, :]))
        ax[c + 2 * nb_of_channels].plot(spots_positions[c][:, 2], spots_positions[c][:, 1], 'k.', ms=2)
        # ax[c + 2 * nb_of_channels].clim(np.amin(raw_img[:, c, :, :]), np.amax(raw_img[:, c, :, :]))
        ax[c + 2 * nb_of_channels].set_title('Max_intensity_channel_' + str(c))

    plt.show()


def plot_peaks(profiles, peaks, properties, resolutions, res_indexes):
    fig, axes = plt.subplots(ncols=1, nrows=len(profiles), squeeze=False, figsize=(48, 24))

    for c, profile in enumerate(profiles):

        ax = axes.ravel()

        ax[c].plot(profile)
        ax[c].plot(peaks[c], profile[peaks[c]], "x")
        ax[c].vlines(x=peaks[c],
                     ymin=profile[peaks[c]] - properties[c]['prominences'],
                     ymax=profile[peaks[c]],
                     color='green')
        for i in res_indexes[c]:
            ax[c].vlines(x=peaks[c][i],
                         ymin=profile[peaks[c][i]] + .02,
                         ymax=1,
                         color='red',
                         linestyles='dashed')
            ax[c].vlines(x=peaks[c][i+1],
                         ymin=profile[peaks[c][i+1]] + .02,
                         ymax=1,
                         color='red',
                         linestyles='dashed')
            ax[c].annotate(s=f'{resolutions[c]:.3f}',
                           xy=(((peaks[c][i] + peaks[c][i+1]) / 2), profile[peaks[c][i+1]] + .1),
                           # ((vline_value - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                           # xycoords='axes fraction', verticalalignment='right',
                           horizontalalignment='center',
                           fontsize='xx-large',
                           fontweight='bold',
                           # rotation=270
                           )

    # plt.legend()
    plt.show()

