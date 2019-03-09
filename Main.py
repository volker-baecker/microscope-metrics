import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
import metrics
import numpy as np
from scipy.interpolate import griddata

raw_img = imageio.volread('/home/julio/PycharmProjects/OMERO.metrics/Images/Test_image_WF_ALX.ome.tif')
# raw_img = imageio.volread('/home/julio/PycharmProjects/OMERO.metrics/Images/Test_image_SIR_ALX.ome.tif')
# raw_img = imageio.volread('/Users/julio/Desktop/20160215_R506_Argolight_SIM_001_visit_13_WF.ome.tif')
# raw_img = imageio.volread('/Users/julio/Desktop/20160215_R506_Argolight_SIM_001_visit_13_SIR_ALX.dv/20160215_R506_Argolight_SIM_001_visit_13_SIR_ALX.ome.tif')
n_channels = raw_img.shape[1]
x_size = raw_img.shape[2]
y_size = raw_img.shape[3]

label_images = list()
properties = list()
positions = list()

fig, axes = plt.subplots(ncols=n_channels, nrows=3, squeeze=False, figsize=(12, 6))

for c in range(n_channels):
    p, i = metrics.argolight_field(channel=raw_img[:,c,:,:])
    label_images.append(i)
    properties.append(p)

    positions.append(np.array([x['weighted_centroid'] for x in properties[c]]))
    weighted_centroid = np.array([x['weighted_centroid'][0] for x in properties[c]])
    areas = np.array([x['area'] for x in properties[c]])
    max_intensity = np.array([x['max_intensity'] for x in properties[c]])
    grid_x, grid_y = np.mgrid[0:x_size:1, 0:y_size:1]
    interpolated = griddata(positions[-1][:, 1:], max_intensity[:], (grid_x, grid_y), method='cubic')

    ax = axes.ravel()
    ax[c] = plt.subplot(3, 4, c + 1)

    ax[c].imshow(raw_img[:, c, :, :].max(0), cmap='gray')
    ax[c].set_title('raw_channel_' + str(c))

    ax[c + n_channels].imshow(label_images[c].max(0))
    ax[c + n_channels].set_title('segmented_channel_' + str(c))

    ax[c + 2 * n_channels].imshow(interpolated.T,
                                  extent=(0, x_size, 0, y_size),
                                  origin='lower',
                                  cmap=cm.hot,
                                  vmin=np.amin(raw_img[:, c, :, :]),
                                  vmax=np.amax(raw_img[:, c, :, :]))
    ax[c + 2 * n_channels].plot(positions[-1][:, 1], positions[-1][:, 2], 'k.', ms=2)
    # ax[c + 2 * n_channels].clim(np.amin(raw_img[:, c, :, :]), np.amax(raw_img[:, c, :, :]))
    ax[c + 2 * n_channels].set_title('Max_intensity_channel_' + str(c))

# fig.colorbar(interpolated.T)

out = metrics.analise_distances_matrix(positions)

plt.show()
