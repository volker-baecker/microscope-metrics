import matplotlib.pyplot as plt
import imageio
import metrics
import numpy as np
from scipy.interpolate import griddata

# raw_img = imageio.volread('/home/julio/PycharmProjects/OMERO.metrics/Images/Test_image_WF_ALX.ome.tif')
# raw_img = imageio.volread('/Users/julio/Desktop/20160215_R506_Argolight_SIM_001_visit_13_WF.ome.tif')
raw_img = imageio.volread('/Users/julio/Desktop/20160215_R506_Argolight_SIM_001_visit_13_SIR_ALX.dv/20160215_R506_Argolight_SIM_001_visit_13_SIR_ALX.ome.tif')
n_channels = raw_img.shape[1]
x_size = raw_img.shape[2]
y_size = raw_img.shape[3]

label_images = list()
properties = list()

fig, axes = plt.subplots(ncols=n_channels, nrows=2, squeeze=False, figsize=(12, 6))

for c in range(n_channels):
    p, i = metrics.argolight_field(channel=raw_img[:,c,:,:])
    label_images.append(i)
    properties.append(p)

    positions = np.array([x['weighted_centroid'] for x in properties[c]])
    areas = np.array([x['max_intensity'] for x in properties[c]])
    grid_x, grid_y = np.mgrid[0:x_size:1, 0:y_size:1]
    interpolated = griddata(positions[:, 1:], areas[:], (grid_x, grid_y), method='cubic')
    ax = axes.ravel()
    # ax[c] = plt.subplot(1, 4, c + 1)
    ax[c] = plt.subplot(2, 4, c + 1)

    ax[c].imshow(interpolated.T, extent=(0, x_size, 0, y_size), origin='lower')
    ax[c].plot(positions[:, 1], positions[:, 2], 'k.', ms=5)
    ax[c].set_title('Interpolated' + str(c))
    # ax[c].axis('off')

    ax[c + n_channels].imshow(label_images[c].max(0)) #, extent=(0,512,0,512), origin='lower')
    ax[c + n_channels].set_title('raw_' + str(c))
    # ax[c + n_channels].axis('off')

plt.show()
