import matplotlib.pyplot as plt
import imageio
import metrics
import numpy as np
from scipy.interpolate import griddata

raw_img = imageio.volread('/home/julio/PycharmProjects/OMERO.metrics/Images/Test_image_WF_ALX.ome.tif')

n_channels = raw_img.shape[1]

label_images = list()
properties = list()

for c in range(n_channels):
    p, i = metrics.argolight_field(channel=raw_img[:,c,:,:])
    label_images.append(i)
    properties.append(p)

positions = np.array([x['weighted_centroid'] for x in properties[c]])
# positions.append([x['weighted_centroid'] for x in properties[c]])
i = np.zeros((1, 15, 15))
interpolated = griddata(positions[:, :2], positions[:, 2:], i)
print(interpolated)
fig, axes = plt.subplots(ncols=n_channels, nrows=2, squeeze=False, figsize=(12, 6))
ax = axes.ravel()
for c in range(n_channels):
    # ax[c] = plt.subplot(1, 4, c + 1)
    ax[c] = plt.subplot(2, 4, c + 1)

    ax[c].imshow(label_images[c].max(0))
    ax[c].set_title('segmented_' + str(c))
    # ax[c].axis('off')

    ax[c + n_channels].imshow(raw_img[:, c, :, :].max(0))
    ax[c + n_channels].set_title('raw_' + str(c))
    # ax[c + n_channels].axis('off')

plt.show()
