"""These are some possibly useful code snippets"""

# Thresholding and labeling
import matplotlib.pyplot as plt
import imageio
from skimage.filters import threshold_otsu, apply_hysteresis_threshold
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, cube, octahedron, ball
from scipy.spatial.distance import cdist

raw_img = imageio.volread('/home/julio/PycharmProjects/OMERO.metrics/Images/Test_image_WF_ALX.ome.tif')

n_channels = raw_img.shape[1]

properties = []

positions = []

fig, axes = plt.subplots(ncols=n_channels, nrows=2, squeeze=False, figsize=(12, 6))
ax = axes.ravel()

for c in range(n_channels):

    properties.append([])

    thresh = threshold_otsu(raw_img[:, c, :, :])

    # We may try here hysteresis thresholding
    thresholded = apply_hysteresis_threshold(raw_img[:, c, :, :], low=(thresh * .9), high=(thresh * 1.1))

    bw = closing(thresholded, cube(2))
    cleared = clear_border(bw)
    label_image = label(cleared)
    regions = regionprops(label_image, raw_img[:, c, :, :])

    # for region in regions:
    #     properties[c].append({'label': region.label,
    #                           'area': region.area,
    #                           # 'convex_area': region.convex_area,
    #                           'centroid': region.centroid,
    #                           'weighted_centroid': region.weighted_centroid,
    #                           'max_intensity': region.max_intensity,
    #                           'mean_intensity': region.mean_intensity,
    #                           'min_intensity': region.min_intensity
    #                           })
    #
    # positions.append([x['weighted_centroid'] for x in properties[c]])

    # ax[c] = plt.subplot(1, 4, c + 1)
    ax[c] = plt.subplot(2, 4, c + 1)

    ax[c].imshow(label_image.max(0))
    ax[c].set_title('segmented_' + str(c))
    # ax[c].axis('off')

    ax[c + n_channels].imshow(raw_img[:, c, :, :].max(0))
    ax[c + n_channels].set_title('raw_' + str(c))
    # ax[c + n_channels].axis('off')



# distances = cdist(positions[1], positions[2])
#
# min_distances = [x.min() for x in distances]
# print('Nr of regions in 0: ', len(positions[0]))
# print('Nr of regions in 1: ', len(positions[1]))
# print('Nr of regions in 2: ', len(positions[2]))
# print('Nr of regions in 3: ', len(positions[3]))
# print("nr of pair distances 1-2: ", len(min_distances))
# print(min_distances)


plt.show()
