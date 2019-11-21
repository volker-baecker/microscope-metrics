import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
from itertools import product, permutations
import metrics
import numpy as np
from scipy.interpolate import griddata
from omero.gateway import BlitzGateway
from credentials import HOST, PORT, USER, PASSWORD, GROUP


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

    nb_of_channels = raw_stack.shape[2]
    x_dim = raw_stack.shape[3]
    y_dim = raw_stack.shape[4]

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

        ax[c].imshow(raw_stack[0, :, c, :, :].max(0), cmap='gray')
        ax[c].set_title('raw_channel_' + str(c))

        ax[c + nb_of_channels].imshow(labels_stack[0, :, c, :, :].max(0))
        ax[c + nb_of_channels].set_title('segmented_channel_' + str(c))

        ax[c + 2 * nb_of_channels].imshow(np.flipud(interpolated),
                                          extent=(0, x_dim, y_dim, 0),
                                          origin='lower',
                                          cmap=cm.hot,
                                          vmin=np.amin(raw_stack[0, :, c, :, :]),
                                          vmax=np.amax(raw_stack[0, :, c, :, :]))
        ax[c + 2 * nb_of_channels].plot(spots_positions[c][:, 2], spots_positions[c][:, 1], 'k.', ms=2)
        # ax[c + 2 * nb_of_channels].clim(np.amin(raw_img[:, c, :, :]), np.amax(raw_img[:, c, :, :]))
        ax[c + 2 * nb_of_channels].set_title('Max_intensity_channel_' + str(c))

    plt.show()


def plot_peaks(profiles, peaks, properties):
    fig, axes = plt.subplots(ncols=1, nrows=len(profiles), squeeze=False, figsize=(48, 24))

    for i, profile in enumerate(profiles):

        ax = axes.ravel()

        ax[i].plot(profile)
        ax[i].plot(peaks[i], profile[peaks[i]], "x")
        ax[i].vlines(x=peaks[i], ymin=profile[peaks[i]] - properties[i]["prominences"],
                   ymax=profile[peaks[i]], color="C1")
    plt.show()


def get_5d_stack(image):
    # We will further work with stacks of the shape TZCXY
    image_shape = (image.getSizeT(),
                   image.getSizeZ(),
                   image.getSizeC(),
                   image.getSizeX(),
                   image.getSizeY())
    nr_planes = image.getSizeT() * \
                image.getSizeZ() * \
                image.getSizeC()

    zct_list = list(product(range(image_shape[1]),
                            range(image_shape[2]),
                            range(image_shape[0])))
    pixels = image.getPrimaryPixels()
    pixels_type = pixels.getPixelsType()
    if pixels_type.value == 'float':
        data_type = pixels_type.value + str(pixels_type.bitSize)  # TODO: Verify this is working for all data types
    else:
        data_type = pixels_type.value
    stack = np.zeros((nr_planes,
                      image.getSizeX(),
                      image.getSizeY()), dtype=data_type)
    np.stack(list(pixels.getPlanes(zct_list)), out=stack)
    stack = np.reshape(stack, image_shape)

    return stack


def main_local():
    # raw_img = imageio.volread('/Users/julio/PycharmProjects/OMERO.metrics/Images/201702_RI508_Argolight-1-1_010_WF_ALX.ome.tif')
    raw_img = imageio.volread(
        '/Users/julio/PycharmProjects/OMERO.metrics/Images/201702_RI508_Argolight-1-1_010_SIR_ALX.ome.tif')
    # raw_img = imageio.volread('/home/julio/PycharmProjects/OMERO.metrics/Images/Test_image_SIR_ALX.ome.tif')
    # raw_img = imageio.volread('/Users/julio/Desktop/20170215_R506_Argolight_SIM_001_visit_13_WF.ome.tif')
    # raw_img = imageio.volread('/Users/julio/Desktop/20160215_R506_Argolight_SIM_001_visit_13_SIR_ALX.dv/20160215_R506_Argolight_SIM_001_visit_13_SIR_ALX.ome.tif')
    n_channels = raw_img.shape[1]
    x_size = raw_img.shape[2]
    y_size = raw_img.shape[3]

    labels_image = metrics.segment_image(image=raw_img)

    spots_properties, spots_positions = metrics.compute_spots_properties(raw_img,labels_image)

    spots_distances = metrics.compute_distances_matrix(spots_positions)

    plot_homogeneity_map(data=BF,
                         nb_of_channels=n_channels,
                         x_dim=x_size,
                         y_dim=y_size)

    # out = metrics.analise_distances_matrix(positions)


def analyze_spots(image):

    x_size = image.getSizeX()
    y_size = image.getSizeY()
    z_size = image.getSizeZ()
    c_size = image.getSizeC()
    t_size = image.getSizeT()

    pixels = image.getPrimaryPixels()

    x_pixel_size = pixels.getPhysicalSizeX().getValue()
    y_pixel_size = pixels.getPhysicalSizeY().getValue()
    z_pixel_size = pixels.getPhysicalSizeZ().getValue()

    x_pixel_size_unit = pixels.getPhysicalSizeX().getUnit().name
    y_pixel_size_unit = pixels.getPhysicalSizeY().getUnit().name
    z_pixel_size_unit = pixels.getPhysicalSizeZ().getUnit().name

    # TODO: warn if xyz units are not the same

    raw_stack = get_5d_stack(image)

    labels_stack = metrics.segment_image(image=raw_stack,
                                         min_distance=30,
                                         sigma=(1, 3, 3),
                                         method='local_max',
                                         hysteresis_levels=(.7, 1.))

    spots_properties, spots_positions = metrics.compute_spots_properties(raw_stack, labels_stack)

    spots_distances = metrics.compute_distances_matrix(positions=spots_positions,
                                                       sigma=2.0,
                                                       pixel_size=(z_pixel_size,
                                                                   x_pixel_size,
                                                                   y_pixel_size))

    plot_homogeneity_map(raw_stack=raw_stack,
                         spots_properties=spots_properties,
                         spots_positions=spots_positions,
                         labels_stack=labels_stack)

    plot_distances_maps(distances=spots_distances,
                        x_dim=1024,
                        y_dim=1024)


def analyze_resolution(image, axis):

    x_size = image.getSizeX()
    y_size = image.getSizeY()
    z_size = image.getSizeZ()
    c_size = image.getSizeC()
    t_size = image.getSizeT()

    pixels = image.getPrimaryPixels()

    x_pixel_size = pixels.getPhysicalSizeX().getValue()
    y_pixel_size = pixels.getPhysicalSizeY().getValue()
    z_pixel_size = pixels.getPhysicalSizeZ().getValue()

    x_pixel_size_unit = pixels.getPhysicalSizeX().getUnit().name
    y_pixel_size_unit = pixels.getPhysicalSizeY().getUnit().name
    z_pixel_size_unit = pixels.getPhysicalSizeZ().getUnit().name

    # TODO: warn if xyz units are not the same

    raw_stack = get_5d_stack(image)

    profiles, peaks, peak_properties = metrics.compute_resolution(raw_stack, axis=axis)

    plot_peaks(profiles, peaks, peak_properties)


def main(spots_image_id=7,
         vertical_stripes_image_id=3,
         horizontal_stripes_image_id=5):
    conn = BlitzGateway(username=USER,
                        passwd=PASSWORD,
                        group=GROUP,
                        port=PORT,
                        host=HOST)
    conn.connect()  # TODO: assert this somehow

    spots_image = conn.getObject("Image", spots_image_id)
    analyze_spots(spots_image)

    vertical_res_image = conn.getObject("Image", vertical_stripes_image_id)
    analyze_resolution(vertical_res_image, 2)

    horizontal_res_image = conn.getObject("Image", horizontal_stripes_image_id)
    analyze_resolution(horizontal_res_image, 1)

    conn.close()


if __name__ == '__main__':
    main()

