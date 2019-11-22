import unittest
#
#
# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)
#
#
# if __name__ == '__main__':
#     unittest.main()

from omero.gateway import BlitzGateway
import imageio
from metrics.interface import omero
from metrics.tools import segment_image, compute_distances_matrix, compute_spots_properties
from metrics.samples.argolight import compute_resolution
from metrics import plot
from credentials import HOST, PORT, USER, PASSWORD, GROUP


# def main_local():
#     # raw_img = imageio.volread('/Users/julio/PycharmProjects/OMERO.metrics/Images/201702_RI508_Argolight-1-1_010_WF_ALX.ome.tif')
#     raw_img = imageio.volread(
#         '/Users/julio/PycharmProjects/OMERO.metrics/Images/201702_RI508_Argolight-1-1_010_SIR_ALX.ome.tif')
#     # raw_img = imageio.volread('/home/julio/PycharmProjects/OMERO.metrics/Images/Test_image_SIR_ALX.ome.tif')
#     # raw_img = imageio.volread('/Users/julio/Desktop/20170215_R506_Argolight_SIM_001_visit_13_WF.ome.tif')
#     # raw_img = imageio.volread('/Users/julio/Desktop/20160215_R506_Argolight_SIM_001_visit_13_SIR_ALX.dv/20160215_R506_Argolight_SIM_001_visit_13_SIR_ALX.ome.tif')
#     n_channels = raw_img.shape[1]
#     x_size = raw_img.shape[2]
#     y_size = raw_img.shape[3]
#
#     labels_image = metrics.segment_image(image=raw_img)
#
#     spots_properties, spots_positions = metrics.compute_spots_properties(raw_img,labels_image)
#
#     spots_distances = metrics.compute_distances_matrix(spots_positions)
#
#     plot_homogeneity_map(data=BF,
#                          nb_of_channels=n_channels,
#                          x_dim=x_size,
#                          y_dim=y_size)
#
#     # out = metrics.analise_distances_matrix(positions)


def analyze_spots(image):

    image_shape = omero.get_image_shape(image)
    pixel_sizes = omero.get_pixel_sizes(image)

    # TODO: warn if xyz units are not the same

    raw_stack = omero.get_5d_stack(image)

    labels_stack = segment_image(image=raw_stack,
                                 min_distance=30,
                                 sigma=(1, 3, 3),
                                 method='local_max',
                                 hysteresis_levels=(.7, 1.))

    spots_properties, spots_positions = compute_spots_properties(raw_stack, labels_stack)

    spots_distances = compute_distances_matrix(positions=spots_positions,
                                               sigma=2.0,
                                               pixel_size=pixel_sizes)

    plot.plot_homogeneity_map(raw_stack=raw_stack,
                              spots_properties=spots_properties,
                              spots_positions=spots_positions,
                              labels_stack=labels_stack)

    plot.plot_distances_maps(distances=spots_distances,
                             x_dim=image_shape[-2],
                             y_dim=image_shape[-1])


def analyze_resolution(image, axis):

    image_shape = omero.get_image_shape(image)
    pixel_sizes = omero.get_pixel_sizes(image)
    pixel_units = omero.get_pixel_units(image)

    # TODO: warn if xyz units are not the same

    raw_stack = omero.get_5d_stack(image)

    profiles, peaks, peak_properties = compute_resolution(raw_stack, axis=axis)

    plot.plot_peaks(profiles, peaks, peak_properties)


def main(spots_image_id=7,
         vertical_stripes_image_id=3,
         horizontal_stripes_image_id=5):
    conn = BlitzGateway(username=USER,
                        passwd=PASSWORD,
                        group=GROUP,
                        port=PORT,
                        host=HOST)
    conn.connect()  # TODO: assert this somehow

    # spots_image = omero.get_image(connection=conn,
    #                               image_id=spots_image_id)
    # analyze_spots(spots_image)

    vertical_res_image = omero.get_image(connection=conn,
                                         image_id=vertical_stripes_image_id)
    analyze_resolution(vertical_res_image, 2)

    horizontal_res_image = omero.get_image(connection=conn,
                                           image_id=horizontal_stripes_image_id)
    analyze_resolution(horizontal_res_image, 1)

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

    conn.close()


if __name__ == '__main__':
    main()

