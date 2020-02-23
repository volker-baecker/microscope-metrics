# import unittest
#
#
# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)
#
#
# if __name__ == '__main__':
#     unittest.main()

import imageio
from metrics.interface import omero
from metrics.tools import segment_image, compute_distances_matrix, compute_spots_properties
from metrics.samples.argolight import compute_resolution
from metrics import plot
from credentials import HOST, PORT, USER, PASSWORD, GROUP
from dask import delayed

RUN_MODE = 'local'
# RUN_MODE = 'omero'

spots_image_id = 7
vertical_stripes_image_id = 3
horizontal_stripes_image_id = 5
spots_image_path = '/Users/julio/PycharmProjects/OMERO.metrics/Images/201702_RI510_Argolight-1-1_010_SIR_ALX.dv/201702_RI510_Argolight-1-1_010_SIR_ALX_THR.ome.tif'
vertical_stripes_image_path = '/Users/julio/PycharmProjects/OMERO.metrics/Images/201702_RI510_Argolight-1-1_004_SIR_ALX.dv/201702_RI510_Argolight-1-1_004_SIR_ALX_THR.ome.tif'
horizontal_stripes_image_path = '/Users/julio/PycharmProjects/OMERO.metrics/Images/201702_RI510_Argolight-1-1_005_SIR_ALX.dv/201702_RI510_Argolight-1-1_005_SIR_ALX_THR.ome.tif'


def get_local_data(path):
    raw_img = imageio.volread(path)
    pixel_sizes = (0.039, 0.039, 0.125)
    pixel_units = 'MICRON'

    return {'image_data': raw_img, 'pixel_sizes': pixel_sizes, 'pixel_units': pixel_units}


def get_omero_data(image_id):
    conn = omero.open_connection(username=USER,
                                 password=PASSWORD,
                                 group=GROUP,
                                 port=PORT,
                                 host=HOST)

    image = omero.get_image(conn, image_id)
    raw_img = omero.get_5d_stack(image)
    pixel_sizes = omero.get_pixel_sizes(image)
    pixel_units = omero.get_pixel_units(image)

    conn.close()

    return {'image_data': raw_img, 'pixel_sizes': pixel_sizes, 'pixel_units': pixel_units}


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


def analyze_spots(image, pixel_sizes):

    labels_stack = segment_image(image=image,
                                 min_distance=30,
                                 sigma=(1, 3, 3),
                                 method='local_max',
                                 hysteresis_levels=(.6, 0.8))

    spots_properties, spots_positions = compute_spots_properties(image, labels_stack, remove_center_cross=False)

    spots_distances = compute_distances_matrix(positions=spots_positions,
                                               sigma=2.0,
                                               pixel_size=pixel_sizes)

    plot.plot_homogeneity_map(raw_stack=image,
                              spots_properties=spots_properties,
                              spots_positions=spots_positions,
                              labels_stack=labels_stack)

    plot.plot_distances_maps(distances=spots_distances,
                             x_dim=image.shape[-2],
                             y_dim=image.shape[-1])


def analyze_resolution(image, pixel_sizes, pixel_units, axis):

    profiles, peaks, peak_properties, resolution_values = compute_resolution(image=image,
                                                                             axis=axis,
                                                                             prominence=.2,
                                                                             do_angle_refinement=False)

    plot.plot_peaks(profiles, peaks, peak_properties)


def main(run_mode):

    if run_mode == 'local':
        spots_image = get_local_data(spots_image_path)
        vertical_res_image = get_local_data(vertical_stripes_image_path)
        horizontal_res_image = get_local_data(horizontal_stripes_image_path)

    elif run_mode == 'omero':
        spots_image = get_omero_data(spots_image_id)
        vertical_res_image = get_omero_data(vertical_stripes_image_id)
        horizontal_res_image = get_omero_data(horizontal_stripes_image_id)

    analyze_spots(spots_image['image_data'], spots_image['pixel_sizes'])

    analyze_resolution(vertical_res_image['image_data'],
                       vertical_res_image['pixel_sizes'],
                       vertical_res_image['pixel_units'],
                       2)

    analyze_resolution(horizontal_res_image['image_data'],
                       horizontal_res_image['pixel_sizes'],
                       horizontal_res_image['pixel_units'],
                       1)

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

if __name__ == '__main__':
    main(RUN_MODE)

