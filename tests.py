
# import configuration parser
from metrics.utils.utils import MetricsConfig

# import logging
import logging

# import Argolight analysis tools
from metrics.samples import argolight

# import other sample types analysis tools

# TODO: these imports should go somewhere else in the future
import imageio
import numpy as np
from itertools import product
from metrics.interface import omero
from credentials import *

# TODO: these constants should go somewhere else in the future. Basically are recovered by OMERO scripting interface
# RUN_MODE = 'local'
RUN_MODE = 'omero'

# spots_image_id = 7
# vertical_stripes_image_id = 3
# horizontal_stripes_image_id = 5
spots_image_path = '/Users/julio/PycharmProjects/OMERO.metrics/Images/201702_RI510_Argolight-1-1_010_SIR_ALX.dv/201702_RI510_Argolight-1-1_010_SIR_ALX_THR.ome.tif'
vertical_stripes_image_path = '/Users/julio/PycharmProjects/OMERO.metrics/Images/201702_RI510_Argolight-1-1_004_SIR_ALX.dv/201702_RI510_Argolight-1-1_004_SIR_ALX_THR.ome.tif'
horizontal_stripes_image_path = '/Users/julio/PycharmProjects/OMERO.metrics/Images/201702_RI510_Argolight-1-1_005_SIR_ALX.dv/201702_RI510_Argolight-1-1_005_SIR_ALX_THR.ome.tif'
config_file = 'my_microscope_config.ini'

# Creating logging services
logger = logging.getLogger('metrics')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler('metrics.log')
fh.setLevel(logging.ERROR)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def get_local_data(path):
    raw_img = imageio.volread(path)
    logger.info(f'Reading image {path}')
    pixel_size = (0.125, 0.039, 0.039)
    pixel_units = 'MICRON'

    return {'image_data': raw_img, 'pixel_size': pixel_size, 'pixel_units': pixel_units}


def get_omero_data(image_id):
    conn = omero.open_connection(username=USER,
                                 password=PASSWORD,
                                 group=GROUP,
                                 port=PORT,
                                 host=HOST)
    image = omero.get_image(conn, image_id)
    image_name = image.getName()
    raw_img = omero.get_intensities(image)
    # Images from OMERO come in zctxy dimensions and locally as zcxy.
    # The easyest for the moment is to remove t from the omero image
    if raw_img.shape[2] == 1:
        raw_img = np.squeeze(raw_img, 2)
    else:
        raise Exception("Image has a time dimension. Time is not yet implemented for this analysis")
    pixel_size = omero.get_pixel_size(image)
    pixel_units = omero.get_pixel_units(image)

    conn.close()

    return {'image_data': raw_img, 'image_name': image_name, 'pixel_size': pixel_size, 'pixel_units': pixel_units}


def save_data_table(table_name, names, desc, data, image_id):

    conn = omero.open_connection(username=USER,
                                 password=PASSWORD,
                                 group=GROUP,
                                 port=PORT,
                                 host=HOST)

    try:
        table_ann = omero.create_annotation_table(connection=conn,
                                                  table_name=table_name,
                                                  column_names=names,
                                                  column_descriptions=desc,
                                                  values=data)

        image = omero.get_image(conn, image_id)

        omero.link_annotation(image, table_ann)
    finally:
        conn.close()


def save_data_key_values(key_values, image_id):

    conn = omero.open_connection(username=USER,
                                 password=PASSWORD,
                                 group=GROUP,
                                 port=PORT,
                                 host=HOST)
    try:
        map_ann = omero.create_annotation_map(connection=conn,
                                              annotation=key_values,
                                              client_editable=True)
        image = omero.get_image(conn, image_id)
        omero.link_annotation(image, map_ann)

    finally:
        conn.close()


def create_laser_power_keys(laser_lines, units, dataset_id):
    conn = omero.open_connection(username=USER,
                                 password=PASSWORD,
                                 group=GROUP,
                                 port=PORT,
                                 host=HOST)

    key_values = {str(k): '' for k in laser_lines}
    key_values['power_units'] = units

    try:
        map_ann = omero.create_annotation_map(connection=conn,
                                              annotation=key_values,
                                              client_editable=True)
        dataset = omero.get_dataset(conn, dataset_id)
        omero.link_annotation(dataset, map_ann)

    finally:
        conn.close()


def save_spots_point_rois(names: list, data: list, image_id):
    conn = omero.open_connection(username=USER,
                                 password=PASSWORD,
                                 group=GROUP,
                                 port=PORT,
                                 host=HOST)
    image = omero.get_image(conn, image_id)
    nb_channels = image.getSizeC()

    for c in range(nb_channels):
        shapes = list()
        for x, y, z, l in zip(data[names.index(f'ch{c:02d}_XWeightedCentroid')][0],
                              data[names.index(f'ch{c:02d}_YWeightedCentroid')][0],
                              data[names.index(f'ch{c:02d}_ZWeightedCentroid')][0],
                              data[names.index(f'ch{c:02d}_MaskLabels')][0],
                              ):
            shapes.append(omero.create_shape_point(x, y, z, shape_name=f'ch{c:02d}_{l}'))

        omero.create_roi(connection=conn,
                         image=image,
                         shapes=shapes)

    conn.close()


def save_line_rois(data: dict, image_id):
    conn = omero.open_connection(username=USER,
                                 password=PASSWORD,
                                 group=GROUP,
                                 port=PORT,
                                 host=HOST)
    image = omero.get_image(conn, image_id)
    nb_channels = image.getSizeC()

    for c in range(nb_channels):
        shapes = list()
        for l in range(len(data[f'ch{c:02d}_peak_positions'])):
            for p in range(2):
                if data['resolution_axis'] == 1:  # Y resolution -> horizontal rois
                    axis_len = image.getSizeX()
                    x1_pos = int((axis_len / 2) - (axis_len * data['measured_band'] / 2))
                    y1_pos = data[f'ch{c:02d}_peak_positions'][l][p]
                    x2_pos = int((axis_len / 2) + (axis_len * data['measured_band'] / 2))
                    y2_pos = data[f'ch{c:02d}_peak_positions'][l][p]
                elif data['resolution_axis'] == 2:  # X resolution -> vertical rois
                    axis_len = image.getSizeY()
                    y1_pos = int((axis_len / 2) - (axis_len * data['measured_band'] / 2))
                    x1_pos = data[f'ch{c:02d}_peak_positions'][l][p]
                    y2_pos = int((axis_len / 2) + (axis_len * data['measured_band'] / 2))
                    x2_pos = data[f'ch{c:02d}_peak_positions'][l][p]
                else:
                    raise ValueError('Only axis 1 and 2 (X and Y) are supported')

                line = omero.create_shape_line(x1_pos=x1_pos + .5,
                                               y1_pos=y1_pos + .5,
                                               x2_pos=x2_pos + .5,
                                               y2_pos=y2_pos + .5,
                                               z_pos=data[f'ch{c:02d}_focus'],
                                               t_pos=0,
                                               line_name=f'ch{c:02d}_{l}_{p}',
                                               # stroke_color=,
                                               stroke_width=2
                                               )
                shapes.append(line)

        omero.create_roi(connection=conn, image=image, shapes=shapes)

    conn.close()


def save_labels_image(labels_image, name, description, dataset_id, source_image_id):
    conn = omero.open_connection(username=USER,
                                 password=PASSWORD,
                                 group=GROUP,
                                 port=PORT,
                                 host=HOST)
    zct_list = list(product(range(labels_image.shape[0]),
                            range(labels_image.shape[1]),
                            range(labels_image.shape[2])))
    zct_generator = (labels_image[z, c, t, :, :] for z, c, t in zct_list)

    dataset = conn.getObject('Dataset', dataset_id)

    conn.createImageFromNumpySeq(zctPlanes=zct_generator,
                                 imageName=name,
                                 sizeZ=labels_image.shape[0],
                                 sizeC=labels_image.shape[1],
                                 sizeT=labels_image.shape[2],
                                 description=description,
                                 dataset=dataset,
                                 sourceImageId=source_image_id)
    conn.close()


def main(run_mode):
    logger.info('Metrics started')

    config = MetricsConfig()
    config.read(config_file)

    if run_mode == 'local':
        spots_image = get_local_data(spots_image_path)
        vertical_res_image = get_local_data(horizontal_stripes_image_path)
        horizontal_res_image = get_local_data(vertical_stripes_image_path)

    elif run_mode == 'omero':
        # spots_image = get_omero_data(spots_image_id)
        vertical_res_image = get_omero_data(horizontal_stripes_image_id)
        horizontal_res_image = get_omero_data(vertical_stripes_image_id)

    else:
        raise Exception('run mode not defined')

    if config.has_section('EXCITATION_POWER'):
        pm_conf = config['EXCITATION_POWER']
        if pm_conf['do_laser_power_measurement']:
            logger.info(f'Running laser power measurements')
            try:
                laser_lines = pm_conf.getlist('laser_power_measurement_wavelengths')
            except KeyError as e:
                laser_lines = config['WAVELENGTHS'].getlist('excitation')
            units = pm_conf['laser_power_measurement_units']
            create_laser_power_keys(laser_lines, units, dataset_id)

    if config.has_section('ARGOLIGHT'):
        logger.info(f'Running analysis on Argolight samples')
        al_conf = config['ARGOLIGHT']
        if al_conf.getboolean('do_spots'):
            logger.info(f'Analyzing spots image...')
            labels, names, desc, data, key_values = argolight.analyze_spots(image=spots_image['image_data'],
                                                                            pixel_size=spots_image['pixel_size'],
                                                                            pixel_size_units=spots_image['pixel_units'],
                                                                            low_corr_factors=al_conf.getlistfloat('low_threshold_correction_factors'),
                                                                            high_corr_factors=al_conf.getlistfloat('high_threshold_correction_factors'))
            labels = np.expand_dims(labels, 2)
            save_labels_image(labels_image=labels,
                              name=f'{spots_image["image_name"]}_rois',
                              description='Image with detected spots labels. Image intensities correspond to roi labels',  # TODO: add link reference to the raw image
                              dataset_id=dataset_id,
                              source_image_id=spots_image_id)

            save_data_table('AnalysisDate_argolight_D', names, desc, data, spots_image_id)

            save_data_key_values(key_values, spots_image_id)

            save_spots_point_rois(names, data, spots_image_id)  # nb_channels=len(al_conf.getlist('wavelengths'))

        if al_conf.getboolean('do_vertical_res'):
            logger.info(f'Analyzing vertical resolution...')
            profiles, key_values = argolight.analyze_resolution(image=vertical_res_image['image_data'],
                                                                pixel_size=vertical_res_image['pixel_size'],
                                                                pixel_units=vertical_res_image['pixel_units'],
                                                                axis=1,
                                                                measured_band=al_conf.getfloat('res_measurement_band')
                                                                )

            save_data_key_values(key_values=key_values,
                                 image_id=horizontal_stripes_image_id)

            save_line_rois(data=key_values,
                           image_id=horizontal_stripes_image_id)

            np.save('data/vertical_res_profile', profiles)

        if al_conf.getboolean('do_horizontal_res'):
            logger.info(f'Analyzing horizontal resolution...')
            profiles, key_values = argolight.analyze_resolution(image=horizontal_res_image['image_data'],
                                                                pixel_size=horizontal_res_image['pixel_size'],
                                                                pixel_units=horizontal_res_image['pixel_units'],
                                                                axis=2,
                                                                measured_band=al_conf.getfloat('res_measurement_band')
                                                                )

            save_data_key_values(key_values=key_values,
                                 image_id=vertical_stripes_image_id)

            save_line_rois(data=key_values,
                           image_id=vertical_stripes_image_id)

            np.save('data/horizontal_res_profile', profiles)

        logger.info('Metrics finished')


if __name__ == '__main__':
    main(RUN_MODE)

