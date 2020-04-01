
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

import os
print(os.environ['OMERODIR'])

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

    return {'image_data': raw_img, 'pixel_size': pixel_size, 'pixel_units': pixel_units}


def save_spots_data_table(table_name, names, desc, data):

    conn = omero.open_connection(username=USER,
                                 password=PASSWORD,
                                 group=GROUP,
                                 port=PORT,
                                 host=HOST)

    try:
        table = omero.create_annotation_table(connection=conn,
                                              table_name=table_name,
                                              column_names=names,
                                              column_descriptions=desc,
                                              values=data)

        image = omero.get_image(conn, spots_image_id)

        omero.link_annotation(image, table)
    finally:
        conn.close()
    # We want to save:
    # A table per image containing the following columns:
    # - source Image
    # - Per channel
    #   + RoiColumn(name='chXX_MaxIntegratedIntensityRoi', description='ROI with the highest integrated intensity.', values)
    #   + RoiColumn(name='chXX_MinIntegratedIntensityRoi', description='ROI with the lowest integrated intensity.', values)
    #   - LongArrayColumn(name='chXX_roiCentroidLabels', description='Labels of the centroids ROIs.', size=(verify size), values)
    #   + LongArrayColumn(name='chXX_roiMaskLabels', description='Labels of the mask ROIs.', size=(verify size), values)
    #   + FloatArrayColumn(name='chXX_roiVolume', description='Volume of the ROIs.', size=(verify size), values)
    # + StringColumn(name='roiVolumeUnit', description='Volume units for the ROIs.', size=(max size), values)
    #   + FloatArrayColumn(name='chXX_roiMinIntensity', description='Minimum intensity of the ROIs.', size=(verify size), values)
    #   + FloatArrayColumn(name='chXX_roiMaxIntensity', description='Maximum intensity of the ROIs.', size=(verify size), values)
    #   + FloatArrayColumn(name='chXX_roiMeanIntensity', description='Mean intensity of the ROIs.', size=(verify size), values)
    #   + FloatArrayColumn(name='chXX_roiIntegratedIntensity', description='Integrated intensity of the ROIs.', size=(verify size), values)
    #   + FloatArrayColumn(name='chXX_roiXWeightedCentroid', description='Wighted Centroid X coordinates of the ROIs.', size=(verify size), values)
    #   + FloatArrayColumn(name='chXX_roiYWeightedCentroid', description='Wighted Centroid Y coordinates of the ROIs.', size=(verify size), values)
    #   + FloatArrayColumn(name='chXX_roiZWeightedCentroid', description='Wighted Centroid Z coordinates of the ROIs.', size=(verify size), values)
    # + StringColumn(name='roiWeightedCentroidUnits', description='Wighted Centroid coordinates units for the ROIs.', size=(max size), values)
    # - Per channel permutation
    #   + FloatArrayColumn(name='chXX_chYY_chARoiLabels', description='Labels of the ROIs in channel A.', size=(verify size), values)
    #   + FloatArrayColumn(name='chXX_chYY_chBRoiLabels', description='Labels of the ROIs in channel B.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_chYY_XDistance', description='Distance in X between Weighted Centroids of mutually closest neighbouring ROIs in channels A and B.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_chYY_YDistance', description='Distance in Y between Weighted Centroids of mutually closest neighbouring ROIs in channels A and B.', size=(verify size), values)
    #   - FloatArrayColumn(name='chXX_chYY_ZDistance', description='Distance in Z between Weighted Centroids of mutually closest neighbouring ROIs in channels A and B.', size=(verify size), values)
    #   + FloatArrayColumn(name='chXX_chYY_3dDistance', description='Distance in 3D between Weighted Centroids of mutually closest neighbouring ROIs in channels A and B.', size=(verify size), values)
    # + StringColumn(name='DistanceUnits', description='Wighted Centroid coordinates units for the ROIs.', size=(max size), values)


def main(run_mode):
    logger.info('Metrics started')

    config = MetricsConfig()
    config.read(config_file)

    if run_mode == 'local':
        spots_image = get_local_data(spots_image_path)
        vertical_res_image = get_local_data(horizontal_stripes_image_path)
        horizontal_res_image = get_local_data(vertical_stripes_image_path)

    elif run_mode == 'omero':
        spots_image = get_omero_data(spots_image_id)
        # vertical_res_image = get_omero_data(horizontal_stripes_image_id)
        # horizontal_res_image = get_omero_data(vertical_stripes_image_id)

    else:
        raise Exception('run mode not defined')

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

            save_spots_data_table('AnalysisDate_argolight_D', names, desc, data)

        if al_conf.getboolean('do_vertical_res'):
            logger.info(f'Analyzing vertical resolution...')
            argolight.analyze_resolution(vertical_res_image['image_data'],
                                         vertical_res_image['pixel_size'],
                                         vertical_res_image['pixel_units'],
                                         1)

        if al_conf.getboolean('do_horizontal_res'):
            logger.info(f'Analyzing horizontal resolution...')
            argolight.analyze_resolution(horizontal_res_image['image_data'],
                                         horizontal_res_image['pixel_size'],
                                         horizontal_res_image['pixel_units'],
                                         0)

        logger.info('Metrics finished')


if __name__ == '__main__':
    main(RUN_MODE)

