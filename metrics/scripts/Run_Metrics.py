#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
  Copyright (C) 2020 CNRS. All rights reserved.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along
  with this program; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

------------------------------------------------------------------------------

This script runs OMERO metrics on the selected dataset.

@author Julio Mateos Langerak
<a href="mailto:julio.mateos-langerak@igh.cnrs.fr">julio.mateos-langerak@igh.cnrs.fr</a>
@version Alpha0.1
<small>
(<b>Internal version:</b> $Revision: $Date: $)
</small>
@since 3.0-Beta4.3
"""

# import omero dependencies
import omero.scripts as scripts
import omero.gateway as gateway
from omero.rtypes import rlong, rstring, robject

# import configuration parser
from metrics.utils.utils import MetricsConfig

# import logging
import logging

# import Argolight analysis tools
from metrics.samples import argolight

# import other sample types analysis tools

import numpy as np
from itertools import product
from metrics.interface import omero

config_file = 'my_microscope_config.ini'

# Creating logging services
logger = logging.getLogger('metrics')
logger.setLevel(logging.DEBUG)

# # create file handler which logs even debug messages
# fh = logging.FileHandler('metrics.log')
# fh.setLevel(logging.ERROR)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
# logger.addHandler(fh)
# logger.addHandler(ch)


def get_omero_data(image):
    image_name = image.getName()
    raw_img = omero.get_intensities(image)
    # Images from OMERO come in zctxy dimensions and locally as zcxy.
    # The easiest for the moment is to remove t from the omero image
    if raw_img.shape[2] == 1:
        raw_img = np.squeeze(raw_img, 2)
    else:
        raise Exception("Image has a time dimension. Time is not yet implemented for this analysis")
    pixel_size = omero.get_pixel_size(image)
    pixel_units = omero.get_pixel_units(image)

    return {'image_data': raw_img, 'image_name': image_name, 'pixel_size': pixel_size, 'pixel_units': pixel_units}


def save_data_table(conn, table_name, col_names, col_descriptions, col_data, omero_obj):

    table_ann = omero.create_annotation_table(connection=conn,
                                              table_name=table_name,
                                              column_names=col_names,
                                              column_descriptions=col_descriptions,
                                              values=col_data)

    omero_obj.linkAnnotation(table_ann)
    # omero.link_annotation(omero_obj, table_ann)


def save_data_key_values(conn, key_values, omero_obj):

    map_ann = omero.create_annotation_map(connection=conn,
                                          annotation=key_values,
                                          client_editable=True)
    omero_obj.linkAnnotation(map_ann)
    # omero.link_annotation(omero_obj, map_ann)


def create_laser_power_keys(conn, laser_lines, units, dataset):
    # TODO: remove this function
    key_values = {str(k): '' for k in laser_lines}
    key_values['power_units'] = units

    map_ann = omero.create_annotation_map(connection=conn,
                                          annotation=key_values,
                                          client_editable=True)
    omero.link_annotation(dataset, map_ann)


def save_spots_point_rois(conn, names, data, image):
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


def save_line_rois(conn, data, image):
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


def save_labels_image(conn, labels_image, image_name, description, dataset, source_image_id, metrics_tag_id=None):

    zct_list = list(product(range(labels_image.shape[0]),
                            range(labels_image.shape[1]),
                            range(labels_image.shape[2])))
    zct_generator = (labels_image[z, c, t, :, :] for z, c, t in zct_list)

    new_image = conn.createImageFromNumpySeq(zctPlanes=zct_generator,
                                             imageName=image_name,
                                             sizeZ=labels_image.shape[0],
                                             sizeC=labels_image.shape[1],
                                             sizeT=labels_image.shape[2],
                                             description=description,
                                             dataset=dataset,
                                             sourceImageId=source_image_id)

    if metrics_tag_id is not None:
        tag = conn.getObject('Annotation', metrics_tag_id)
        if tag is None:
            logger.warning('Metrics tag is not found. New images will not be tagged. Verify metrics tag existence and id.')
        else:
            new_image.linkAnnotation(tag)


def get_tagged_images_in_dataset(dataset, tag_name):
    images = list()
    for image in dataset.listChildren():
        for ann in image.listAnnotations():
            if type(ann) == gateway.TagAnnotationWrapper:
                if ann.getValue() == tag_name:
                    images.append(image)
    return images


def analyze_dataset(connection, script_params, dataset, config=None):

    if config is None:  # TODO: We might remove this line in the final script. Config is provided for debugging
        # Get the project / microscope
        config = MetricsConfig()
        microscope_prj = dataset.getParent()  # We assume one project per dataset

        for ann in microscope_prj.listAnnotations():
            if type(ann) == gateway.FileAnnotationWrapper:
                if ann.getFileName() == script_params['Configuration file name']:
                    config.read_string(ann.getFileInChunks().__next__().decode())  # TODO: Fix this for large config files
                    break

    if config.has_section('EXCITATION_POWER'):
        ep_conf = config['EXCITATION_POWER']
        if ep_conf['do_laser_power_measurement']:
            logger.info(f'Running laser power measurements')
            try:
                laser_lines = ep_conf.getlist('laser_power_measurement_wavelengths')
            except KeyError:
                laser_lines = config['WAVELENGTHS'].getlist('excitation')
            units = ep_conf['laser_power_measurement_units']
            create_laser_power_keys(connection, laser_lines, units, dataset)

    if config.has_section('ARGOLIGHT'):
        logger.info(f'Running analysis on Argolight samples')
        al_conf = config['ARGOLIGHT']
        if al_conf.getboolean('do_spots'):
            spots_images = get_tagged_images_in_dataset(dataset, al_conf['spots_image_tag'])
            for image in spots_images:
                spots_image = get_omero_data(image=image)
                logger.info(f'Analyzing spots image...')
                labels, \
                    names, \
                    desc, \
                    data, \
                    key_values = argolight.analyze_spots(image=spots_image['image_data'],
                                                         pixel_size=spots_image['pixel_size'],
                                                         pixel_size_units=spots_image['pixel_units'],
                                                         low_corr_factors=al_conf.getlistfloat('low_threshold_correction_factors'),
                                                         high_corr_factors=al_conf.getlistfloat('high_threshold_correction_factors'))
                labels = np.expand_dims(labels, 2)
                save_labels_image(conn=connection,
                                  labels_image=labels,
                                  image_name=f'{spots_image["image_name"]}_rois',
                                  description='Image with detected spots labels. Image intensities correspond to roi labels',
                                  # TODO: add link reference to the raw image
                                  dataset=dataset,
                                  source_image_id=image.getId(),
                                  metrics_tag_id=config['MAIN'].getint('metrics_tag_id'))

                save_data_table(conn=connection,
                                table_name='AnalysisDate_argolight_D',
                                col_names=names,
                                col_descriptions=desc,
                                col_data=data,
                                omero_obj=image)

                save_data_key_values(conn=connection,
                                     key_values=key_values,
                                     omero_obj=image)

                save_spots_point_rois(conn=connection,
                                      names=names,
                                      data=data,
                                      image=image)  # nb_channels=len(al_conf.getlist('wavelengths'))

        if al_conf.getboolean('do_vertical_res'):
            vertical_res_images = get_tagged_images_in_dataset(dataset, al_conf['vertical_res_image_tag'])
            for image in vertical_res_images:
                vertical_res_image = get_omero_data(image=image)
                logger.info(f'Analyzing vertical resolution...')
                profiles, key_values = argolight.analyze_resolution(image=vertical_res_image['image_data'],
                                                                    pixel_size=vertical_res_image['pixel_size'],
                                                                    pixel_units=vertical_res_image['pixel_units'],
                                                                    axis=1,
                                                                    measured_band=al_conf.getfloat('res_measurement_band'),
                                                                    precision=al_conf.getint('res_precision'))

                save_data_key_values(conn=connection,
                                     key_values=key_values,
                                     omero_obj=image)

                save_line_rois(conn=connection,
                               data=key_values,
                               image=image)

        if al_conf.getboolean('do_horizontal_res'):
            horizontal_res_images = get_tagged_images_in_dataset(dataset, al_conf['horizontal_res_image_tag'])
            for image in horizontal_res_images:
                horizontal_res_image = get_omero_data(image=image)
                logger.info(f'Analyzing horizontal resolution...')
                profiles, key_values = argolight.analyze_resolution(image=horizontal_res_image['image_data'],
                                                                    pixel_size=horizontal_res_image['pixel_size'],
                                                                    pixel_units=horizontal_res_image['pixel_units'],
                                                                    axis=2,
                                                                    measured_band=al_conf.getfloat('res_measurement_band'),
                                                                    precision=al_conf.getint('res_precision'))

                save_data_key_values(conn=connection,
                                     key_values=key_values,
                                     omero_obj=image)

                save_line_rois(conn=connection,
                               data=key_values,
                               image=image)

        logger.info('Metrics finished')


def run_script_local():
    from credentials import USER, PASSWORD, GROUP, PORT, HOST
    conn = gateway.BlitzGateway(username=USER,
                                passwd=PASSWORD,
                                group=GROUP,
                                port=PORT,
                                host=HOST)

    script_params = {'Dataset ID': 1,
                     'Configuration file name': 'monthly_config.ini',
                     'Comment': 'This is a test comment'}

    try:
        conn.connect()

        logger.info(f'Metrics started using parameters: \n{script_params}')

        logger.info(f'Connection successful: {conn.isConnected()}')

        dataset = conn.getObject('Dataset', script_params['Dataset ID'])

        # Getting the configuration file associated with the microscope
        config = MetricsConfig()
        config.read(script_params['Configuration file name'])

        analyze_dataset(connection=conn,
                        script_params=script_params,
                        dataset=dataset,
                        config=config)

    finally:
        logger.info('Closing connection')
        conn.close()


def run_script():

    client = scripts.client(
        'Run_Metrics.py',
        """This is the main script of omero.metrics. It will run the analysis on the selected 
        dataset. For more information check \n
        http://www.mri.cnrs.fr\n
        Copyright: Write here some copyright info""",  # TODO: copyright info

        scripts.String(
            "Data_Type", optional=False, grouping="1",
            description="The data you want to work with.", values=[rstring('Dataset')],
            default="Dataset"),

        scripts.List(
            "IDs", optional=False, grouping="1",
            description="List of Dataset IDs").ofType(rlong(0)),

        scripts.String(
            'Configuration file name', optional=False, grouping='1', default='monthly_config.ini',
            description='Add here any eventuality that you want to add to the analysis'
        ),

        scripts.String(
            'Comment', optional=True, grouping='2',
            description='Add here any eventuality that you want to add to the analysis'
        ),
    )

    try:
        script_params = {}
        for key in client.getInputKeys():
            if client.getInput(key):
                script_params[key] = client.getInput(key, unwrap=True)

        logger.info(f'Metrics started using parameters: \n{script_params}')

        conn = gateway.BlitzGateway(client_obj=client)

        # Verify user is part of metrics group by checking current group. If not, abort the script
        if conn.getGroupFromContext().getName() != 'metrics':
            raise PermissionError('You are not authorized to run this script in the current context.')

        logger.info(f'Connection success: {conn.isConnected()}')

        datasets = conn.getObjects('Dataset', script_params['IDs'])  # generator of datasets

        for dataset in datasets:
            logger.info(f'analyzing data from Dataset: {dataset.getId()}')
            analyze_dataset(connection=conn,
                            script_params=script_params,
                            dataset=dataset)

    finally:
        client.closeSession()


if __name__ == '__main__':
    # run_script()
    run_script_local()

