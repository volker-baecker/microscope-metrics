
import numpy as np
from itertools import product
from metrics.interface import omero

# import Argolight analysis tools
from metrics.samples import psf_beads, argolight

from datetime import datetime
from json import dumps

# import logging
import logging

# Creating logging services
module_logger = logging.getLogger('metrics.analysis')


def _get_metadata_from_name(name, token_left, token_right, metadata_type=None):
    start = name.find(token_left)
    if start == -1:
        return None
    name = name[start + len(token_left):]
    end = name.find(token_right)
    name = metadata_type(name[:end])
    if metadata_type is None:
        return name
    else:
        return metadata_type(name)


def get_omero_data(image):
    image_name = image.getName()
    raw_img = omero.get_intensities(image)
    # Images from OMERO come in zctyx dimensions and locally as zcxy.
    # The easiest for the moment is to remove t from the omero image
    if raw_img.shape[2] == 1:
        raw_img = np.squeeze(raw_img, 2)  # TODO: Fix this time dimension.
    else:
        raise Exception("Image has a time dimension. Time is not yet implemented for this analysis")
    pixel_size = omero.get_pixel_size(image)
    pixel_units = omero.get_pixel_units(image)

    try:
        objective_settings = image.getObjectiveSettings()
        refractive_index = objective_settings.getRefractiveIndex()
        objective = objective_settings.getObjective()
        lens_na = objective.getLensNA()
        lens_magnification = objective.getNominalMagnification()
    except AttributeError:
        module_logger.warning(f'Image {image.getName()} does not have a declared objective settings.'
                              f'Falling back to metadata stored in image name.')
        refractive_index = _get_metadata_from_name(image_name, '_ri-', '_', float)
        lens_na = _get_metadata_from_name(image_name, '_na-', '_', float)
        lens_magnification = _get_metadata_from_name(image_name, '_mag-', '_', float)

    channels = image.getChannels()
    excitation_waves = [ch.getExcitationWave() for ch in channels]
    emission_waves = [ch.getEmissionWave() for ch in channels]
    if excitation_waves[0] is None:
        module_logger.warning(f'Image {image.getName()} does not have a declared channels settings.'
                              f'Falling back to metadata stored in image name.')
        excitation_waves = [_get_metadata_from_name(image_name, '_ex-', '_', float)]  # TODO: make this work with more than one channel
        emission_waves = [_get_metadata_from_name(image_name, '_em-', '_', float)]

    return {'image_data': raw_img,
            'image_name': image_name,
            'pixel_size': pixel_size,
            'pixel_units': pixel_units,
            'refractive_index': refractive_index,
            'lens_na': lens_na,
            'lens_magnification': lens_magnification,
            'excitation_waves': excitation_waves,
            'emission_waves': emission_waves,
            }


def save_data_table(conn, table_name, col_names, col_descriptions, col_data, omero_obj, namespace):

    table_ann = omero.create_annotation_table(connection=conn,
                                              table_name=table_name,
                                              column_names=col_names,
                                              column_descriptions=col_descriptions,
                                              values=col_data,
                                              namespace=namespace)

    omero_obj.linkAnnotation(table_ann)
    # omero.link_annotation(omero_obj, table_ann)


def save_data_key_values(conn, key_values, omero_obj, namespace):

    map_ann = omero.create_annotation_map(connection=conn,
                                          annotation=key_values,
                                          namespace=namespace)
    omero_obj.linkAnnotation(map_ann)


def save_spots_point_rois(conn, names, data, image):
    nb_channels = image.getSizeC()
    channels_shapes = [[] for x in range(nb_channels)]

    for x, y, z, c, l in zip(data[names.index('x_weighted_centroid')],
                             data[names.index('y_weighted_centroid')],
                             data[names.index('z_weighted_centroid')],
                             data[names.index('channel')],
                             data[names.index('mask_labels')],
                             ):
        channels_shapes[c].append(omero.create_shape_point(x, y, z, c_pos=c, shape_name=f'{l}'))

    for shapes in channels_shapes:
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
                                               c_pos=c,
                                               line_name=f'ch{c:02d}_{l}_{p}',
                                               # TODO: add color to line: stroke_color=,
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
            module_logger.warning('Metrics tag is not found. New images will not be tagged. Verify metrics tag existence and id.')
        else:
            new_image.linkAnnotation(tag)


def create_laser_power_keys(conn, laser_lines, units, dataset, namespace):
    # TODO: Move this function somewhere else
    key_values = {str(k): '' for k in laser_lines}
    key_values['date'] = datetime.now().strftime("%Y-%m-%d")
    key_values['power_units'] = units

    map_ann = omero.create_annotation_map(connection=conn,
                                          annotation=key_values,
                                          namespace=namespace)
    omero.link_annotation(dataset, map_ann)


def analyze_dataset(connection, script_params, dataset, config):

    module_logger.info(f'Analyzing data from Dataset: {dataset.getId()}')
    module_logger.info(f'Date and time: {datetime.now()}')

    if config.has_section('EXCITATION_POWER'):
        ep_conf = config['EXCITATION_POWER']
        if ep_conf['do_laser_power_measurement']:
            namespace = f'metrics/excitation_power/{config["MAIN"]["config_version"]}'
            module_logger.info(f'Running laser power measurements')
            try:
                laser_lines = ep_conf.getlist('laser_power_measurement_wavelengths')
            except KeyError:
                laser_lines = config['WAVELENGTHS'].getlist('excitation')
            units = ep_conf['laser_power_measurement_units']
            create_laser_power_keys(conn=connection,
                                    laser_lines=laser_lines,
                                    units=units,
                                    dataset=dataset,
                                    namespace=None)

    if config.has_section('ARGOLIGHT'):
        module_logger.info(f'Running analysis on Argolight samples')
        al_conf = config['ARGOLIGHT']
        if al_conf.getboolean('do_spots'):
            namespace = f'metrics/argolight/spots/{config["MAIN"]["config_version"]}'
            spots_images = omero.get_tagged_images_in_dataset(dataset, al_conf['spots_image_tag'])
            for image in spots_images:
                spots_image = get_omero_data(image=image)
                module_logger.info(f'Analyzing spots image...')
                labels, \
                    properties, \
                    distances,  \
                    key_values = argolight.analyze_spots(image=spots_image['image_data'],
                                                         pixel_size=spots_image['pixel_size'],
                                                         pixel_size_units=spots_image['pixel_units'],
                                                         low_corr_factors=al_conf.getlistfloat('low_threshold_correction_factors'),
                                                         high_corr_factors=al_conf.getlistfloat('high_threshold_correction_factors'))
                labels = np.expand_dims(labels, 2)
                save_labels_image(conn=connection,
                                  labels_image=labels,
                                  image_name=f'{spots_image["image_name"]}_rois',
                                  description=f'Image with detected spots labels. Image intensities correspond to roi labels.\nSource Image Id:{image.getId()}',
                                  dataset=dataset,
                                  source_image_id=image.getId(),
                                  metrics_tag_id=config['MAIN'].getint('metrics_tag_id'))

                save_data_table(conn=connection,
                                table_name='AnalysisDate_argolight_D_properties',
                                col_names=[p['name'] for p in properties],
                                col_descriptions=[p['desc'] for p in properties],
                                col_data=[p['data'] for p in properties],
                                omero_obj=image,
                                namespace=namespace)

                save_data_table(conn=connection,
                                table_name='AnalysisDate_argolight_D_distances',
                                col_names=[p['name'] for p in distances],
                                col_descriptions=[p['desc'] for p in distances],
                                col_data=[p['data'] for p in distances],
                                omero_obj=image,
                                namespace=namespace)

                save_data_key_values(conn=connection,
                                     key_values=key_values,
                                     omero_obj=image,
                                     namespace=namespace)

                save_spots_point_rois(conn=connection,
                                      names=[p['name'] for p in properties],
                                      data=[p['data'] for p in properties],
                                      image=image)

        if al_conf.getboolean('do_vertical_res'):
            namespace = f'metrics/argolight/vertical_res/{config["MAIN"]["config_version"]}'
            vertical_res_images = omero.get_tagged_images_in_dataset(dataset, al_conf['vertical_res_image_tag'])
            for image in vertical_res_images:
                vertical_res_image = get_omero_data(image=image)
                module_logger.info(f'Analyzing vertical resolution...')
                profiles, key_values = argolight.analyze_resolution(image=vertical_res_image['image_data'],
                                                                    pixel_size=vertical_res_image['pixel_size'],
                                                                    pixel_units=vertical_res_image['pixel_units'],
                                                                    axis=1,
                                                                    measured_band=al_conf.getfloat('res_measurement_band'),
                                                                    precision=al_conf.getint('res_precision'))

                save_data_key_values(conn=connection,
                                     key_values=key_values,
                                     omero_obj=image,
                                     namespace=namespace)

                save_line_rois(conn=connection,
                               data=key_values,
                               image=image)

        if al_conf.getboolean('do_horizontal_res'):
            namespace = f'metrics/argolight/horizontal_res/{config["MAIN"]["config_version"]}'
            horizontal_res_images = omero.get_tagged_images_in_dataset(dataset, al_conf['horizontal_res_image_tag'])
            for image in horizontal_res_images:
                horizontal_res_image = get_omero_data(image=image)
                module_logger.info(f'Analyzing horizontal resolution...')
                profiles, key_values = argolight.analyze_resolution(image=horizontal_res_image['image_data'],
                                                                    pixel_size=horizontal_res_image['pixel_size'],
                                                                    pixel_units=horizontal_res_image['pixel_units'],
                                                                    axis=2,
                                                                    measured_band=al_conf.getfloat('res_measurement_band'),
                                                                    precision=al_conf.getint('res_precision'))

                save_data_key_values(conn=connection,
                                     key_values=key_values,
                                     omero_obj=image,
                                     namespace=namespace)

                save_line_rois(conn=connection,
                               data=key_values,
                               image=image)

    if config.has_section('PSF_BEADS'):
        module_logger.info(f'Running analysis on PSF beads samples')
        psf_conf = config['PSF_BEADS']
        if psf_conf.getboolean('do_beads'):
            namespace = f'metrics/psf_beads/spots/{config["MAIN"]["config_version"]}'
            psf_images = omero.get_tagged_images_in_dataset(dataset, psf_conf['psf_image_tag'])
            for image in psf_images:
                psf_image = get_omero_data(image=image)
                module_logger.info(f'Analyzing PSF image: {image.getName()}')
                bead_images, properties, key_values = psf_beads.analyze_image(image_data=psf_image,
                                                                              config=psf_conf)

                for bead_image in bead_images:
                    save_labels_image(conn=connection,
                                      labels_image=labels,
                                      image_name=f'{spots_image["image_name"]}_rois',
                                      description=f'Image with detected spots labels. Image intensities correspond to roi labels.\nSource Image Id:{image.getId()}',
                                      dataset=dataset,
                                      source_image_id=image.getId(),
                                      metrics_tag_id=config['MAIN'].getint('metrics_tag_id'))

                save_data_table(conn=connection,
                                table_name='AnalysisDate_argolight_D_properties',
                                col_names=[p['name'] for p in properties],
                                col_descriptions=[p['desc'] for p in properties],
                                col_data=[p['data'] for p in properties],
                                omero_obj=image,
                                namespace=namespace)

                save_data_table(conn=connection,
                                table_name='AnalysisDate_argolight_D_distances',
                                col_names=[p['name'] for p in distances],
                                col_descriptions=[p['desc'] for p in distances],
                                col_data=[p['data'] for p in distances],
                                omero_obj=image,
                                namespace=namespace)

                save_data_key_values(conn=connection,
                                     key_values=key_values,
                                     omero_obj=image,
                                     namespace=namespace)

                save_spots_point_rois(conn=connection,
                                      names=[p['name'] for p in properties],
                                      data=[p['data'] for p in properties],
                                      image=image)



    module_logger.info(f'Analysis finished for dataset: {dataset.getId()}')
