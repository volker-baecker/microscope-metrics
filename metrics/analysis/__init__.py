
import numpy as np
from itertools import product
from metrics.interface import omero

# import Argolight analyses tools
from metrics.samples.argolight import ArgolightSample
from metrics.samples.psf_beads import PSFBeadsSample

from datetime import datetime

# import logging
import logging

# Creating logging services
module_logger = logging.getLogger('metrics.analysis')

# Namespace constants
NAMESPACE_PREFIX = 'metrics'
NAMESPACE_ANALYZED = 'analyzed'
NAMESPACE_VALIDATED = 'validated'
# TODO: Add a special case editable


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
    image_id = image.getId()
    raw_img = omero.get_intensities(image)
    # Images from OMERO come in zctyx dimensions and locally as zcxy.
    # The easiest for the moment is to remove t from the omero image
    if raw_img.shape[2] == 1:
        raw_img = np.squeeze(raw_img, 2)  # TODO: Fix this time dimension.
    else:
        raise Exception("Image has a time dimension. Time is not yet implemented for this analysis")
    pixel_size = omero.get_pixel_size(image)
    pixel_size_units = omero.get_pixel_size_units(image)

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
            'image_id': image_id,
            'pixel_size': pixel_size,
            'pixel_size_units': pixel_size_units,
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


def create_roi(conn, shapes, image, name, description):
    new_shapes = list()
    type_to_func = {'point': omero.create_shape_point,
                    'line': omero.create_shape_line,
                    'rectangle': omero.create_shape_rectangle,
                    'ellipse': omero.create_shape_ellipse,
                    'polygon': omero.create_shape_polygon,
                    'mask': omero.create_shape_mask}

    for shape in shapes:
        new_shapes.append(type_to_func[shape['type']](**shape['args']))

    omero.create_roi(connection=conn,
                     image=image,
                     shapes=new_shapes,
                     name=name,
                     description=description)


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


def create_image(conn, image_intensities, image_name, description, dataset, source_image_id, metrics_tag_id=None):

    zct_list = list(product(range(image_intensities.shape[0]),
                            range(image_intensities.shape[1]),
                            range(image_intensities.shape[2])))
    zct_generator = (image_intensities[z, c, t, :, :] for z, c, t in zct_list)

    new_image = conn.createImageFromNumpySeq(zctPlanes=zct_generator,
                                             imageName=image_name,
                                             sizeZ=image_intensities.shape[0],
                                             sizeC=image_intensities.shape[1],
                                             sizeT=image_intensities.shape[2],
                                             description=description,
                                             dataset=dataset,
                                             sourceImageId=source_image_id)

    if metrics_tag_id is not None:
        tag = conn.getObject('Annotation', metrics_tag_id)
        if tag is None:
            module_logger.warning('Metrics tag is not found. New images will not be tagged. Verify metrics tag existence and id.')
        else:
            new_image.linkAnnotation(tag)

    return new_image


def create_laser_power_keys(conn, laser_lines, units, dataset, namespace):
    # TODO: Move this function somewhere else
    key_values = {str(k): '' for k in laser_lines}
    key_values['date'] = datetime.now().strftime("%Y-%m-%d")
    key_values['power_units'] = units

    map_ann = omero.create_annotation_map(connection=conn,
                                          annotation=key_values,
                                          annotation_description=namespace)
    omero.link_annotation(dataset, map_ann)


def analyze_dataset(connection, script_params, dataset, config):

    # TODO: must note in mapann the analyses that were done

    module_logger.info(f'Analyzing data from Dataset: {dataset.getId()}')
    module_logger.info(f'Date and time: {datetime.now()}')

    if config.has_section('EXCITATION_POWER'):
        ep_conf = config['EXCITATION_POWER']
        if ep_conf.getboolean('analyze_laser_power_measurement'):
            namespace = f'{NAMESPACE_PREFIX}/{NAMESPACE_ANALYZED}/excitation_power/laser_power_measurement/{config["MAIN"]["config_version"]}'
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
                                    namespace=namespace)

    if config.has_section('ARGOLIGHT'):
        module_logger.info(f'Running analysis on Argolight samples')
        al_conf = config['ARGOLIGHT']
        argolight = ArgolightSample(config=al_conf)
        if al_conf.getboolean('analyze_spots'):
            namespace = f'{NAMESPACE_PREFIX}/{NAMESPACE_ANALYZED}/argolight/spots/{config["MAIN"]["config_version"]}'
            spots_images = omero.get_tagged_images_in_dataset(dataset, al_conf.getint('spots_image_tag_id'))
            for image in spots_images:
                spots_image = get_omero_data(image=image)
                out_images, \
                    out_rois, \
                    out_tags, \
                    out_dicts, \
                    out_tables = argolight.analyze_spots(image=spots_image,
                                                         config=al_conf)

                for out_image in out_images:
                    create_image(conn=connection,
                                 image_intensities=out_image['image_data'],
                                 image_name=out_image['image_name'],
                                 description=f'Source Image Id:{image.getId()}\n{out_image["image_desc"]}',
                                 dataset=dataset,
                                 source_image_id=image.getId(),
                                 metrics_tag_id=config['MAIN'].getint('metrics_tag_id'))

                for out_roi in out_rois:
                    create_roi(conn=connection,
                               shapes=out_roi['shapes'],
                               image=image,
                               name=out_roi['name'],
                               description=out_roi['desc']
                               )

                for out_tag in out_tags:
                    pass  # TODO implement interface to save tags

                for out_table_name, out_table in out_tables.items():
                    save_data_table(conn=connection,
                                    table_name=out_table_name,
                                    col_names=[p['name'] for p in out_table],
                                    col_descriptions=[p['desc'] for p in out_table],
                                    col_data=[p['data'] for p in out_table],
                                    omero_obj=image,
                                    namespace=namespace)

                for out_dict in out_dicts:
                    save_data_key_values(conn=connection,
                                         key_values=out_dict,
                                         omero_obj=image,
                                         namespace=namespace)

        if al_conf.getboolean('analyze_vertical_resolution'):
            namespace = f'{NAMESPACE_PREFIX}/{NAMESPACE_ANALYZED}/argolight/vertical_resolution/{config["MAIN"]["config_version"]}'
            vertical_res_images = omero.get_tagged_images_in_dataset(dataset, al_conf.getint('vertical_resolution_image_tag_id'))
            for image in vertical_res_images:
                vertical_res_image = get_omero_data(image=image)
                module_logger.info(f'Analyzing vertical resolution...')
                out_images, \
                    out_rois, \
                    out_tags, \
                    out_dicts, \
                    out_tables = argolight.analyze_vertical_resolution(image=vertical_res_image,
                                                                       config=al_conf)

                for out_image in out_images:
                    create_image(conn=connection,
                                 image_intensities=out_image['image_data'],
                                 image_name=out_image['image_name'],
                                 description=f'Source Image Id:{image.getId()}\n{out_image["image_desc"]}',
                                 dataset=dataset,
                                 source_image_id=image.getId(),
                                 metrics_tag_id=config['MAIN'].getint('metrics_tag_id'))

                for out_roi in out_rois:
                    create_roi(conn=connection,
                               shapes=out_roi['shapes'],
                               image=image,
                               name=out_roi['name'],
                               description=out_roi['desc'])

                for out_tag in out_tags:
                    pass  # TODO implement interface to save tags

                for out_dict in out_dicts:
                    save_data_key_values(conn=connection,
                                         key_values=out_dict,
                                         omero_obj=image,
                                         namespace=namespace)

                for out_table_name, out_table in out_tables.items():
                    save_data_table(conn=connection,
                                    table_name=out_table_name,
                                    col_names=[p['name'] for p in out_table],
                                    col_descriptions=[p['desc'] for p in out_table],
                                    col_data=[p['data'] for p in out_table],
                                    omero_obj=image,
                                    namespace=namespace)

        if al_conf.getboolean('analyze_horizontal_resolution'):
            namespace = f'{NAMESPACE_PREFIX}/{NAMESPACE_ANALYZED}/argolight/horizontal_resolution/{config["MAIN"]["config_version"]}'
            horizontal_res_images = omero.get_tagged_images_in_dataset(dataset, al_conf.getint('horizontal_resolution_image_tag_id'))
            for image in horizontal_res_images:
                horizontal_res_image = get_omero_data(image=image)
                module_logger.info(f'Analyzing horizontal resolution...')
                out_images, \
                    out_rois, \
                    out_tags, \
                    out_dicts, \
                    out_tables = argolight.analyze_horizontal_resolution(image=horizontal_res_image,
                                                                         config=al_conf)

                for out_image in out_images:
                    create_image(conn=connection,
                                 image_intensities=out_image['image_data'],
                                 image_name=out_image['image_name'],
                                 description=f'Source Image Id:{image.getId()}\n{out_image["image_desc"]}',
                                 dataset=dataset,
                                 source_image_id=image.getId(),
                                 metrics_tag_id=config['MAIN'].getint('metrics_tag_id'))

                for out_roi in out_rois:
                    create_roi(conn=connection,
                               shapes=out_roi['shapes'],
                               image=image,
                               name=out_roi['name'],
                               description=out_roi['desc'])

                for out_tag in out_tags:
                    pass  # TODO implement interface to save tags

                for out_dict in out_dicts:
                    save_data_key_values(conn=connection,
                                         key_values=out_dict,
                                         omero_obj=image,
                                         namespace=namespace)

                for out_table_name, out_table in out_tables.items():
                    save_data_table(conn=connection,
                                    table_name=out_table_name,
                                    col_names=[p['name'] for p in out_table],
                                    col_descriptions=[p['desc'] for p in out_table],
                                    col_data=[p['data'] for p in out_table],
                                    omero_obj=image,
                                    namespace=namespace)

    if config.has_section('PSF_BEADS'):
        psf_conf = config['PSF_BEADS']
        psf_beads = PSFBeadsSample(config=psf_conf)
        if psf_conf.getboolean('analyze_beads'):
            module_logger.info(f'Running analyses on PSF beads samples')
            namespace = f'{NAMESPACE_PREFIX}/{NAMESPACE_ANALYZED}/psf_beads/beads/{config["MAIN"]["config_version"]}'
            psf_images = omero.get_tagged_images_in_dataset(dataset, psf_conf.getint('beads_image_tag_id'))
            for image in psf_images:
                psf_image = get_omero_data(image=image)
                module_logger.info(f'Analyzing PSF image: {image.getName()}')
                out_images, \
                    out_rois, \
                    out_tags, \
                    out_dicts, \
                    out_tables = psf_beads.analyze_beads(image=psf_image,
                                                         config=psf_conf)
                for out_image in out_images:
                    create_image(conn=connection,
                                 image_intensities=out_image['image_data'],
                                 image_name=out_image['image_name'],
                                 description=f'Source Image Id:{image.getId()}\n{out_image["image_desc"]}',
                                 dataset=dataset,
                                 source_image_id=image.getId(),
                                 metrics_tag_id=config['MAIN'].getint('metrics_tag_id'))

                for out_roi in out_rois:
                    create_roi(conn=connection,
                               shapes=out_roi['shapes'],
                               image=image,
                               name=out_roi['name'],
                               description=out_roi['desc'])

                for out_tag in out_tags:
                    pass  # TODO implement interface to save tags

                for out_dict in out_dicts:
                    save_data_key_values(conn=connection,
                                         key_values=out_dict,
                                         omero_obj=image,
                                         namespace=namespace)

                for out_table_name, out_table in out_tables.items():
                    save_data_table(conn=connection,
                                    table_name=out_table_name,
                                    col_names=[p['name'] for p in out_table],
                                    col_descriptions=[p['desc'] for p in out_table],
                                    col_data=[p['data'] for p in out_table],
                                    omero_obj=image,
                                    namespace=namespace)
                #
                #
                # for i, bead_image in enumerate(bead_images):
                #     new_image = create_image(conn=connection,
                #                              image_intensities=bead_image,
                #                              image_name=f'{psf_image["image_name"]}_bead-{i:02d}',
                #                              description=f'Image crop with detected bead. Source Image Id:{image.getId()}',
                #                              dataset=dataset,
                #                              source_image_id=image.getId(),
                #                              metrics_tag_id=config['MAIN'].getint('metrics_tag_id'))
                #     properties[[p['name'] for p in properties].index('bead_image')]['data'][i] = new_image
                #     new_shape = omero.create_shape_point(x_pos=properties[[p['name'] for p in properties].index('x_centroid')]['data'][i] + .5,
                #                                          y_pos=properties[[p['name'] for p in properties].index('y_centroid')]['data'][i] + .5,
                #                                          z_pos=properties[[p['name'] for p in properties].index('z_centroid')]['data'][i],
                #                                          c_pos=0,
                #                                          shape_name=f'{i:02d}',
                #                                          stroke_color=(0, 255, 0, 128),
                #                                          fill_color=(50, 255, 50, 20),
                #                                          stroke_width=2)
                #     new_roi = omero.create_roi(connection, image, [new_shape])
                #     # properties[[p['name'] for p in properties].index('bead_roi')]['data'][i] = new_roi
                #
                # # Saving properties table
                # if len(properties[0]['data']) > 0:
                #     save_data_table(conn=connection,
                #                     table_name='Analysis_PSF_properties',
                #                     col_names=[p['name'] for p in properties],
                #                     col_descriptions=[p['desc'] for p in properties],
                #                     col_data=[p['data'] for p in properties],
                #                     omero_obj=image,
                #                     namespace=namespace)
                #
                # # Saving profiles table
                # if len(profiles_x) > 0:
                #     save_data_table(conn=connection,
                #                     table_name='Analysis_PSF_X_profiles',
                #                     col_names=[p['name'] for p in profiles_x],
                #                     col_descriptions=[p['desc'] for p in profiles_x],
                #                     col_data=[p['data'] for p in profiles_x],
                #                     omero_obj=image,
                #                     namespace=namespace)
                #     save_data_table(conn=connection,
                #                     table_name='Analysis_PSF_Y_profiles',
                #                     col_names=[p['name'] for p in profiles_y],
                #                     col_descriptions=[p['desc'] for p in profiles_y],
                #                     col_data=[p['data'] for p in profiles_y],
                #                     omero_obj=image,
                #                     namespace=namespace)
                #     save_data_table(conn=connection,
                #                     table_name='Analysis_PSF_Z_profiles',
                #                     col_names=[p['name'] for p in profiles_z],
                #                     col_descriptions=[p['desc'] for p in profiles_z],
                #                     col_data=[p['data'] for p in profiles_z],
                #                     omero_obj=image,
                #                     namespace=namespace)
                #
                # # Saving key-values
                # save_data_key_values(conn=connection,
                #                      key_values=key_values,
                #                      omero_obj=image,
                #                      namespace=namespace)
                #
                # # Annotating edge discarded beads
                # discarded_shapes = list()
                # for pos in positions_edge_discarded:
                #     discarded_shapes.append(omero.create_shape_point(x_pos=pos[2].item() + .5,
                #                                                      y_pos=pos[1].item() + .5,
                #                                                      z_pos=pos[0].item() + .5,
                #                                                      c_pos=0,
                #                                                      shape_name='discarded: edge',
                #                                                      stroke_color=(255, 0, 0, 128),
                #                                                      fill_color=(255, 100, 100, 50),
                #                                                      stroke_width=2))
                # omero.create_roi(connection, image, discarded_shapes)
                #
                # # Annotating proximity discarded beads
                # discarded_shapes = list()
                # for pos in positions_proximity_discarded:
                #     discarded_shapes.append(omero.create_shape_point(x_pos=pos[2].item() + .5,
                #                                                      y_pos=pos[1].item() + .5,
                #                                                      z_pos=pos[0].item() + .5,
                #                                                      c_pos=0,
                #                                                      shape_name='discarded: proximity',
                #                                                      stroke_color=(255, 0, 0, 128),
                #                                                      fill_color=(255, 100, 100, 50),
                #                                                      stroke_width=2))
                # omero.create_roi(connection, image, discarded_shapes)
                #
                # # Annotating intensity discarded beads
                # discarded_shapes = list()
                # for pos in positions_intensity_discarded:
                #     discarded_shapes.append(omero.create_shape_point(x_pos=pos[2].item() + .5,
                #                                                      y_pos=pos[1].item() + .5,
                #                                                      z_pos=pos[0].item() + .5,
                #                                                      c_pos=0,
                #                                                      shape_name='discarded: intensity',
                #                                                      stroke_color=(255, 0, 0, 128),
                #                                                      fill_color=(255, 100, 100, 50),
                #                                                      stroke_width=2))
                # omero.create_roi(connection, image, discarded_shapes)

    if script_params['Comment'] != '':  # TODO: This is throuwing an error if no comment
        module_logger.info('Adding comment to Dataset.')
        comment_annotation = omero.create_annotation_comment(connection=connection,
                                                             comment_string=script_params['Comment'],
                                                             namespace=f'{NAMESPACE_PREFIX}/{NAMESPACE_ANALYZED}/comment/comment/{config["MAIN"]["config_version"]}')
        omero.link_annotation(dataset, comment_annotation)

    module_logger.info(f'Analysis finished for dataset: {dataset.getId()}')
