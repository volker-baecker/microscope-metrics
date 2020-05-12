
import numpy as np
from itertools import product
from metrics.interface import omero as interface
from datetime import datetime
import logging

# import inspect
# from os import path
# import importlib
# importlib.resources.contents('metrics.samples')
# importlib.import_module('.argolight', package='metrics.samples')

# import dataset analysis
from metrics.samples.dataset import DatasetConfigurator

# import samples
from metrics.samples.argolight import ArgolightConfigurator
from metrics.samples.psf_beads import PSFBeadsConfigurator

SAMPLE_CONFIGURATORS = [ArgolightConfigurator,
                        PSFBeadsConfigurator]
# noinspection PyUnresolvedReferences
SAMPLE_HANDLERS = [c.SAMPLE_CLASS for c in SAMPLE_CONFIGURATORS]
SAMPLE_SECTIONS = [c.CONFIG_SECTION for c in SAMPLE_CONFIGURATORS]
SAMPLE_ANALYSES = [c.ANALYSES for c in SAMPLE_CONFIGURATORS]

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


def get_image_data(image):
    image_name = image.getName()
    image_id = image.getId()
    raw_img = interface.get_intensities(image)
    # Images from Interface come in zctyx dimensions and locally as zcxy.
    # The easiest for the moment is to remove t
    if raw_img.shape[2] == 1:
        raw_img = np.squeeze(raw_img, 2)  # TODO: Fix this time dimension.
    else:
        raise Exception("Image has a time dimension. Time is not yet implemented for this analysis")
    pixel_size = interface.get_pixel_size(image)
    pixel_size_units = interface.get_pixel_size_units(image)

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


def save_data_table(conn, table_name, col_names, col_descriptions, col_data, interface_obj, namespace):

    table_ann = interface.create_annotation_table(connection=conn,
                                                  table_name=table_name,
                                                  column_names=col_names,
                                                  column_descriptions=col_descriptions,
                                                  values=col_data,
                                                  namespace=namespace)

    interface_obj.linkAnnotation(table_ann)


def save_data_key_values(conn, key_values, interface_obj, namespace, editable=False):
    map_ann = interface.create_annotation_map(connection=conn,
                                              annotation=key_values,
                                              namespace=namespace)
    interface_obj.linkAnnotation(map_ann)


def create_roi(conn, shapes, image, name, description):
    new_shapes = list()
    type_to_func = {'point': interface.create_shape_point,
                    'line': interface.create_shape_line,
                    'rectangle': interface.create_shape_rectangle,
                    'ellipse': interface.create_shape_ellipse,
                    'polygon': interface.create_shape_polygon,
                    'mask': interface.create_shape_mask}

    for shape in shapes:
        new_shapes.append(type_to_func[shape['type']](**shape['args']))

    interface.create_roi(connection=conn,
                         image=image,
                         shapes=new_shapes,
                         name=name,
                         description=description)


def create_image(conn, image_intensities, image_name, description, dataset, source_image_id=None, metrics_generated_tag_id=None):

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

    if metrics_generated_tag_id is not None:
        tag = conn.getObject('Annotation', metrics_generated_tag_id)
        if tag is None:
            module_logger.warning('Metrics tag is not found. New images will not be tagged. Verify metrics tag existence and id.')
        else:
            new_image.linkAnnotation(tag)

    return new_image


def analyze_dataset(connection, script_params, dataset, config):
    # TODO: must note in mapann the analyses that were done

    module_logger.info(f'Analyzing data from Dataset: {dataset.getId()}')
    module_logger.info(f'Date and time: {datetime.now()}')

    # This dictionary keeps track if the hard or soft limits are passed or not for the whole dataset
    dataset_limits_passed = {'uhl_passed': True,
                             'lhl_passed': True,
                             'usl_passed': True,
                             'lsl_passed': True,
                             'limits': list(),
                             'sources': list()}

    for section, analyses, handler in zip(SAMPLE_SECTIONS, SAMPLE_ANALYSES, SAMPLE_HANDLERS):
        if config.has_section(section):
            module_logger.info(f'Running analysis on {section.capitalize()} sample(s)')
            section_conf = config[section]
            handler_instance = handler(config=section_conf)
            for analysis in analyses:
                if section_conf.getboolean(f'analyze_{analysis}'):
                    namespace = (f'{NAMESPACE_PREFIX}/'
                                 f'{NAMESPACE_ANALYZED}/'
                                 f'{handler.get_module()}/'
                                 f'{analysis}/'
                                 f'{config["MAIN"]["config_version"]}')
                    images = interface.get_tagged_images_in_dataset(dataset, section_conf.getint(f'{analysis}_image_tag_id'))
                    for image in images:
                        image_data = get_image_data(image=image)
                        out_images,         \
                            out_rois,       \
                            out_tags,       \
                            out_dicts,      \
                            out_tables,     \
                            image_limits_passed = handler_instance.analyze_image(image=image_data,
                                                                                 analyses=analysis,
                                                                                 config=section_conf)

                        for out_image in out_images:
                            create_image(conn=connection,
                                         image_intensities=out_image['image_data'],
                                         image_name=out_image['image_name'],
                                         description=f'Source Image Id:{image.getId()}\n{out_image["image_desc"]}',
                                         dataset=dataset,
                                         source_image_id=image.getId(),
                                         metrics_generated_tag_id=config['MAIN'].getint('metrics_generated_tag_id'))  # TODO: Must go into metrics config

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
                                            interface_obj=image,
                                            namespace=namespace)

                        for out_dict in out_dicts:
                            save_data_key_values(conn=connection,
                                                 key_values=out_dict,
                                                 interface_obj=image,
                                                 namespace=namespace)

                        for ilp in image_limits_passed:
                            for k, v in ilp.items():
                                if v is False:
                                    dataset_limits_passed[k] = False
                                elif type(v) is list:
                                    dataset_limits_passed[k].extend(v)

    dataset_section = DatasetConfigurator.CONFIG_SECTION
    dataset_analyses = DatasetConfigurator.ANALYSES
    dataset_handler = DatasetConfigurator.SAMPLE_CLASS

    if config.has_section(dataset_section):
        module_logger.info(f'Running analysis on dataset')
        ds_conf = config[dataset_section]
        ds_handler_instance = dataset_handler(config=ds_conf)
        for dataset_analysis in dataset_analyses:
            if ds_conf.getboolean(f'analyze_{dataset_analysis}'):
                namespace = (f'{NAMESPACE_PREFIX}/'
                             f'{NAMESPACE_ANALYZED}/'
                             f'{dataset_handler.get_module()}/'
                             f'{dataset_analysis}/'
                             f'{config["MAIN"]["config_version"]}')
                out_images,         \
                    out_tags,       \
                    out_dicts,      \
                    out_editables,  \
                    out_tables,     \
                    image_limits_passed = ds_handler_instance.analyze_dataset(dataset=dataset,
                                                                              analyses=dataset_analysis,
                                                                              config=ds_conf)
                for out_image in out_images:
                    create_image(conn=connection,
                                 image_intensities=out_image['image_data'],
                                 image_name=out_image['image_name'],
                                 description=out_image["image_desc"],
                                 dataset=dataset,
                                 metrics_generated_tag_id=config['MAIN'].getint('metrics_generated_tag_id'))  # TODO: Must go into metrics config

                for out_tag in out_tags:
                    pass  # TODO implement interface to save tags

                for out_table_name, out_table in out_tables.items():
                    save_data_table(conn=connection,
                                    table_name=out_table_name,
                                    col_names=[p['name'] for p in out_table],
                                    col_descriptions=[p['desc'] for p in out_table],
                                    col_data=[p['data'] for p in out_table],
                                    interface_obj=dataset,
                                    namespace=namespace)

                for out_dict, out_editable in zip(out_dicts, out_editables):
                    if out_editable:
                        tmp_namespace = None
                    else:
                        tmp_namespace = namespace
                    save_data_key_values(conn=connection,
                                         key_values=out_dict,
                                         interface_obj=dataset,
                                         namespace=tmp_namespace)

                for ilp in image_limits_passed:
                    for k, v in ilp.items():
                        if v is False:
                            dataset_limits_passed[k] = False
                        elif type(v) is list:
                            dataset_limits_passed[k].extend(v)

    # Save final dataset limits passed tests and corresponding tags
    namespace = (f'{NAMESPACE_PREFIX}/'
                 f'{NAMESPACE_ANALYZED}/'
                 'dataset/'
                 'limits_verification/'
                 f'{config["MAIN"]["config_version"]}')

    # dataset_limits_passed['limits'] = list(dict.fromkeys(dataset_limits_passed['limits']))  # Remove duplicates

    if dataset_limits_passed['uhl_passed'] and dataset_limits_passed['lhl_passed']:  # Hard limits passed
        interface.link_annotation_tag(connection, dataset, config['MAIN'].getint('passed_hard_limits_tag_id'))
    else:
        interface.link_annotation_tag(connection, dataset, config['MAIN'].getint('not_passed_hard_limits_tag_id'))

    if dataset_limits_passed['usl_passed'] and dataset_limits_passed['lsl_passed']:  # Soft limits passed
        interface.link_annotation_tag(connection, dataset, config['MAIN'].getint('passed_soft_limits_tag_id'))
    else:
        interface.link_annotation_tag(connection, dataset, config['MAIN'].getint('not_passed_soft_limits_tag_id'))

    save_data_key_values(conn=connection,
                         key_values=dataset_limits_passed,
                         interface_obj=dataset,
                         namespace=namespace)

    try:
        if script_params['Comment'] != '':  # TODO: This is throwing an error if no comment
            module_logger.info('Adding comment to Dataset.')
            namespace = (f'{NAMESPACE_PREFIX}/'
                         f'{NAMESPACE_ANALYZED}/'
                         'comment/'
                         'comment/'
                         f'{config["MAIN"]["config_version"]}')

            comment_annotation = interface.create_annotation_comment(connection=connection,
                                                                     comment_string=script_params['Comment'],
                                                                     namespace=namespace)
            interface.link_annotation(dataset, comment_annotation)
    except KeyError:
        module_logger.info('No comments added')

    module_logger.info(f'Analysis finished for dataset: {dataset.getId()}')
