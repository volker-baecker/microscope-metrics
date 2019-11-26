from omero.gateway import BlitzGateway, TagAnnotationWrapper, MapAnnotationWrapper
from omero.constants import metadata
from omero import grid
import numpy as np
from operator import mul
from itertools import product
from functools import reduce
from json import dumps
from random import choice
from string import ascii_letters

COLUMN_TYPES = {'string': grid.StringColumn,
                'long': grid.LongColumn,
                'bool': grid.BoolColumn,
                'double': grid.DoubleColumn,
                'plate': grid.PlateColumn,
                'well': grid.WellColumn,
                'roi': grid.RoiColumn,
                }

def open_connection(username, password, group, port, host):
    conn = BlitzGateway(username=username,
                        passwd=password,
                        group=group,
                        port=port,
                        host=host)
    try:
        conn.connect()
    except Exception as e:
        raise e
    return conn


def get_image(connection, image_id):
    try:
        image = connection.getObject('Image', image_id)
    except Exception as e:
        raise e
    return image


def get_dataset(connection, dataset_id):
    try:
        dataset = connection.getObject('Dataset', dataset_id)
    except Exception as e:
        raise e
    return dataset


def get_project(connection, project_id):
    try:
        project = connection.getObject('Project', project_id)
    except Exception as e:
        raise e
    return project


def get_image_shape(image):
    try:
        image_shape = (image.getSizeT(),
                       image.getSizeZ(),
                       image.getSizeC(),
                       image.getSizeX(),
                       image.getSizeY())
    except Exception as e:
        raise e

    return image_shape


def get_pixel_sizes(image):
    pixels = image.getPrimaryPixels()

    pixel_sizes = (pixels.getPhysicalSizeX().getValue(),
                   pixels.getPhysicalSizeY().getValue(),
                   pixels.getPhysicalSizeZ().getValue())
    return pixel_sizes


def get_pixel_units(image):
    pixels = image.getPrimaryPixels()

    pixel_size_units = (pixels.getPhysicalSizeX().getUnit().name,
                        pixels.getPhysicalSizeY().getUnit().name,
                        pixels.getPhysicalSizeZ().getUnit().name)
    return pixel_size_units


def get_5d_stack(image):
    # We will further work with stacks of the shape TZCXY
    image_shape = get_image_shape(image)

    nr_planes = reduce(mul, image_shape[:-2])

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


# In this section we give some convenience functions to send data back to OMERO #

def create_tag_annotation(conn, tag_string):
    tag_ann = TagAnnotationWrapper(conn)
    tag_ann.setValue(tag_string)
    tag_ann.save()

    return tag_ann


def _serialize_map_value(value):
    if isinstance(value, str):
        return value
    else:
        try:
            return dumps(value)
        except ValueError as e:
            # TODO: log an error
            return dumps(value.__str__())


def _dict_to_map(dictionary):
    """Converts a dictionary into a list of key:value pairs to be fed as map annotation.
    If value is not a string we serialize it as a json string"""
    map_annotation = [[k, _serialize_map_value(v)] for k, v in dictionary.items()]
    return map_annotation


def create_map_annotation(conn, annotation, client_editable=True):
    """Creates a map_annotation for OMERO. It can create a map annotation from a
    dictionary or from a list of 2 elements list.
    """
    # Convert a dictionary into a map annotation
    if isinstance(annotation, dict):
        annotation = _dict_to_map(annotation)
    elif isinstance(annotation, list):
        pass  # TODO: assert that the list is compliant with the OMERO format
    else:
        raise Exception(f'Could not convert {annotation} to a map_annotation')

    map_ann = MapAnnotationWrapper(conn)

    if client_editable:
        namespace = metadata.NSCLIENTMAPANNOTATION  # This makes the annotation editable in the client
        map_ann.setNs(namespace)

    map_ann.setValue(annotation)
    map_ann.save()

    return map_ann


def create_file_annotation(conn, file_path, namespace=None, description=None):
    """Creates a file annotation and uploads it to OMERO"""

    file_ann = conn.createFileAnnfromLocalFile(localPath=file_path,
                                               mimetype=None,
                                               namespace=namespace,
                                               desc=description)
    return file_ann


def _create_column(data_type, kwargs):
    column_class = COLUMN_TYPES[data_type]

    return column_class(**kwargs)


def _create_table(column_names, columns_descriptions, values):
    columns = list()
    for cn, cd, v in zip(column_names, columns_descriptions, values):
        if isinstance(v[0], str):
            size = len(max(v, key=len))
            args = {'name': cn, 'description': cd, 'size': size, 'values': v}
            columns.append(_create_column(data_type='string', kwargs=args))
        elif isinstance(v[0], int):
            args = {'name': cn, 'description': cd, 'values': v}
            columns.append(_create_column(data_type='long', kwargs=args))
        elif isinstance(v[0], float):
            columns.append(_create_column(data_type='double', name=cn, description=cd, values=v))
        elif isinstance(v[0], bool):
            columns.append(_create_column(data_type='string', name=cn, description=cd, values=v))
        else:
            raise Exception(f'Could not detect column datatype for {v[0]}')

    return columns


def create_table_annotation(conn, table_name, column_names, column_descriptions, values, namespace=None, description=None):
    """Creates a table annotation from a list of lists"""

    table_name = f'{table_name}_{"".join([choice(ascii_letters) for n in range(32)])}'

    columns = _create_table(column_names=column_names,
                          columns_descriptions=column_descriptions,
                          values=values)
    resources = conn.c.sf.sharedResources()
    repository_id = resources.repositories().descriptions[0].getId().getValue()
    table = resources.newTable(repository_id, table_name)
    table.initialize(columns)

    original_file = table.getOriginalFile()
    table.close()  # when we are done, close.
    return original_file


# orig_file_id = orig_file.id.val
# # ...so you can attach this data to an object e.g. Dataset
# file_ann = omero.model.FileAnnotationI()
# # use unloaded OriginalFileI
# file_ann.setFile(omero.model.OriginalFileI(orig_file_id, False))
# file_ann = conn.getUpdateService().saveAndReturnObject(file_ann)
# link = omero.model.DatasetAnnotationLinkI()
# link.setParent(omero.model.DatasetI(datasetId, False))
# link.setChild(omero.model.FileAnnotationI(file_ann.getId().getValue(), False))
# conn.getUpdateService().saveAndReturnObject(link)
#
