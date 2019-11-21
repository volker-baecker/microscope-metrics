from omero.gateway import BlitzGateway
import numpy as np
from operator import mul
from itertools import product


class Connection(BlitzGateway):
    def __init__(self, args):
        super(self).__init__()


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
