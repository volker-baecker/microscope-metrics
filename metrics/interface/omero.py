import numpy as np
from omero.gateway import BlitzGateway





def get_5d_stack(image):
    # We will further work with stacks of the shape TZCXY
    image_shape = (image.getSizeT(),
                   image.getSizeZ(),
                   image.getSizeC(),
                   image.getSizeX(),
                   image.getSizeY())
    nr_planes = image.getSizeT() * \
                image.getSizeZ() * \
                image.getSizeC()

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

