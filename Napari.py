import napari
from vispy.color import Colormap
import numpy as np
from omero.gateway import BlitzGateway
from itertools import product

conn = BlitzGateway('mateos', 'farinato', port=4064, host="omero.mri.cnrs.fr")
conn.connect()

IMAGE_ID = 384915
image = conn.getObject("Image", IMAGE_ID)


def get_5d_stack(image):
    # We will further work with stacks of the shape TZCXY
    image_shape = (image.getSizeT(),
                   image.getSizeZ(),
                   image.getSizeC(),
                   image.getSizeX(),
                   image.getSizeY())
    zct_list = list(product(range(image_shape[1]),
                            range(image_shape[2]),
                            range(image_shape[0])))
    pixels = image.getPrimaryPixels()
    pixels_type = pixels.getPixelsType()
    data_type = pixels_type.value + str(pixels_type.bitSize)  # TODO: Verify this is working for all data types
    stack = np.zeros(image_shape, dtype=data_type)
    np.stack(pixels.getPlanes(zct_list), out=stack)

    return stack



def get_z_stack(img, c=0, t=0):
    zct_list = [(z, c, t) for z in range(img.getSizeZ())]
    pixels = image.getPrimaryPixels()
    return np.array(list(pixels.getPlanes(zct_list)))

if __name__ == '__main__':
    with napari.gui_qt():
        viewer = napari.Viewer()

        for c, channel in enumerate(image.getChannels()):
            print('loading channel %s' % c)
            data = get_z_stack(image, c=c)
            # use current rendering settings from OMERO
            color = channel.getColor().getRGB()
            color = [r/256 for r in color]
            cmap = Colormap([[0, 0, 0], color])
            # Z-scale for 3D viewing
            size_x = image.getPixelSizeX()
            size_z = image.getPixelSizeZ()
            z_scale = size_z / size_x
            viewer.add_image(data, blending='additive',
                             colormap=('from_omero', cmap),
                             scale=[1, z_scale, 1, 1],
                             name=channel.getLabel())

        print('closing conn...')
        conn.close()