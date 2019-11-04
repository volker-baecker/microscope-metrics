import napari
from vispy.color import Colormap
import numpy
from omero.gateway import BlitzGateway

conn = BlitzGateway('mateos', 'farinato', port=4064, host="omero.mri.cnrs.fr")
conn.connect()

IMAGE_ID = 384915
image = conn.getObject("Image", IMAGE_ID)


def get_z_stack(img, c=0, t=0):
    zct_list = [(z, c, t) for z in range(img.getSizeZ())]
    pixels = image.getPrimaryPixels()
    return numpy.array(list(pixels.getPlanes(zct_list)))


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