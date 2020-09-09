"""This module takes care of the data model used by microscope-metrics.
It creates a few classes representing input data and output data
"""
from abc import ABC, abstractmethod
import numpy as np

class MetricsImage:
    """This class represents a single image including the intensity data and the metadata.
    Instances of this class are used by the analysis routines to get the necessary data to perform the analysis"""

    def __init__(self, data: np.ndarray=None, metadata: dict=None):
        self.data = data
        self.metadata = metadata

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def metadata(self):
        return self.metadata

    @metadata.setter
    def metadata(self, metadata: dict):
        self.metadata = metadata


class MetricsOutput:
    """This class is used by microscope-metrics to return the output of an analysis.
    """

    def __init__(self):
        pass


class OutputFeature(ABC):
    def __init__(self, name: str, description: str = None):
        self.name = name
        self.description = description

    @abstractmethod
    def __len__(self):
        pass


class Roi(OutputFeature):
    def __init__(self,
                 name: str,
                 shapes: list,
                 description: str = None,
                 ):
        super().__init__(name, description)
        self.shapes = shapes

    def __len__(self):
        pass

    @property
    def shapes(self):
        return self._shapes

    @shapes.setter
    def shapes(self, shapes):
        self._shapes = shapes


class Shape:
    def __init__(self,
                 fill_color: tuple = (10, 10, 10, 10),
                 stroke_color: tuple = (255, 255, 255, 255),
                 stroke_width: int = 1
                 ):
        self.fill_color = fill_color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width


class Point(Shape):
    def __init__(self, x, y, z=None, c=None, t=None, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.z = z
        self.c = c
        self.t = t

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if isinstance(value, (int, float)):
            self._x = value
        else:
            raise ValueError('x position for a point must be numeric')

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if isinstance(value, (int, float)):
            self._y = value
        else:
            raise ValueError('y position for a point must be numeric')

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        if isinstance(value, (int, float)) or value is None:
            self._z = value
        else:
            raise ValueError('z position for a point must be numeric or None')

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        if isinstance(value, int) or value is None:
            self._c = value
        else:
            raise ValueError('c position for a point must be integer or None')

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        if isinstance(value, int) or value is None:
            self._t = value
        else:
            raise ValueError('t position for a point must be integer or None')


class Line(Shape):
    def __init__(self, start: tuple, end: tuple, z=None, c=None, t=None, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end
        self.z = z
        self.c = c
        self.t = t

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        if len(value) == 2 and all(isinstance(dim, (int, float)) for dim in value):
            self._start = value
        else:
            raise ValueError('start coordinates of a line must be 2 and numeric')

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        if len(value) == 2 and all(isinstance(dim, (int, float)) for dim in value):
            self._end = value
        else:
            raise ValueError('end coordinates of a line must be 2 and numeric')

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        if isinstance(value, (int, float)) or value is None:
            self._z = value
        else:
            raise ValueError('z position for a line must be numeric or None')

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        if isinstance(value, int) or value is None:
            self._c = value
        else:
            raise ValueError('c position for a line must be integer or None')

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        if isinstance(value, int) or value is None:
            self._t = value
        else:
            raise ValueError('t position for a line must be integer or None')


class Rectangle(Shape):
    pass


class Ellipse(Shape):
    pass


class Polygone_closed(Shape):
    pass


class Polygone_open(Shape):
    pass


class Mask(Shape):
    pass


class Table(OutputFeature):
    pass


class KeyValue(OutputFeature):
    pass


class Tag(OutputFeature):
    pass


class Description(OutputFeature):
    pass


class Comment(OutputFeature):
    pass