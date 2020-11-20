"""This module takes care of the data model used by microscopemetrics.
It creates a few classes representing input data and output data
"""
from abc import ABC
from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import validator, BaseConfig, create_model
from pydantic.color import Color

from pandas import DataFrame
from numpy import ndarray

from typing import Union, List, Tuple, Any


class MetadataConfig(BaseConfig):
    validate_assignment = True


@dataclass
class MetricsDataset:
    """This class represents a single dataset including the intensity data and the name.
    Instances of this class are used by the analysis routines to get the necessary data to perform the analysis"""

    data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def add_metadata_requirement(self,
                                 name: str,
                                 description: str,
                                 data_type,
                                 optional: bool,
                                 units: str = None,
                                 default: Any = None,
                                 replace: bool = False):
        if not replace and name in self.metadata:
            raise KeyError(f'The key {name} is already used. Use argument replace=True to explicitly replace it')

        model = create_model(name,
                             value=(data_type, default),
                             description=(str, description),
                             optional=(bool, optional),
                             units=(str, units),
                             default=(data_type, default),
                             __config__=MetadataConfig)

        self.metadata[name] = model()
        setattr(self, name, self.metadata[name])

    def remove_metadata_requirement(self, name: str):
        try:
            del (self.metadata[name])
        except KeyError as e:
            raise KeyError(f'Metadata "{name}" does not exist') from e

    def describe_metadata_requirement(self):
        str_output = []
        for name, req in self.metadata.items():
            str_output.append('----------')
            str_output.append('Name: ' + name)
            str_output.append(req.__repr__())
        str_output.append('----------')
        str_output = '\n'.join(str_output)
        return str_output

    def list_unmet_requirements(self):
        return [req.name for _, req in self.metadata.items() if not req.optional and req.value is None]

    def validate_requirements(self):
        return all(req.value is not None for _, req in self.metadata.items() if not req.optional)

    def get_metadata_values(self, name: Union[str, list]):
        if isinstance(name, str):
            try:
                return self.metadata[name].value
            except KeyError as e:
                raise KeyError(f'Metadatum "{name}" does not exist') from e
        elif isinstance(name, list):
            return {k: self.get_metadata_values(k) for k in name}
        else:
            raise TypeError('get_metadata_values requires a string or list of strings')

    def get_metadata_units(self, name: Union[str, list]):
        if isinstance(name, str):
            try:
                return self.metadata[name].units
            except KeyError as e:
                raise KeyError(f'Metadatum "{name}" does not exist') from e
        elif isinstance(name, list):
            return {k: self.get_metadata_units(k) for k in name}
        else:
            raise TypeError('get_metadata_units requires a string or list of strings')

    def get_metadata_defaults(self, name: Union[str, list]):
        if isinstance(name, str):
            try:
                return self.metadata[name].default
            except KeyError as e:
                raise KeyError(f'Metadatum "{name}" does not exist') from e
        elif isinstance(name, list):
            return {k: self.get_metadata_defaults(k) for k in name}
        else:
            raise TypeError('get_metadata_units requires a string or list of strings')

    def set_metadata(self, name: str, value):
        try:
            self.metadata[name].value = value
        except KeyError as e:
            raise KeyError(f'Metadata "{name}" is not a valid requirement') from e

    def del_metadata(self, name: str):
        try:
            self.metadata[name].value = None
        except KeyError as e:
            raise KeyError(f'Metadata "{name}" does not exist') from e

    def list_metadata_names(self):
        return [k for k, _ in self.metadata]


@dataclass
class MetricsOutput:
    """This class is used by microscopemetrics to return the output of an analysis.
    """
    description: str = field(default=None)
    properties: dict = field(default_factory=dict, init=False)

    def get_property(self, name: str):
        return self.properties[name]

    def delete(self, name: str):
        del (self.properties[name])

    def append(self, output_property):
        if isinstance(output_property, OutputProperty):
            self.properties.update({output_property.name: output_property})
        else:
            raise TypeError('Property appended must be a subtype of OutputProperty')

    def extend(self, properties_list: list):
        for p in properties_list:
            self.append(p)

    def _get_properties_by_type(self, property_type):
        return [v for _, v in self.properties.items() if v.type == property_type]

    def get_images(self):
        return self._get_properties_by_type('Image')

    def get_rois(self):
        return self._get_properties_by_type('Roi')

    def get_tags(self):
        return self._get_properties_by_type('Tag')

    def get_key_values(self):
        return self._get_properties_by_type('KeyValues')

    def get_tables(self):
        return self._get_properties_by_type('Table')

    def get_comments(self):
        return self._get_properties_by_type('Comment')


@dataclass
class OutputProperty(ABC):
    name: str
    description: str

    def describe(self):
        """
        Returns a pretty string describing the property. Includes name, type and description.
        :return: str
        """
        return f'Name: {self.name}\n' \
               f'Type: {self.type}\n' \
               f'Description: {self.description}'

    @property
    def type(self):
        """
        Returns the type of the OutputProperty as a string.
        :return: str
        """
        return self.__class__.__name__


@dataclass
class Image(OutputProperty):
    data: Any

    @validator('data', allow_reuse=True)
    def _is_ndarray(cls, d):
        if isinstance(d, ndarray):
            return d
        else:
            raise TypeError('image output must be a Numpy ndarray')


@dataclass
class Roi(OutputProperty):
    shapes: list


@dataclass
class Shape(ABC):
    z: float = field(default=None)
    c: int = field(default=None)
    t: int = field(default=None)
    fill_color: Color = field(default=Color((10, 10, 10, 0.1)))
    stroke_color: Color = field(default=Color((255, 255, 255, 1.0)))
    stroke_width: int = field(default=1)
    label: str = field(default=None)


@dataclass
class Point(Shape):
    x: float = field(default=None, metadata={'units': 'PIXELS'})
    y: float = field(default=None, metadata={'units': 'PIXELS'})


@dataclass
class Line(Shape):
    x1: float = field(default=None, metadata={'units': 'PIXELS'})
    y1: float = field(default=None, metadata={'units': 'PIXELS'})
    x2: float = field(default=None, metadata={'units': 'PIXELS'})
    y2: float = field(default=None, metadata={'units': 'PIXELS'})


@dataclass
class Rectangle(Shape):
    x: float = field(default=None, metadata={'units': 'PIXELS'})
    y: float = field(default=None, metadata={'units': 'PIXELS'})
    w: float = field(default=None, metadata={'units': 'PIXELS'})
    h: float = field(default=None, metadata={'units': 'PIXELS'})


@dataclass
class Ellipse(Shape):
    x: float = field(default=None, metadata={'units': 'PIXELS'})
    y: float = field(default=None, metadata={'units': 'PIXELS'})
    x_rad: float = field(default=None, metadata={'units': 'PIXELS'})
    y_rad: float = field(default=None, metadata={'units': 'PIXELS'})


@dataclass
class Polygon(Shape):
    points: List[Tuple[float, float]] = field(default=None, metadata={'units': 'PIXELS'})
    is_open: bool = field(default=False)


class Mask(Shape):
    pass


@dataclass
class Tag(OutputProperty):
    tag_value: str


@dataclass
class KeyValues(OutputProperty):
    key_values: dict

    @validator('key_values', allow_reuse=True)
    def _may_be_casted_to_str(cls, k_v):
        if all(isinstance(v, (str, int, float, list, tuple)) for _, v in k_v.items()):
            return k_v
        else:
            raise TypeError('Values for a KeyValue property must be str, int, float, list or tuple')


@dataclass
class Table(OutputProperty):
    table: Any

    @validator('table', allow_reuse=True)
    def _may_be_casted_to_df(cls, t):
        if isinstance(t, (DataFrame, dict)):
            return t
        else:
            raise TypeError('table must be a pandas DataFrame or a dict')


@dataclass
class Comment(OutputProperty):
    comment: str
