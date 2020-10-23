"""This module takes care of the data model used by microscopemetrics.
It creates a few classes representing input data and output data
"""
from abc import ABC
from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import BaseModel, validate_arguments, validator

# TODO: remove this dependency and use pydantic
from typeguard import check_type

from pandas import DataFrame
from numpy import ndarray

from typing import Union, List, Tuple, Any

# This is for future python 3.9
# class AnnotationFactory:
#     def __init__(self, type_hint):
#         self.type_hint = type_hint
#
#     def __getitem__(self, key):
#         if isinstance(key, tuple):
#             return Annotated[(self.type_hint, ) + key]
#         return Annotated[self.type_hint, key]
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}({self.type_hint})"
#
#
# Float = AnnotationFactory(float)
# Int = AnnotationFactory(int)


@dataclass
class MetricsDataset:
    """This class represents a single dataset including the intensity data and the name.
    Instances of this class are used by the analysis routines to get the necessary data to perform the analysis"""

    data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def add_metadata_requirement(self, name: str, description: str, md_type, optional: bool, units: str = None, default: Any = None):
        self.metadata[name] = {'description': description,
                               'type': md_type,
                               'optional': optional,
                               'value': default,
                               'units': units,
                               'default': default}

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
            str_output.extend([f'{k.capitalize()}: {v}' for k, v in req.items()])
        str_output.append('----------')
        str_output = '\n'.join(str_output)
        return str_output

    def verify_requirements(self, strict=False):
        validated = []
        reasons = []
        for name, req in self.metadata.items():
            v, r = self._verify_requirement(name, req, strict)
            validated.append(v)
            reasons.append(r)
        return all(validated), reasons

    @staticmethod
    def _verify_requirement(name, requirement, strict):
        if requirement['optional'] and not strict:
            return True, f'{name} is optional'
        else:
            if requirement['value'] is None:
                return False, f'{name} has None value'
            else:
                try:
                    check_type(name, requirement['value'], requirement['type'])
                    return True, f'{name} is the correct type'
                except TypeError:
                    return False, f'{name} is not of the correct type ({requirement["type"]})'

    def get_metadata_values(self, name: Union[str, list]):
        if isinstance(name, str):
            try:
                return self.metadata[name]['value']
            except KeyError as e:
                raise KeyError(f'Metadatum "{name}" does not exist') from e
        elif isinstance(name, list):
            return {k: self.get_metadata_values(k) for k in name}
        else:
            raise TypeError('get_metadata_values requires a string or list of strings')

    def get_metadata_units(self, name: Union[str, list]):
        if isinstance(name, str):
            try:
                return self.metadata[name]['units']
            except KeyError as e:
                raise KeyError(f'Metadatum "{name}" does not exist') from e
        elif isinstance(name, list):
            return {k: self.get_metadata_units(k) for k in name}
        else:
            raise TypeError('get_metadata_units requires a string or list of strings')

    def get_metadata_defaults(self, name: Union[str, list]):
        if isinstance(name, str):
            try:
                return self.metadata[name]['default']
            except KeyError as e:
                raise KeyError(f'Metadatum "{name}" does not exist') from e
        elif isinstance(name, list):
            return {k: self.get_metadata_defaults(k) for k in name}
        else:
            raise TypeError('get_metadata_units requires a string or list of strings')

    def set_metadata(self, name: str, value):
        try:
            expected_type = self.metadata[name]['type']
        except KeyError as e:
            raise KeyError(f'Metadata "{name}" is not a valid requirement') from e
        check_type(name, value, expected_type)
        self.metadata[name]['value'] = value

    def empty_metadata(self, name: str, replace_with_default: bool):
        try:
            if replace_with_default:
                self.metadata[name]['value'] = self.metadata[name]['default']
            else:
                self.metadata[name]['value'] = None
        except KeyError as e:
            raise KeyError(f'Metadata "{name}" is not a valid requirement') from e

    def del_metadata(self, name: str):
        try:
            self.metadata[name]['value'] = None
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
    z: int = field(default=None)
    c: int = field(default=None)
    t: int = field(default=None)
    fill_color: Tuple[int, int, int, int] = field(default=(10, 10, 10, 10))
    stroke_color: Tuple[int, int, int, int] = field(default=(255, 255, 255, 255))
    stroke_width: int = field(default=1)

    # Exmaple for python 3.9 annotating units
    # z: Int['z plane number'] = field(default=None)
    # c: Int['channel number'] = field(default=None)
    # t: Int['time frame'] = field(default=None)
    # fill_color: tuple[Int['red component'], Int['green component'], Int['blue component'], Int['alpha component']] =
    #             field(default=(10, 10, 10, 10))
    # stroke_color: tuple[Int['red component'], Int['green component'], Int['blue component'], Int['alpha component']] =
    #               field(default=(255, 255, 255, 255))
    # stroke_width: int  = field(default=1)


@dataclass
class Point(Shape):
    x: float = field(default=None, metadata={'units': 'PIXELS'})
    y: float = field(default=None, metadata={'units': 'PIXELS'})

    # x: Float['pixels'] = field(default=None)
    # y: Float['pixels'] = field(default=None)


@dataclass
class Line(Shape):
    x1: float = field(default=None, metadata={'units': 'PIXELS'})
    y1: float = field(default=None, metadata={'units': 'PIXELS'})
    x2: float = field(default=None, metadata={'units': 'PIXELS'})
    y2: float = field(default=None, metadata={'units': 'PIXELS'})
    # x1: Float['pixels'] = field(default=None)
    # y1: Float['pixels'] = field(default=None)
    # x2: Float['pixels'] = field(default=None)
    # y2: Float['pixels'] = field(default=None)


@dataclass
class Rectangle(Shape):
    x: float = field(default=None, metadata={'units': 'PIXELS'})
    y: float = field(default=None, metadata={'units': 'PIXELS'})
    w: float = field(default=None, metadata={'units': 'PIXELS'})
    h: float = field(default=None, metadata={'units': 'PIXELS'})
    # x: Float['pixels'] = field(default=None)
    # y: Float['pixels'] = field(default=None)
    # w: Float['pixels'] = field(default=None)
    # h: Float['pixels'] = field(default=None)


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
    # point_list: Annotated(list[tuple[float, float]], "pixels")
    # is_open: Annotated(bool, "is open")


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
        if all(isinstance(v, (str, int, float)) for _, v in k_v.items()):
            return k_v
        else:
            raise TypeError('Values for a KeyValue property must be str, int or float')


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
