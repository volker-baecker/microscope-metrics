"""This module takes care of the data model used by microscope-metrics.
It creates a few classes representing input data and output data
"""

import numpy as np

class MetricsImage:
    """This class represents a single image including the intensity data and the metadata.
    Instances of this class are used by the analysis routines to get the necessary data to perform the analysis"""

    def __init__(self, data: np.ndarray=None, metadata: dict={}):
        self.data = data
        self.metadata = metadata

    @property
    def data(self):
        return self.data

    @data.setter
    def set_data(self, data: np.ndarray):
        self.data = data

    @property
    def metadata(self):
        return self.metadata

    @metadata.setter
    def set_metadata(self, metadata: dict):
        self.metadata = metadata


class MetricsOutput:
    """This class is used by microscope-metrics to return the output of an analysis.
    """

    def __init__(self):
        pass

    @property


class Roi:
    pass

class Table:
    pass

class KeyValue: