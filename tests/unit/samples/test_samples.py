from typing import List, Tuple

import numpy as np

import pytest
from microscopemetrics.samples import *


@pytest.fixture
def sample_analysis():
    class MyAnalysis(Analysis):
        def __init__(self):
            description = "This is the description of the analysis class"
            super().__init__(description=description)
            self.add_requirement('pixel_sizes',
                                 'This is the physical sizes of the pixels',
                                 Tuple[float, float, float],
                                 False)
            self.add_requirement('emission_wavelengths',
                                 'Emission Wavelengths',
                                 Tuple,
                                 False)
            self.add_requirement('display_color',
                                 'This will make it rainbowy',
                                 str,
                                 True)

        @register_image_analysis
        def analyze_some_image_feature(self):
            new_array = self.input.data + 1
            new_image = model.Image(name='output_image',
                                    description="Just a sum of 1 to input data",
                                    data=new_array)
            self.output.append(new_image)

            new_point = model.Point(x=4, y=8)
            new_line = model.Line(x1=5, y1=9, x2=56, y2=33)
            new_roi = model.Roi(name='Shape',
                                description="A description for the ROI",
                                shapes=[new_point, new_line])

            self.output.append(new_roi)

            new_key_values = model.KeyValues(name="Key Values",
                                             description="A description for the key values",
                                             key_values={'resolution': 42, 'quality': "Good"})

            self.output.append(new_key_values)

    return MyAnalysis


@pytest.fixture
def sample_analysis_with_data(sample_analysis):

    sample_analysis_with_data = sample_analysis()

    sample_analysis_with_data.input.data = np.ndarray([0, 1, 2, 3, 4])

    sample_analysis_with_data.set_metadata('pixel_sizes', (.2, .2, .5))
    sample_analysis_with_data.set_metadata('emission_wavelengths', (488, 561, 642))
    sample_analysis_with_data.set_metadata('display_color', 'green')

    return sample_analysis_with_data


def test_analysis_requirements(sample_analysis_with_data):

    assert sample_analysis_with_data.verify_requirements() is True
    with pytest.raises(KeyError):
        sample_analysis_with_data.input.remove_metadata('non_existing')
    sample_analysis_with_data.empty_metadata('pixel_sizes')
    assert sample_analysis_with_data.verify_requirements() is False
    sample_analysis_with_data.set_metadata('pixel_sizes', (.2, .2, .5))
    assert sample_analysis_with_data.verify_requirements() is True


def test_analysis_inheritance(sample_analysis_with_data):

    assert isinstance(sample_analysis_with_data, Analysis)

    assert sample_analysis_with_data.verify_requirements(strict=True) is True

    IMAGE_ANALYSIS_REGISTRY['analyze_some_image_feature'](sample_analysis_with_data)

    assert len(sample_analysis_with_data.output.get_rois()) == 1

