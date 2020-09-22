from typing import List, Tuple

import pytest
from microscopemetrics.samples import samples


def test_sample_constructor():
    class MySampleType(samples.Sample):
        def __init__(self):
            super().__init__()

    my_sample = MySampleType()
    assert isinstance(my_sample, MySampleType)


def test_subclassing_analysis():
    class MyAnalysisType(samples.Analysis):
        def __init__(self):
            super().__init__()

    analysis = MyAnalysisType()
    assert isinstance(analysis, MyAnalysisType)


def test_analysis_requirements():
    class MyAnalysisType(samples.Analysis):
        def __init__(self):
            super().__init__()
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
    analysis = MyAnalysisType()
    assert analysis.validate_requirements() is False
    analysis.set_metadata('pixel_sizes', (.2, .2, .5))
    assert analysis.validate_requirements() is False
    analysis.set_metadata('emission_wavelengths', (488, 561, 642))
    assert analysis.validate_requirements() is True
    assert analysis.validate_requirements(strict=True) is False
    analysis.set_metadata('display_color', 'green')
    assert analysis.validate_requirements(strict=True) is True













