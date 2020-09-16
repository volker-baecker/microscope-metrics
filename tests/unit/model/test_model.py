import pytest
from microscopemetrics.model import model
from typing import Union, List, Tuple


@pytest.fixture
def empty_input_dataset():
    class Dataset(model.MetricsDataset):
        def __init__(self):
            super().__init__()

    metrics_dataset = Dataset()
    return metrics_dataset

@pytest.fixture
def filled_input_dataset():
    class Dataset(model.MetricsDataset):
        def __init__(self):
            super().__init__()
    metrics_dataset = Dataset()
    metrics_dataset.data = [[1,2,3],[4,5,6]]
    metrics_dataset.metadata_add_requirement(name='pixel size',
                                             desc='Well you bet how big this is...',
                                             type=List[float],
                                             optional=False)
    metrics_dataset.metadata_add_requirement(name='Wavelength',
                                             desc='Well you bet what color this is...',
                                             type=Union[int, float],
                                             optional=True)

    return metrics_dataset




def test_constructor_MetricsDataset():
    metrics_dataset = model.MetricsDataset()
    assert isinstance(metrics_dataset, model.MetricsDataset)


def test_add_input_data(empty_input_dataset):
    empty_input_dataset.data = 5
    assert empty_input_dataset.data == 5
    empty_input_dataset.data = 42
    assert empty_input_dataset.data == 42


def test_add_metadata_requirements(empty_input_dataset):
    empty_input_dataset.metadata_add_requirement(name='pixel size',
                                                 desc='Well you bet what this is...',
                                                 type=List[float],
                                                 optional=False)
    assert empty_input_dataset._metadata['pixel size']['desc'] == 'Well you bet what this is...'
    assert empty_input_dataset._metadata['pixel size']['type'] == List[float]
    assert not empty_input_dataset._metadata['pixel size']['optional']



def test_set_metadata(filled_input_dataset):
    filled_input_dataset.set_metadata('pixel size', [.2, .2, .5])
    assert filled_input_dataset.get_metadata('pixel size') == [.2, .2, .5]
    filled_input_dataset.set_metadata('pixel size', [.2, .2, 2])
    assert filled_input_dataset.get_metadata('pixel size') == [.2, .2, 2]
    with pytest.raises(TypeError):
        filled_input_dataset.set_metadata('pixel size', [.2, .2, 'not'])

    filled_input_dataset.set_metadata('wavelength', 488)
    assert filled_input_dataset.get_metadata('wavelength') == 488
    filled_input_dataset.set_metadata('wavelength', 488.7)
    assert filled_input_dataset.get_metadata('wavelength') == 488.7
    with pytest.raises(TypeError):
        filled_input_dataset.set_metadata('wavelength', 'blue')




