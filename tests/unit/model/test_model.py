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
    metrics_dataset.metadata_add_requirement(name='wavelength',
                                             desc='Well you bet what color this is...',
                                             type=Union[int, float],
                                             optional=True)

    return metrics_dataset


@pytest.fixture
def metrics_output():
    metrics_output = model.MetricsOutput(description='Test_output')

    shapes = [model.Point(5, 5),
              model.Line(5, 5, 12, 12),
              model.Rectangle(5, 5, 12, 12),
              model.Ellipse(5, 5, 12, 12),
              model.Polygon([(5, 5), (12, 12), (9, 9)]),
              ]

    roi = model.Roi(shapes=shapes, name='roi_name', description='roi_description')
    metrics_output.append_property(roi)

    tag1 = model.Tag(tag_value='test_tag1', name='tag1_name', description='tag1_description')
    metrics_output.append_property(tag1)

    tag2 = model.Tag(tag_value='test_tag2', name='tag2_name', description='tag2_description')
    metrics_output.append_property(tag2)

    key_values = model.KeyValues(key_values={'key': 42}, name='key_value_name', description='key_value_description')
    metrics_output.append_property(key_values)

    table = model.Table(name='table_name', description='table_description')
    metrics_output.append_property(table)

    comment = model.Comment(comment='A beautiful image', name='comment_name', description='comment_description')
    metrics_output.append_property(comment)

    return metrics_output


def test_constructor_MetricsDataset():
    metrics_dataset = model.MetricsDataset()
    assert isinstance(metrics_dataset, model.MetricsDataset)


def test_set_get_input_data(empty_input_dataset):
    empty_input_dataset.data = 5
    assert empty_input_dataset.data == 5
    empty_input_dataset.data = 42
    assert empty_input_dataset.data == 42


def test_add_remove_input_metadata_requirements(empty_input_dataset):
    empty_input_dataset.metadata_add_requirement(name='pixel size',
                                                 desc='Well you bet what this is...',
                                                 type=List[float],
                                                 optional=False)
    assert empty_input_dataset.get_metadata('pixel size') is None
    assert empty_input_dataset._metadata['pixel size']['desc'] == 'Well you bet what this is...'
    assert empty_input_dataset._metadata['pixel size']['type'] == List[float]
    assert not empty_input_dataset._metadata['pixel size']['optional']
    empty_input_dataset.metadata_remove_requirement('pixel size')
    assert len(empty_input_dataset._metadata) == 0


def test_set_get_del_metadata(filled_input_dataset):
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
    filled_input_dataset.del_metadata('wavelength')
    assert filled_input_dataset.get_metadata('wavelength') is None


def test_constructor_MetricsOutput():
    metrics_output = model.MetricsOutput('This is a description')
    assert isinstance(metrics_output, model.MetricsOutput)


def test_constructor_Roi():
    point = model.Point(5, 5)
    roi = model.Roi(shapes=[point],
                    name='ROI')
    assert isinstance(roi, model.Roi)


def test_constructor_Tag():
    tag = model.Tag(tag_value='test_tag',
                    name='123')
    assert isinstance(tag, model.Tag)


def test_constructor_KeyValues():
    key_values = model.KeyValues(key_values={'a_key': 42},
                                 name='a_name')
    assert isinstance(key_values, model.KeyValues)


def test_constructor_Table():
    table = model.Table(name='a_table')
    assert isinstance(table, model.Table)


def test_constructor_Comment():
    comment = model.Comment(comment='A beautiful image',
                            name='This is a comment')
    assert isinstance(comment, model.Comment)


def test_reading_metrics_output(metrics_output):

    assert isinstance(metrics_output.get_property('key_value_name'), model.KeyValues)

    assert len(metrics_output.get_tags()) == 2
    metrics_output.delete_property('tag2_name')
    assert len(metrics_output.get_tags()) == 1
    tag2 = model.Tag(tag_value='test_tag2', name='tag2_name', description='tag2_description')
    metrics_output.append_property(tag2)
    assert len(metrics_output.get_tags()) == 2
    tag_list = [model.Tag(tag_value='test_tag3', name='tag3_name', description='tag3_description'),
                model.Tag(tag_value='test_tag4', name='tag4_name', description='tag4_description')]
    metrics_output.extend_properties(tag_list)
    assert len(metrics_output.get_tags()) == 4

    assert f'{metrics_output.get_tags()[3].describe()}' == 'Name: tag4_name\nType: Tag\nDescription: tag4_description'

    assert len(metrics_output.get_images()) == 0
    assert len(metrics_output.get_rois()) == 1
    assert len(metrics_output.get_tags()) == 4
    assert len(metrics_output.get_key_values()) == 1
    assert len(metrics_output.get_tables()) == 1
    assert len(metrics_output.get_comments()) == 1


def test_reading_output_properties(metrics_output):
    pass

