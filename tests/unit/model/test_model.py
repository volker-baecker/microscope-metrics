import pytest
from microscopemetrics.model import model


@pytest.fixture
def input_dataset():
    pass


def test_constructor_MetricsDataset():
    metrics_dataset = model.MetricsDataset()
    assert isinstance(metrics_dataset, model.MetricsDataset)